import copy
import logging
import os

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import save_config_file, accuracy, save_checkpoint

class MoCo(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.encoder_q = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        run_name = getattr(self.args, 'run_name', None)
        self.writer = SummaryWriter(log_dir=run_name)
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, 'training.log'),
            level=logging.DEBUG
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        # Key encoder: EMA copy of query encoder, no gradients
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Queue: shape (out_dim, queue_size), L2-normalised
        self.queue_size = self.args.moco_queue_size
        self.momentum   = self.args.moco_momentum
        queue = F.normalize(
            torch.randn(self.args.out_dim, self.queue_size, device=self.args.device), dim=0
        )
        self.queue     = queue
        self.queue_ptr = torch.zeros(1, dtype=torch.long, device=self.args.device)

    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data = self.momentum * pk.data + (1.0 - self.momentum) * pq.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            # wrap around
            overflow = end - self.queue_size
            self.queue[:, ptr:]     = keys[:batch_size - overflow].T
            self.queue[:, :overflow] = keys[batch_size - overflow:].T
        self.queue_ptr[0] = end % self.queue_size

    def moco_loss(self, im_q, im_k):
        q = F.normalize(self.encoder_q(im_q), dim=1)          # (N, dim)

        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.encoder_k(im_k), dim=1)      # (N, dim)

        # (N, 1) positive + (N, K) negatives from queue
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())

        logits = torch.cat([l_pos, l_neg], dim=1) / self.args.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.args.device)

        self._dequeue_and_enqueue(k)
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler('cuda', enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start MoCo v2 training for {self.args.epochs} epochs.")
        logging.info(f"Queue size: {self.queue_size}  Momentum: {self.momentum}")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                im_q = images[0].to(self.args.device)
                im_k = images[1].to(self.args.device)

                with autocast('cuda', enabled=self.args.fp16_precision):
                    logits, labels = self.moco_loss(im_q, im_k)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            warmup_epochs = getattr(self.args, 'warmup_epochs', 10)
            if epoch_counter < warmup_epochs:
                warmup_scale = (epoch_counter + 1) / warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.args.lr * warmup_scale
            else:
                self.scheduler.step()

            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1: {top1[0]}")

            ep = epoch_counter + 1
            if ep % 200 == 0 and ep < self.args.epochs:
                ckpt_name = 'checkpoint_{:04d}.pth.tar'.format(ep)
                save_checkpoint({
                    'epoch': ep,
                    'arch':  self.args.arch,
                    'state_dict': self.encoder_q.state_dict(),
                    'optimizer':  self.optimizer.state_dict(),
                }, filename=os.path.join(self.writer.log_dir, ckpt_name))

        logging.info("Training has finished.")
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch':  self.args.arch,
            'state_dict': self.encoder_q.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
        }, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint saved at {self.writer.log_dir}.")
