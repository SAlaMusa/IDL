import torch


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer (Layer-wise Adaptive Rate Scaling).
    Used by SimCLR for self-supervised pretraining.

    Each parameter group gets a per-layer learning rate computed as:
        local_lr = eta * ||w|| / (||g|| + weight_decay * ||w||)
    and the effective update is:
        v = momentum * v + local_lr * (g + weight_decay * w)
        w = w - lr * v

    where lr is the global (base) learning rate set by the scheduler.

    Bias and BatchNorm parameters (ndim <= 1) are excluded from LARS scaling
    and weight decay — this is standard practice and matches the original paper.

    Reference: You et al., "Large Batch Training of Convolutional Networks", 2017.

    Args:
        params:              model parameters
        lr:                  base learning rate (scaled by scheduler)
        weight_decay:        L2 regularization coefficient (default: 1e-6 per proposal)
        momentum:            SGD momentum (default: 0.9)
        eta:                 LARS trust coefficient (default: 0.001)
        exclude_bias_and_bn: if True, skip LARS scaling for bias/BN params (recommended)
    """

    def __init__(self, params, lr, weight_decay=1e-6, momentum=0.9, eta=0.001,
                 exclude_bias_and_bn=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, exclude_bias_and_bn=exclude_bias_and_bn)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                eta = group['eta']
                lr = group['lr']

                # Bias and BN parameters (1-D tensors) skip LARS scaling and weight decay
                skip = group['exclude_bias_and_bn'] and p.ndim <= 1

                if not skip:
                    param_norm = torch.norm(p)
                    grad_norm = torch.norm(grad)
                    if param_norm > 0 and grad_norm > 0:
                        local_lr = eta * param_norm / (grad_norm + weight_decay * param_norm)
                    else:
                        local_lr = 1.0
                    grad = grad.add(p, alpha=weight_decay)
                    grad = grad.mul(local_lr)

                # SGD momentum update
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                p.add_(buf, alpha=-lr)

        return loss
