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

    Reference: You et al., "Large Batch Training of Convolutional Networks", 2017.

    Args:
        params:         model parameters
        lr:             base learning rate (will be scaled by scheduler)
        weight_decay:   L2 regularization coefficient (default: 1e-6 per proposal)
        momentum:       SGD momentum (default: 0.9)
        eta:            LARS trust coefficient (default: 0.001)
        exclude_bias_and_bn: if True, skip LARS scaling for bias/BN params (recommended)
    """

    def __init__(self, params, lr, weight_decay=1e-6, momentum=0.9, eta=0.001,
                 exclude_bias_and_bn=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, exclude_bias_and_bn=exclude_bias_and_bn)
        super(LARS, self).__init__(params, defaults)

    @staticmethod
    def _is_bias_or_bn(param, param_name):
        """Bias and BatchNorm parameters should not be LARS-scaled or weight-decayed."""
        return param.ndim <= 1 or 'bias' in param_name or 'bn' in param_name

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for name, p in zip(
                [n for pg in self.param_groups for n in pg.get('names', [''])],
                group['params']
            ):
                if p.grad is None:
                    continue

                grad = p.grad
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                eta = group['eta']
                lr = group['lr']

                # Determine if this param should skip LARS scaling / weight decay
                skip = group['exclude_bias_and_bn'] and p.ndim <= 1

                if not skip:
                    # LARS local learning rate
                    param_norm = torch.norm(p)
                    grad_norm = torch.norm(grad)
                    # Avoid division by zero
                    if param_norm > 0 and grad_norm > 0:
                        local_lr = eta * param_norm / (grad_norm + weight_decay * param_norm)
                    else:
                        local_lr = 1.0
                    # Apply weight decay and local scaling to gradient
                    grad = grad.add(p, alpha=weight_decay)
                    grad = grad.mul(local_lr)
                # Else: apply gradient as-is (no LARS, no weight decay for bias/BN)

                # SGD with momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                p.add_(buf, alpha=-lr)

        return loss
