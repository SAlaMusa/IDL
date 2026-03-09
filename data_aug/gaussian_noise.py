import torch


class GaussianNoise:
    """Add Gaussian noise to a tensor image.

    Operates on a float tensor with values in [0, 1].
    Output is clamped back to [0, 1].
    """

    def __init__(self, std=0.3):
        self.std = std

    def __call__(self, tensor):
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"
