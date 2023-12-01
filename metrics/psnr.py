import torch
import torch.nn as nn
from torch import Tensor


def mse(X: Tensor, Y: Tensor):
    return ((X.float() - Y.float()) ** 2).reshape(X.shape[0], -1).mean(dim=-1)


def psnr(X: Tensor, Y: Tensor, data_range: float = 1., epsilon: float = 1e-8):
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")
    if X.ndim != 4:
        raise ValueError(f"Input images should be 4-d tensors, but got {X.shape}")
    if not X.dtype == Y.dtype:
        raise ValueError(f"Input images should have the same dtype, but got {X.dtype} and {Y.dtype}.")
    if X.min() < 0:
        raise ValueError(f"Input images should be in range [0, {data_range}], but got {X.min().item()}")
    if Y.min() < 0:
        raise ValueError(f"Input images should be in range [0, {data_range}], but got {Y.min().item()}")
    if X.max() > data_range:
        raise ValueError(f"Input images should be in range [0, {data_range}], but got {X.max().item()}")
    if Y.max() > data_range:
        raise ValueError(f"Input images should be in range [0, {data_range}], but got {Y.max().item()}")
    return 10 * torch.log10(data_range ** 2 / (mse(X.float(), Y.float()) + epsilon))


class PSNR(nn.Module):
    def __init__(self, data_range: float = 1., epsilon: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        result = psnr(X, Y, data_range=self.data_range, epsilon=self.epsilon)
        if self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        elif self.reduction == 'none':
            return result
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')
