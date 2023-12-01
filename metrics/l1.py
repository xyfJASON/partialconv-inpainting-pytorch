import torch
import torch.nn as nn
from torch import Tensor


def l1(X: Tensor, Y: Tensor):
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")
    if X.ndim != 4:
        raise ValueError(f"Input images should be 4-d tensors, but got {X.shape}")
    if not X.dtype == Y.dtype:
        raise ValueError(f"Input images should have the same dtype, but got {X.dtype} and {Y.dtype}.")
    return (torch.abs(X.float() - Y.float())).reshape(X.shape[0], -1).mean(dim=-1)


class L1(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        result = l1(X, Y)
        if self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        elif self.reduction == 'none':
            return result
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')
