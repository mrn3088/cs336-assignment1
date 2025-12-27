import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from jaxtyping import Float
from cs336_basics.utils import init_linear_weights


class Linear(nn.Module):
    """
    A linear transformation module.
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        - in_features (int): The number of input features.
        - out_features (int): The number of output features.
        - device (torch.device | None): The device to use for the module.
        - dtype (torch.dtype | None): The dtype to use for the module.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        init_linear_weights(self.weight)

    def forward(self, x: Float[Tensor, " ... in_features"]) -> Float[Tensor, " ... out_features"]:
        """
        Forward pass for the linear transformation. This function should accept the following parameters:
        - x (Float[Tensor, " ... in_features"]): The input tensor to the linear transformation.
        """
        return einsum(x, self.weight, " ... in_features, out_features in_features -> ... out_features")
