import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from cs336_basics.linear import Linear


def SiLU(x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
    """
    A SiLU activation function.
    """
    return x * torch.sigmoid(x)


class SwiGLUFFN(nn.Module):
    """A SwiGLU Feed-Forward Network module."""

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Construct a SwiGLU Feed-Forward Network module. This function should accept the following parameters:
        - d_model (int): Hidden dimension of the model.
        - d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        - device (torch.device | None): Device to use for the module.
        - dtype (torch.dtype | None): dtype to use for the module.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Forward pass for the SwiGLU Feed-Forward Network module. This function should accept the following parameters:
        - x (Float[Tensor, " ... d_model"]): The input tensor to the SwiGLU Feed-Forward Network module.
        """
        assert x.shape[-1] == self.d_model, (
            f"Expected input tensor to have shape (..., {self.d_model}), but got {x.shape}"
        )

        return self.w2(SiLU(self.w1(x)) * self.w3(x))
