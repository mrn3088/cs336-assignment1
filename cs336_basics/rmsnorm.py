import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float


class RMSNorm(nn.Module):
    """A RMSNorm module."""

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Construct an RMSNorm module. This function should accept the following parameters:
        - d_model (int): Hidden dimension of the model.
        - eps (float): The epsilon value to prevent division by zero.
        - device (torch.device | None): The device to use for the module.
        - dtype (torch.dtype | None): The dtype to use for the module.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Forward pass for the RMSNorm module. This function should accept the following parameters:
        - x (Float[Tensor, " ..."]): The input tensor to the RMSNorm module.
        """
        assert x.shape[-1] == self.d_model, (
            f"Expected input tensor to have shape (..., {self.d_model}), but got {x.shape}"
        )
        # Upcast to float32 for numerical stability
        in_dtype = x.dtype
        x = x.to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x * inv_rms * self.weight
        # Downcast back to original dtype
        return x.to(in_dtype)
