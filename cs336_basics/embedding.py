import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int


class Embedding(nn.Module):
    """An embedding module."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an embedding module. This function should accept the following parameters:
        - num_embeddings (int): Size of the vocabulary.
        - embedding_dim (int): Dimension of the embedding vectors, i.e., d_model.
        - device (torch.device | None): Device to use for the module.
        - dtype (torch.dtype | None): dtype to use for the module.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: Int[Tensor, " ... "]) -> Float[Tensor, " ... embedding_dim"]:
        """
        Forward pass for the embedding module. This function should accept the following parameters:
        - x (Int[Tensor, " ... "]): The input tensor to the embedding module.
        """
        if x.numel() > 0:
            if x.min() < 0 or x.max() >= self.num_embeddings:
                raise ValueError(f"Token ids must be in the range [0, {self.num_embeddings - 1}]")
        return self.weight[x]
