import torch.nn as nn


def init_linear_weights(weight: nn.Parameter) -> None:
    """
    Initialize the weights of a linear layer.
    """
    std = (2.0 / (weight.shape[0] + weight.shape[1])) ** 0.5
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)


def init_embedding_weights(weight: nn.Parameter) -> None:
    """
    Initialize the weights of an embedding layer.
    """
    nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
