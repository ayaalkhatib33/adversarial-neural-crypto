# (PASTE FROM HERE)

"""
models.py

Neural network models for adversarial neural cryptography experiments.
"""

from typing import Tuple
import torch
import torch.nn as nn


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int = 2,
    activation: nn.Module = nn.ReLU,
    out_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, output_dim))

    if out_activation is not None:
        layers.append(out_activation())

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 1. SYMMETRIC ENCRYPTION MODELS (BIT VECTORS)
# ---------------------------------------------------------------------------

class AliceSymmetric(nn.Module):
    def __init__(self, message_len: int, key_len: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = message_len + key_len
        output_dim = message_len
        self.net = make_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            out_activation=nn.Tanh,
        )

    def forward(self, message: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        x = torch.cat([message, key], dim=-1)
        return self.net(x)


class BobSymmetric(nn.Module):
    def __init__(self, message_len: int, key_len: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = message_len + key_len
        output_dim = message_len
        self.net = make_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            out_activation=nn.Tanh,
        )

    def forward(self, ciphertext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ciphertext, key], dim=-1)
        return self.net(x)


class EveSymmetric(nn.Module):
    def __init__(self, message_len: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = message_len
        output_dim = message_len
        self.net = make_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            out_activation=nn.Tanh,
        )

    def forward(self, ciphertext: torch.Tensor) -> torch.Tensor:
        return self.net(ciphertext)


# ---------------------------------------------------------------------------
# 2. SELECTIVE ENCRYPTION MODELS (NUMERIC DATA)
# ---------------------------------------------------------------------------

class AliceSelective(nn.Module):
    def __init__(self, input_dim: int, key_len: int, rep_dim: int = 16):
        super().__init__()
        self.net = make_mlp(
            input_dim=input_dim + key_len,
            hidden_dim=64,
            output_dim=rep_dim,
            out_activation=nn.Tanh,
        )

    def forward(self, x: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, key], dim=-1))


class BobSelective(nn.Module):
    def __init__(self, rep_dim: int, key_len: int):
        super().__init__()
        self.net = make_mlp(
            input_dim=rep_dim + key_len,
            hidden_dim=64,
            output_dim=1,
        )

    def forward(self, rep: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([rep, key], dim=-1))


class EveSelective(nn.Module):
    def __init__(self, rep_dim: int):
        super().__init__()
        self.net = make_mlp(
            input_dim=rep_dim,
            hidden_dim=64,
            output_dim=1,
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.net(rep)

# (TO HERE)
