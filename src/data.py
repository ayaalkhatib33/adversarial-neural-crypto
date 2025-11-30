"""
data.py

Synthetic data generators for adversarial neural cryptography experiments.
"""

from typing import Tuple

import torch


def bits_to_pm1(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bits in {0,1} to values in {-1,+1}.

    Args:
        x: tensor of 0/1 values.

    Returns:
        Tensor of same shape with values -1 or +1.
    """
    return x * 2.0 - 1.0


def generate_symmetric_batch(
    batch_size: int,
    message_len: int,
    key_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of random messages and keys for the symmetric experiment.

    Messages and keys are sampled as random bits and then mapped to {-1,+1}.

    Returns:
        message: (batch_size, message_len)
        key:     (batch_size, key_len)
    """
    m_bits = torch.randint(0, 2, (batch_size, message_len), device=device, dtype=torch.float32)
    k_bits = torch.randint(0, 2, (batch_size, key_len), device=device, dtype=torch.float32)

    message = bits_to_pm1(m_bits)
    key = bits_to_pm1(k_bits)
    return message, key


def generate_selective_numeric_batch(
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic numeric data for the selective encryption experiment.

    We create tuples (A,B,C,D) with correlations:

        A ~ N(0,1)
        B ~ N(0,1)
        C ~ N(0,1)
        eps ~ N(0, 0.3^2)
        D = A + 0.5*B + 0.8*C + eps

    Alice sees x = (A,B,C,D).
    Bob predicts D.
    Eve predicts C (sensitive attribute).

    Returns:
        x  : (batch_size, 4)  -> (A,B,C,D)
        D  : (batch_size, 1)  -> useful target
        C  : (batch_size, 1)  -> sensitive attribute
    """
    A = torch.randn(batch_size, 1, device=device)
    B = torch.randn(batch_size, 1, device=device)
    C = torch.randn(batch_size, 1, device=device)
    eps = 0.3 * torch.randn(batch_size, 1, device=device)

    D = A + 0.5 * B + 0.8 * C + eps

    x = torch.cat([A, B, C, D], dim=1)
    return x, D, C
