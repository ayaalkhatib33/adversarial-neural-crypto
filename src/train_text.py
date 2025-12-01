"""
Train the adversarial neural cryptography model on text-based data.

Usage:
    python -m src.train_text
"""

import os
import json
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class MLP(nn.Module):
    """Simple fully-connected network used for Alice, Bob, Eve."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
#  Data generation
# -----------------------------


def sample_text_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """
    Generate random character sequences (as one-hot vectors).
    Returns flattened one-hot representation of shape (batch_size, seq_len * vocab_size).
    """
    # random integers [0, vocab_size)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    one_hot = F.one_hot(tokens, num_classes=vocab_size).float()  # (B, L, V)
    return one_hot.view(batch_size, -1)  # (B, L * V)


def sample_keys(batch_size: int, key_dim: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, key_dim, device=device) * 2.0 - 1.0


def compute_metrics(
    plaintext: torch.Tensor,
    bob_out: torch.Tensor,
    eve_out: torch.Tensor,
    sensitive_mask: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        mse_bob = F.mse_loss(bob_out, plaintext).item()
        mse_eve = F.mse_loss(eve_out, plaintext).item()

        pt_sensitive = plaintext * sensitive_mask
        bob_sensitive = bob_out * sensitive_mask
        eve_sensitive = eve_out * sensitive_mask

        mse_bob_sensitive = F.mse_loss(bob_sensitive, pt_sensitive).item()
        mse_eve_sensitive = F.mse_loss(eve_sensitive, pt_sensitive).item()

        pt_nonsensitive = plaintext * (1.0 - sensitive_mask)
        bob_nonsensitive = bob_out * (1.0 - sensitive_mask)
        eve_nonsensitive = eve_out * (1.0 - sensitive_mask)

        mse_bob_nonsensitive = F.mse_loss(bob_nonsensitive, pt_nonsensitive).item()
        mse_eve_nonsensitive = F.mse_loss(eve_nonsensitive, pt_nonsensitive).item()

    return {
        "bob_mse": mse_bob,
        "eve_mse": mse_eve,
        "bob_mse_sensitive": mse_bob_sensitive,
        "eve_mse_sensitive": mse_eve_sensitive,
        "bob_mse_nonsensitive": mse_bob_nonsensitive,
        "eve_mse_nonsensitive": mse_eve_nonsensitive,
    }


def train_text_crypto(
    num_steps: int = 2000,
    batch_size: int = 128,
    seq_len: int = 16,
    vocab_size: int = 32,
    hidden_dim: int = 512,
    key_dim: int = 128,
    eve_weight: float = 1.0,
    lr: float = 1e-3,
    log_every: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Train Alice, Bob, and Eve on synthetic text data.

    Saves:
      - results/text/alice_bob.pt
      - results/text/alice_bob_eve.pt
      - results/text/training_log.json
    """
    device = torch.device(device)
    os.makedirs("results/text", exist_ok=True)

    message_dim = seq_len * vocab_size
    ciphertext_dim = message_dim

    alice = MLP(message_dim + key_dim, hidden_dim, ciphertext_dim).to(device)
    bob = MLP(ciphertext_dim + key_dim, hidden_dim, message_dim).to(device)
    eve = MLP(ciphertext_dim, hidden_dim, message_dim).to(device)

    opt_ab = torch.optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=lr)
    opt_eve = torch.optim.Adam(eve.parameters(), lr=lr)

    sensitive_mask = torch.zeros(message_dim, device=device)
    sensitive_mask[: message_dim // 2] = 1.0

    log: List[Dict] = []

    for step in tqdm(range(1, num_steps + 1), desc="Text crypto training"):
        plaintext = sample_text_batch(batch_size, seq_len, vocab_size, device)
        keys = sample_keys(batch_size, key_dim, device)

        # ---------- Train Eve ----------
        alice.eval()
        bob.eval()
        eve.train()

        with torch.no_grad():
            alice_in = torch.cat([plaintext, keys], dim=1)
            ciphertext = alice(alice_in)

        eve_out = eve(ciphertext)
        loss_eve = F.mse_loss(eve_out, plaintext)

        opt_eve.zero_grad()
        loss_eve.backward()
        opt_eve.step()

        # ---------- Train Alice + Bob ----------
        alice.train()
        bob.train()
        eve.eval()

        alice_in = torch.cat([plaintext, keys], dim=1)
        ciphertext = alice(alice_in)

        bob_in = torch.cat([ciphertext, keys], dim=1)
        bob_out = bob(bob_in)

        with torch.no_grad():
            eve_out = eve(ciphertext)

        loss_bob = F.mse_loss(bob_out, plaintext)
        loss_eve_detached = F.mse_loss(eve_out, plaintext)

        loss_ab = loss_bob - eve_weight * loss_eve_detached

        opt_ab.zero_grad()
        loss_ab.backward()
        opt_ab.step()

        if step % log_every == 0 or step == num_steps:
            metrics = compute_metrics(plaintext, bob_out, eve_out, sensitive_mask)
            metrics["step"] = step
            log.append(metrics)

    torch.save({"alice": alice.state_dict(), "bob": bob.state_dict()}, "results/text/alice_bob.pt")
    torch.save(
        {"alice": alice.state_dict(), "bob": bob.state_dict(), "eve": eve.state_dict()},
        "results/text/alice_bob_eve.pt",
    )

    with open("results/text/training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("Text training finished. Models and log saved in results/text.")


if __name__ == "__main__":
    train_text_crypto()
