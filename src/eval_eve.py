"""
eval_eve.py

Freeze Alice & Bob from the symmetric experiment and retrain a fresh Eve
against them to see whether she can significantly reduce her error.

Usage:
    python -m src.eval_eve --checkpoint results/symmetric/alice_bob.pt
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data import generate_symmetric_batch
from .models import AliceSymmetric, BobSymmetric, EveSymmetric
from .utils import get_device, save_json


def eval_eve(
    checkpoint_path: str,
    batch_size: int = 512,
    num_steps: int = 2000,
    lr: float = 1e-3,
) -> None:
    """
    Retrain a new Eve against a fixed Alice/Bob pair and log her loss.
    """
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)

    message_len = ckpt["message_len"]
    key_len = ckpt["key_len"]

    alice = AliceSymmetric(message_len=message_len, key_len=key_len).to(device)
    bob = BobSymmetric(message_len=message_len, key_len=key_len).to(device)
    alice.load_state_dict(ckpt["alice_state_dict"])
    bob.load_state_dict(ckpt["bob_state_dict"])
    alice.eval()
    bob.eval()

    eve = EveSymmetric(message_len=message_len).to(device)
    criterion = nn.MSELoss()
    opt_eve = optim.Adam(eve.parameters(), lr=lr)

    log: Dict[str, List[float]] = {"eve_loss": []}

    for step in tqdm(range(1, num_steps + 1), desc="Evaluating Eve"):
        message, key = generate_symmetric_batch(
            batch_size=batch_size,
            message_len=message_len,
            key_len=key_len,
            device=device,
        )
        with torch.no_grad():
            ciphertext = alice(message, key)

        eve_pred = eve(ciphertext)
        loss_eve = criterion(eve_pred, message)

        opt_eve.zero_grad()
        loss_eve.backward()
        opt_eve.step()

        log["eve_loss"].append(loss_eve.item())

    out_path = checkpoint_path.replace(".pt", "_eval_eve.json")
    save_json(log, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to alice_bob.pt checkpoint produced by train_symmetric.",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    eval_eve(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
    )
