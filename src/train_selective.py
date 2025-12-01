"""
train_selective.py

Selective encryption on synthetic numeric data.
Run with:

    python -m src.train_selective
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import generate_selective_numeric_batch
from models import AliceSelective, BobSelective, EveSelective
from utils import ensure_dir, get_device, save_checkpoint, save_json, set_seed


def train_selective(
    input_dim: int = 4,   # (A,B,C,D)
    key_len: int = 16,
    rep_dim: int = 16,
    batch_size: int = 512,
    num_steps: int = 5000,
    eve_steps: int = 1,
    alice_bob_steps: int = 1,
    alpha_adv: float = 1.0,
    lr: float = 1e-3,
    seed: int = 123,
    results_dir: str = "results/selective",
) -> None:
    """
    Train Alice/Bob/Eve so that:
      - Bob accurately predicts D,
      - Eve cannot predict C better than a blind baseline.
    """
    set_seed(seed)
    device = get_device()
    ensure_dir(results_dir)

    alice = AliceSelective(input_dim=input_dim, key_len=key_len, rep_dim=rep_dim).to(device)
    bob = BobSelective(rep_dim=rep_dim, key_len=key_len).to(device)
    eve = EveSelective(rep_dim=rep_dim).to(device)

    criterion = nn.MSELoss()
    opt_eve = optim.Adam(eve.parameters(), lr=lr)
    opt_ab = optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=lr)

    log: Dict[str, List[float]] = {"bob_loss": [], "eve_loss": [], "blind_eve_loss": []}

    for step in tqdm(range(1, num_steps + 1), desc="Training selective"):
        # ---- 1) Train Eve ---------------------------------------------------------------
        alice.eval()
        bob.eval()
        eve.train()

        for _ in range(eve_steps):
            x, D, C = generate_selective_numeric_batch(batch_size=batch_size, device=device)
            key = torch.randn(batch_size, key_len, device=device)

            with torch.no_grad():
                rep = alice(x, key)

            c_hat = eve(rep)
            loss_eve = criterion(c_hat, C)

            opt_eve.zero_grad()
            loss_eve.backward()
            opt_eve.step()

        # ---- 2) Train Alice & Bob -------------------------------------------------------
        alice.train()
        bob.train()
        eve.eval()

        for _ in range(alice_bob_steps):
            x, D, C = generate_selective_numeric_batch(batch_size=batch_size, device=device)
            key = torch.randn(batch_size, key_len, device=device)

            rep = alice(x, key)
            d_hat = bob(rep, key)

            loss_bob = criterion(d_hat, D)

            c_hat = eve(rep)
            loss_eve_for_ab = criterion(c_hat, C)

            # Blind Eve: always predicts mean(C)
            c_mean = C.mean(dim=0, keepdim=True)
            c_blind = c_mean.expand_as(C)
            loss_blind = criterion(c_blind, C)

            # Alice/Bob want good D and bad (worse than blind) C for Eve
            loss_ab = loss_bob - alpha_adv * (loss_eve_for_ab - loss_blind)

            opt_ab.zero_grad()
            loss_ab.backward()
            opt_ab.step()

        log["bob_loss"].append(loss_bob.item())
        log["eve_loss"].append(loss_eve.item())
        log["blind_eve_loss"].append(loss_blind.item())

    save_checkpoint(
        {
            "alice_state_dict": alice.state_dict(),
            "bob_state_dict": bob.state_dict(),
            "eve_state_dict": eve.state_dict(),
            "input_dim": input_dim,
            "key_len": key_len,
            "rep_dim": rep_dim,
        },
        f"{results_dir}/alice_bob_eve_selective.pt",
    )

    save_json(log, f"{results_dir}/training_log.json")


if __name__ == "__main__":
    train_selective()
