"""
train_symmetric.py

Train Alice, Bob, and Eve on random bit vectors (symmetric encryption).
Run with:

    python -m src.train_symmetric
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import generate_symmetric_batch
from src.models import AliceSymmetric, BobSymmetric, EveSymmetric
from src.utils import ensure_dir, get_device, save_checkpoint, save_json, set_seed



def train_symmetric(
    message_len: int = 16,
    key_len: int = 16,
    batch_size: int = 512,
    num_steps: int = 5000,
    eve_steps: int = 1,
    alice_bob_steps: int = 1,
    alpha_adv: float = 1.0,
    lr: float = 1e-3,
    seed: int = 42,
    results_dir: str = "results/symmetric",
) -> None:
    """
    Adversarial training loop for symmetric neural cryptography.

    Alice & Bob want Bob's reconstruction error small and Eve's error large.
    Eve wants to minimize her reconstruction error.
    """
    set_seed(seed)
    device = get_device()
    ensure_dir(results_dir)

    alice = AliceSymmetric(message_len=message_len, key_len=key_len).to(device)
    bob = BobSymmetric(message_len=message_len, key_len=key_len).to(device)
    eve = EveSymmetric(message_len=message_len).to(device)

    criterion = nn.MSELoss()

    opt_eve = optim.Adam(eve.parameters(), lr=lr)
    opt_ab = optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=lr)

    log: Dict[str, List[float]] = {"bob_loss": [], "eve_loss": []}

    for step in tqdm(range(1, num_steps + 1), desc="Training symmetric"):
        # ---- 1) Update Eve ----------------------------------------------------------------
        alice.eval()
        bob.eval()
        eve.train()

        for _ in range(eve_steps):
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

        # ---- 2) Update Alice & Bob (adversarially) ---------------------------------------
        alice.train()
        bob.train()
        eve.eval()

        for _ in range(alice_bob_steps):
            message, key = generate_symmetric_batch(
                batch_size=batch_size,
                message_len=message_len,
                key_len=key_len,
                device=device,
            )
            ciphertext = alice(message, key)
            bob_pred = bob(ciphertext, key)

            loss_bob = criterion(bob_pred, message)

            eve_pred = eve(ciphertext)
            loss_eve_for_ab = criterion(eve_pred, message)

            # Alice & Bob minimize Bob loss and *maximize* Eve loss
            loss_ab = loss_bob - alpha_adv * loss_eve_for_ab

            opt_ab.zero_grad()
            loss_ab.backward()
            opt_ab.step()

        log["bob_loss"].append(loss_bob.item())
        log["eve_loss"].append(loss_eve.item())

    # ---- Save models and training log ----------------------------------------------------
    save_checkpoint(
        {
            "alice_state_dict": alice.state_dict(),
            "bob_state_dict": bob.state_dict(),
            "eve_state_dict": eve.state_dict(),
            "message_len": message_len,
            "key_len": key_len,
        },
        f"{results_dir}/alice_bob_eve.pt",
    )

    save_checkpoint(
        {
            "alice_state_dict": alice.state_dict(),
            "bob_state_dict": bob.state_dict(),
            "message_len": message_len,
            "key_len": key_len,
        },
        f"{results_dir}/alice_bob.pt",
    )

    save_json(log, f"{results_dir}/training_log.json")


if __name__ == "__main__":
    print("Starting train_symmetric main...")
    try:
        train_symmetric()
        print("Finished train_symmetric without error.")
    except Exception as e:
        import traceback
        print("ERROR during training:", e)
        traceback.print_exc()
