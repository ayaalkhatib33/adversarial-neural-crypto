# demo_symmetric_checkpoint.py
"""
Demo: use trained symmetric checkpoint to show that
- Bob reconstructs the plaintext very well
- Eve performs much worse

Run from repo root with:
    python demo_symmetric_checkpoint.py
"""

import json
import os

import numpy as np
import torch

# Import from the src package (since src has __init__.py)
from src.data import generate_symmetric_batch
from src.models import AliceSymmetric, BobSymmetric, EveSymmetric
from src.utils import get_device


CHECKPOINT_PATH = "results/symmetric/alice_bob_eve.pt"
TRAIN_LOG_PATH = "results/symmetric/training_log.json"


def pm1_to_bits(x: torch.Tensor) -> torch.Tensor:
    """
    Convert values in [-1, 1] to discrete {-1, +1} "bits"
    using a 0 threshold.
    """
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def bit_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute per-bit accuracy between predicted pm1 values and true pm1 bits.
    """
    pred_bits = pm1_to_bits(pred)
    correct = (pred_bits == target).float().mean().item()
    return correct


def load_trained_models(checkpoint_path: str, device: torch.device):
    """
    Load Alice, Bob, Eve from a training checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    message_len = ckpt["message_len"]
    key_len = ckpt["key_len"]

    alice = AliceSymmetric(message_len=message_len, key_len=key_len).to(device)
    bob = BobSymmetric(message_len=message_len, key_len=key_len).to(device)
    eve = EveSymmetric(message_len=message_len).to(device)

    alice.load_state_dict(ckpt["alice_state_dict"])
    bob.load_state_dict(ckpt["bob_state_dict"])
    eve.load_state_dict(ckpt["eve_state_dict"])

    alice.eval()
    bob.eval()
    eve.eval()

    return alice, bob, eve, message_len, key_len


def pretty_row(x: torch.Tensor) -> str:
    """
    Format a single 1D tensor row as a compact numpy array string.
    """
    arr = x.detach().cpu().numpy()
    return np.array2string(arr, precision=3, floatmode="fixed")


def run_single_demo(
    alice: AliceSymmetric,
    bob: BobSymmetric,
    eve: EveSymmetric,
    message_len: int,
    key_len: int,
    device: torch.device,
    batch_size: int = 32,
):
    """
    Draw a batch of random (plaintext, key) pairs, then:
    - Encrypt with Alice
    - Decrypt with Bob
    - Let Eve attack the ciphertext
    Print one example row + bit accuracies for the whole batch.
    """
    message, key = generate_symmetric_batch(
        batch_size=batch_size,
        message_len=message_len,
        key_len=key_len,
        device=device,
    )

    with torch.no_grad():
        ciphertext = alice(message, key)
        bob_out = bob(ciphertext, key)
        eve_out = eve(ciphertext)

    # Compute bit accuracies
    bob_acc = bit_accuracy(bob_out, message)
    eve_acc = bit_accuracy(eve_out, message)

    # Take the first example for pretty printing
    i = 0
    pt = message[i]
    ct = ciphertext[i]
    bob_i = bob_out[i]
    eve_i = eve_out[i]

    print("\n=== Symmetric Crypto Demo (from checkpoint) ===\n")
    print(f"Message length: {message_len}   Key length: {key_len}\n")

    print("Plaintext (first example):")
    print(" ", pretty_row(pt))
    print("\nCiphertext (first example):")
    print(" ", pretty_row(ct))
    print("\nBob's reconstruction (first example):")
    print(" ", pretty_row(bob_i))
    print("\nEve's reconstruction (first example):")
    print(" ", pretty_row(eve_i))

    print("\n--- Bit Accuracies over batch of size", batch_size, "---")
    print(f"Bob bit accuracy: {bob_acc * 100:.2f}%")
    print(f"Eve bit accuracy: {eve_acc * 100:.2f}%")
    print("(Goal: Bob ≈ 100%, Eve ≈ 50%)")
    print("==============================================\n")


def print_final_losses(train_log_path: str):
    """
    Optional: show the final bob_loss and eve_loss from training_log.json
    so you can mention them in your talk.
    """
    if not os.path.exists(train_log_path):
        print(f"[Note] No training log found at {train_log_path}")
        return

    with open(train_log_path, "r") as f:
        log = json.load(f)

    bob_loss = log.get("bob_loss", [])
    eve_loss = log.get("eve_loss", [])

    if bob_loss:
        print(f"Final Bob loss (MSE): {bob_loss[-1]:.4f}")
    if eve_loss:
        print(f"Final Eve loss (MSE): {eve_loss[-1]:.4f}")
    print()


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}\n")

    alice, bob, eve, message_len, key_len = load_trained_models(
        CHECKPOINT_PATH, device
    )

    run_single_demo(alice, bob, eve, message_len, key_len, device, batch_size=32)
    print_final_losses(TRAIN_LOG_PATH)
