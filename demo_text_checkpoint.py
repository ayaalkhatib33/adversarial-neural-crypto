# demo_text_checkpoint.py
"""
Demo: use trained text crypto checkpoint to show that
- Bob reconstructs the plaintext better than Eve

Run from repo root with:
    python demo_text_checkpoint.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import get_device
from src.train_text import MLP, sample_text_batch, sample_keys

# Use the checkpoint that includes Alice, Bob *and* Eve
CKPT_PATH = "results/text/alice_bob_eve.pt"

# These must match the hyperparameters used in train_text_crypto()
SEQ_LEN = 16
VOCAB_SIZE = 32
HIDDEN_DIM = 512
KEY_DIM = 128


def main():
    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    if "alice" not in ckpt or "bob" not in ckpt:
        raise KeyError(f"Checkpoint at {CKPT_PATH} must contain 'alice' and 'bob' keys.")

    message_dim = SEQ_LEN * VOCAB_SIZE
    ciphertext_dim = message_dim

    # Recreate models exactly as in train_text_crypto()
    alice = MLP(message_dim + KEY_DIM, HIDDEN_DIM, ciphertext_dim).to(device)
    bob = MLP(ciphertext_dim + KEY_DIM, HIDDEN_DIM, message_dim).to(device)
    eve = MLP(ciphertext_dim, HIDDEN_DIM, message_dim).to(device)

    alice.load_state_dict(ckpt["alice"])
    bob.load_state_dict(ckpt["bob"])
    if "eve" in ckpt:
        eve.load_state_dict(ckpt["eve"])
    else:
        print("[Warning] 'eve' not found in checkpoint; Eve will be randomly initialized.")

    alice.eval()
    bob.eval()
    eve.eval()

    # Sample a batch of random text + keys
    batch_size = 32
    plaintext = sample_text_batch(
        batch_size=batch_size,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        device=device,
    )
    keys = sample_keys(batch_size=batch_size, key_dim=KEY_DIM, device=device)

    with torch.no_grad():
        alice_in = torch.cat([plaintext, keys], dim=1)
        ciphertext = alice(alice_in)

        bob_in = torch.cat([ciphertext, keys], dim=1)
        bob_out = bob(bob_in)

        eve_out = eve(ciphertext)

    # Compute MSE (same idea as training metrics)
    mse_bob = F.mse_loss(bob_out, plaintext).item()
    mse_eve = F.mse_loss(eve_out, plaintext).item()

    print("\n=== TEXT ENCRYPTION DEMO (from checkpoint) ===\n")
    print(f"Bob MSE: {mse_bob:.4f}")
    print(f"Eve MSE: {mse_eve:.4f}")
    print("(Goal: Bob MSE << Eve MSE)\n")

    # Show the first example for intuition
    i = 0
    pt = plaintext[i].cpu().numpy()
    bob_i = bob_out[i].cpu().numpy()
    eve_i = eve_out[i].cpu().numpy()

    # Print compact vectors (they're long: seq_len * vocab_size)
    print("Plaintext (first example, flattened one-hot):")
    print(np.array2string(pt, precision=2, floatmode="fixed"))

    print("\nBob reconstruction (first example):")
    print(np.array2string(bob_i, precision=2, floatmode="fixed"))

    print("\nEve reconstruction (first example):")
    print(np.array2string(eve_i, precision=2, floatmode="fixed"))

    print("\n==============================================\n")


if __name__ == "__main__":
    main()
