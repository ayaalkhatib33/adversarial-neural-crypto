# plot_symmetric_training.py
"""
Plot Bob vs Eve loss over training using results/symmetric/training_log.json

Run from repo root with:
    python plot_symmetric_training.py
"""

import json
import os

import matplotlib.pyplot as plt


TRAIN_LOG_PATH = "results/symmetric/training_log.json"
OUTPUT_FIG_PATH = "results/symmetric/symmetric_losses.png"


def main():
    if not os.path.exists(TRAIN_LOG_PATH):
        raise FileNotFoundError(f"Could not find {TRAIN_LOG_PATH}")

    with open(TRAIN_LOG_PATH, "r") as f:
        log = json.load(f)

    bob_loss = log.get("bob_loss", [])
    eve_loss = log.get("eve_loss", [])

    if not bob_loss:
        raise ValueError("No 'bob_loss' found in training_log.json")
    if not eve_loss:
        print("[Warning] No 'eve_loss' in log – plotting Bob only.")
        eve_loss = None

    steps = list(range(1, len(bob_loss) + 1))

    plt.figure()
    plt.plot(steps, bob_loss, label="Bob loss (reconstruction)", linewidth=2)
    if eve_loss is not None:
        plt.plot(steps, eve_loss, label="Eve loss (attack)", linewidth=2)

    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("Symmetric Neural Crypto: Bob vs Eve Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_FIG_PATH), exist_ok=True)
    plt.savefig(OUTPUT_FIG_PATH)
    print(f"Saved plot to {OUTPUT_FIG_PATH}")


    plt.show()


if __name__ == "__main__":
    main()
