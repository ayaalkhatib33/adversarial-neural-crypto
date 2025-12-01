# demo_image_checkpoint.py
#
# Visual demo for the image (MNIST) adversarial crypto experiment.
# Uses the trained checkpoint in results/image/alice_bob_eve.pt and shows:
#   - Original image
#   - Bob's reconstruction
#   - Eve's reconstruction

import torch
import matplotlib.pyplot as plt

from src.train_image import MLP, sample_mnist_batch

# ----------------------------------------------------------------------
# Global device so it's always defined when torch.load is called
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    # ------------------------------------------------------------------
    # 1) HYPERPARAMETERS — MUST MATCH train_image_crypto()
    # ------------------------------------------------------------------
    img_dim = 28 * 28      # from train_image_crypto
    key_dim = 128          # from train_image_crypto
    hidden_dim = 512       # from train_image_crypto

    # ------------------------------------------------------------------
    # 2) Load checkpoint and rebuild Alice, Bob, Eve
    # ------------------------------------------------------------------
    ckpt_path = "results/image/alice_bob_eve.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    alice = MLP(
        input_dim=img_dim + key_dim,
        hidden_dim=hidden_dim,
        output_dim=img_dim,
    ).to(device)

    bob = MLP(
        input_dim=img_dim + key_dim,
        hidden_dim=hidden_dim,
        output_dim=img_dim,
    ).to(device)

    eve = MLP(
        input_dim=img_dim,
        hidden_dim=hidden_dim,
        output_dim=img_dim,
    ).to(device)

    alice.load_state_dict(ckpt["alice"])
    bob.load_state_dict(ckpt["bob"])
    eve.load_state_dict(ckpt["eve"])

    alice.eval()
    bob.eval()
    eve.eval()

    # ------------------------------------------------------------------
    # 3) Sample a small MNIST batch
    # ------------------------------------------------------------------
    batch_size = 8
    plaintext, labels = sample_mnist_batch(
        batch_size=batch_size,
        device=device,
        flatten=True,
    )
    # plaintext shape: (B, img_dim), values in [-1, 1]

    key = torch.randn(batch_size, key_dim, device=device)

    # ------------------------------------------------------------------
    # 4) Run Alice, Bob, Eve
    # ------------------------------------------------------------------
    with torch.no_grad():
        ab_input = torch.cat([plaintext, key], dim=1)
        ciphertext = alice(ab_input)

        bob_input = torch.cat([ciphertext, key], dim=1)
        bob_rec = bob(bob_input)

        eve_rec = eve(ciphertext)

    # ------------------------------------------------------------------
    # 5) Compute and print MSE
    # ------------------------------------------------------------------
    mse_bob = torch.mean((bob_rec - plaintext) ** 2).item()
    mse_eve = torch.mean((eve_rec - plaintext) ** 2).item()

    print("=== IMAGE ENCRYPTION DEMO (from checkpoint) ===")
    print(f"Bob MSE: {mse_bob:.4f}")
    print(f"Eve MSE: {mse_eve:.4f}")
    print("(Goal: Bob MSE << Eve MSE)")
    print()

    # ------------------------------------------------------------------
    # 6) Visualize one example: original vs Bob vs Eve
    # ------------------------------------------------------------------
    idx = 0  # show the first image

    def to_img(vec: torch.Tensor) -> torch.Tensor:
        """Convert flattened [-1,1] vector to 28x28 [0,1] image."""
        img = (vec[idx] + 1.0) / 2.0         # [-1,1] -> [0,1]
        img = img.view(28, 28).cpu()
        return img.clamp(0.0, 1.0)

    orig_img = to_img(plaintext)
    bob_img = to_img(bob_rec)
    eve_img = to_img(eve_rec)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes[0].imshow(orig_img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(bob_img, cmap="gray")
    axes[1].set_title("Bob reconstruction")
    axes[1].axis("off")

    axes[2].imshow(eve_img, cmap="gray")
    axes[2].set_title("Eve reconstruction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
