# demo_image_checkpoint.py
#
# Visual demo for the image (MNIST) adversarial crypto experiment.
# Uses the trained checkpoint in results/image/alice_bob_eve.pt and shows:
#   - Original image
#   - Bob's reconstruction
#   - Eve's reconstruction

import torch
import matplotlib.pyplot as plt

from src.train_image import MLP, ConvNet, sample_mnist_batch

# ----------------------------------------------------------------------
# Global device so it's always defined when torch.load is called
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(use_best: bool = False, use_conv: bool = False) -> None:
    """
    Load and demo the trained image encryption models.
    
    Args:
        use_best: If True, load alice_bob_best.pt (best gap model during training)
                  If False, load alice_bob_eve.pt (final model)
        use_conv: If True, models were trained with ConvNet architecture
                  If False, models use MLP architecture (default)
    """
    # ------------------------------------------------------------------
    # 1) HYPERPARAMETERS – MUST MATCH train_image_crypto()
    # ------------------------------------------------------------------
    img_dim = 28 * 28      # from train_image_crypto
    key_dim = 128          # from train_image_crypto
    hidden_dim = 1024      # UPDATED: was 512, now 1024

    # ------------------------------------------------------------------
    # 2) Load checkpoint and rebuild Alice, Bob, Eve
    # ------------------------------------------------------------------
    if use_best:
        ckpt_path = "results/image/alice_bob_best.pt"
        print("Loading BEST model (highest gap during training)")
    else:
        ckpt_path = "results/image/alice_bob_eve.pt"
        print("Loading FINAL model")
    
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"\nError: Checkpoint not found at {ckpt_path}")
        print("Please run training first: python -m src.train_image")
        return

    # Build models based on architecture type
    if use_conv:
        print("Using ConvNet architecture")
        alice = ConvNet(
            input_dim=img_dim + key_dim,
            output_dim=img_dim,
        ).to(device)

        bob = ConvNet(
            input_dim=img_dim + key_dim,
            output_dim=img_dim,
        ).to(device)

        eve = ConvNet(
            input_dim=img_dim,
            output_dim=img_dim,
        ).to(device)
    else:
        print("Using MLP architecture")
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

    # Load weights
    alice.load_state_dict(ckpt["alice"])
    bob.load_state_dict(ckpt["bob"])
    
    # Eve might not be in best checkpoint
    if "eve" in ckpt:
        eve.load_state_dict(ckpt["eve"])
    else:
        print("Note: Eve weights not in checkpoint (best model only has Alice/Bob)")

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

    # Sample keys (use same distribution as training)
    key = torch.rand(batch_size, key_dim, device=device) * 2.0 - 1.0

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
    gap = mse_eve - mse_bob

    print("\n" + "="*60)
    print("IMAGE ENCRYPTION DEMO (from checkpoint)")
    print("="*60)
    print(f"Bob MSE:  {mse_bob:.4f} (lower is better)")
    print(f"Eve MSE:  {mse_eve:.4f} (higher is better)")
    print(f"Gap:      {gap:+.4f} {'✓ SUCCESS' if gap > 0 else '✗ FAILED'}")
    print()
    print("Interpretation:")
    if gap > 0.2:
        print("  ✓ Excellent! Strong encryption - Eve cannot decrypt")
    elif gap > 0.05:
        print("  ✓ Good! Eve struggles more than Bob")
    elif gap > 0:
        print("  ~ Weak encryption, but Eve is slightly worse than Bob")
    else:
        print("  ✗ Failed! Eve decrypts better than Bob (no encryption)")
    print("="*60 + "\n")

    # ------------------------------------------------------------------
    # 6) Visualize multiple examples
    # ------------------------------------------------------------------
    num_to_show = min(4, batch_size)
    
    def to_img(vec: torch.Tensor, idx: int) -> torch.Tensor:
        """Convert flattened [-1,1] vector to 28x28 [0,1] image."""
        img = (vec[idx] + 1.0) / 2.0         # [-1,1] -> [0,1]
        img = img.view(28, 28).cpu()
        return img.clamp(0.0, 1.0)

    fig, axes = plt.subplots(num_to_show, 4, figsize=(10, 2.5 * num_to_show))
    if num_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_to_show):
        orig_img = to_img(plaintext, i)
        cipher_img = to_img(ciphertext, i)
        bob_img = to_img(bob_rec, i)
        eve_img = to_img(eve_rec, i)
        
        # Compute individual MSEs
        bob_mse_i = torch.mean((bob_rec[i] - plaintext[i]) ** 2).item()
        eve_mse_i = torch.mean((eve_rec[i] - plaintext[i]) ** 2).item()

        # Original
        axes[i, 0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Original (label: {labels[i].item()})")
        axes[i, 0].axis("off")

        # Ciphertext (encrypted)
        axes[i, 1].imshow(cipher_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ciphertext (encrypted)")
        axes[i, 1].axis("off")

        # Bob's reconstruction
        axes[i, 2].imshow(bob_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Bob (MSE: {bob_mse_i:.3f})")
        axes[i, 2].axis("off")

        # Eve's reconstruction
        axes[i, 3].imshow(eve_img, cmap="gray", vmin=0, vmax=1)
        axes[i, 3].set_title(f"Eve (MSE: {eve_mse_i:.3f})")
        axes[i, 3].axis("off")

    plt.suptitle(f"Neural Cryptography Demo - Gap: {gap:+.3f}", fontsize=14, y=1.0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    use_best = "--best" in sys.argv
    use_conv = "--conv" in sys.argv
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python demo_image_checkpoint.py [--best] [--conv]")
        print("  --best: Load best model (highest gap) instead of final model")
        print("  --conv: Use ConvNet architecture (must match training)")
        print("  --help: Show this help message")
    else:
        main(use_best=use_best, use_conv=use_conv)