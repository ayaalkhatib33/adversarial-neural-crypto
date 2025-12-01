# src/train_image.py
"""
Adversarial neural cryptography on image data (MNIST).

Based on "Learning to Protect Communications with Adversarial Neural Cryptography"
(Abadi & Andersen, 2016)

Alice: encrypts flattened image + key
Bob:   decrypts ciphertext + key
Eve:   tries to reconstruct image from ciphertext only
"""

import os
import json
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# our imports (fixed)
from src.utils import *


# -----------------------------
#  Model definitions
# -----------------------------

def sample_mnist_batch(
    batch_size: int,
    device: torch.device,
    flatten: bool = True,
):
    """
    Return a single batch of MNIST images in [-1, 1] range, optionally flattened.
    """
    global _mnist_loader, _mnist_iter

    if "_mnist_loader" not in globals():
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )

        _mnist_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        _mnist_iter = iter(_mnist_loader)

    try:
        images, labels = next(_mnist_iter)
    except StopIteration:
        _mnist_iter = iter(_mnist_loader)
        images, labels = next(_mnist_iter)

    images = images.to(device)

    if flatten:
        images = images.view(images.size(0), -1)

    images = images * 2.0 - 1.0
    return images, labels


class ConvNet(nn.Module):
    """Convolutional network - better for learning encryption patterns."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Reshape flat input to "image" for convolutions
        # Use 4 channels of sqrt(input_dim/4) x sqrt(input_dim/4)
        self.channels = 4
        self.spatial_dim = int((input_dim / self.channels) ** 0.5)
        
        self.net = nn.Sequential(
            # First conv block
            nn.Conv1d(input_dim, 256, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Second conv block
            nn.Conv1d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Third conv block
            nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Flatten handled in forward
        )
        
        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, 1)
            dummy_out = self.net(dummy)
            self.flat_size = dummy_out.view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        x = x.unsqueeze(2)  # (batch, features, 1) for Conv1d
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    """Fallback MLP if ConvNet has issues."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_mnist_dataloader(batch_size: int = 128) -> DataLoader:
    """Returns a DataLoader over MNIST images, normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def sample_keys(batch_size: int, key_dim: int, device: torch.device) -> torch.Tensor:
    """Sample random keys in [-1, 1]."""
    return torch.rand(batch_size, key_dim, device=device) * 2.0 - 1.0


def compute_metrics(
    plaintext: torch.Tensor,
    bob_out: torch.Tensor,
    eve_out: torch.Tensor,
    sensitive_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute reconstruction errors for Bob and Eve (overall and selective)."""
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


def train_image_crypto(
    num_steps: int = 10000,      # Start with 10k, increase if needed
    batch_size: int = 128,       # Standard batch size
    hidden_dim: int = 1024,
    key_dim: int = 128,
    lr: float = 1e-4,            # Single lower learning rate
    log_every: int = 100,
    eve_training_ratio: int = 1,  # Train Eve equally
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_conv: bool = False,      # Set to True to try ConvNet
) -> None:
    """
    Train Alice, Bob, and Eve on MNIST with corrected loss from the paper.
    """
    device = torch.device(device)
    os.makedirs("results/image", exist_ok=True)

    loader = make_mnist_dataloader(batch_size=batch_size)
    data_iter = iter(loader)

    img_dim = 28 * 28
    message_dim = img_dim
    ciphertext_dim = img_dim

    # Create models
    if use_conv:
        print("Using Convolutional networks")
        alice = ConvNet(message_dim + key_dim, ciphertext_dim).to(device)
        bob = ConvNet(ciphertext_dim + key_dim, message_dim).to(device)
        eve = ConvNet(ciphertext_dim, message_dim).to(device)
    else:
        print("Using MLP networks")
        alice = MLP(message_dim + key_dim, hidden_dim, ciphertext_dim).to(device)
        bob = MLP(ciphertext_dim + key_dim, hidden_dim, message_dim).to(device)
        eve = MLP(ciphertext_dim, hidden_dim, message_dim).to(device)

    # Single optimizer for all (as in original paper)
    opt_all = torch.optim.Adam(
        list(alice.parameters()) + list(bob.parameters()) + list(eve.parameters()),
        lr=lr
    )

    sensitive_mask = torch.zeros(message_dim, device=device)
    sensitive_mask[: message_dim // 2] = 1.0

    log: List[Dict] = []
    best_gap = -float('inf')

    print(f"Starting training for {num_steps} steps...")
    print(f"Target: Bob MSE < 0.05, Eve MSE > 0.3, Gap > 0.2")
    print(f"If not converged by step {num_steps}, increase num_steps")
    print("="*70)

    for step in tqdm(range(1, num_steps + 1), desc="Image crypto training"):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, _ = next(data_iter)

        images = images.to(device)
        plaintext = images.view(batch_size, -1)
        keys = sample_keys(batch_size, key_dim, device)

        # Forward pass through all networks
        alice.train()
        bob.train()
        eve.train()

        alice_in = torch.cat([plaintext, keys], dim=1)
        ciphertext = alice(alice_in)

        bob_in = torch.cat([ciphertext, keys], dim=1)
        bob_out = bob(bob_in)

        eve_out = eve(ciphertext)

        # Compute losses
        loss_bob = F.mse_loss(bob_out, plaintext)
        loss_eve = F.mse_loss(eve_out, plaintext)

        # THE KEY FIX: Loss from the original paper
        # L_{A,B} = loss_bob + (1 - loss_eve)^2
        # This means:
        # - Minimize Bob's error (first term)
        # - Maximize Eve's error toward 1.0, which is random guessing for [-1,1] data
        # - The squared term heavily penalizes when Eve does well (loss_eve < 1.0)
        
        # For normalized data in [-1, 1], random guessing gives MSE ≈ 1.0
        # We want Eve's loss to be close to 1.0 (random performance)
        target_eve_loss = 1.0
        
        # Original paper formulation
        loss_ab = loss_bob + (target_eve_loss - loss_eve) ** 2
        
        # Eve tries to minimize her error
        loss_total = loss_ab + loss_eve

        # Single backward pass
        opt_all.zero_grad()
        loss_total.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(alice.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(bob.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(eve.parameters(), max_norm=1.0)
        
        opt_all.step()

        # Logging
        if step % log_every == 0 or step == num_steps:
            alice.eval()
            bob.eval()
            eve.eval()
            
            with torch.no_grad():
                # Fresh evaluation
                eval_keys = sample_keys(batch_size, key_dim, device)
                alice_in = torch.cat([plaintext, eval_keys], dim=1)
                ciphertext = alice(alice_in)
                bob_in = torch.cat([ciphertext, eval_keys], dim=1)
                bob_out = bob(bob_in)
                eve_out = eve(ciphertext)
                
            metrics = compute_metrics(plaintext, bob_out, eve_out, sensitive_mask)
            metrics["step"] = step
            
            gap = metrics["eve_mse"] - metrics["bob_mse"]
            metrics["gap"] = gap
            
            log.append(metrics)
            
            # Save best model
            if gap > best_gap:
                best_gap = gap
                torch.save(
                    {"alice": alice.state_dict(), "bob": bob.state_dict()},
                    "results/image/alice_bob_best.pt"
                )
            
            # Print progress
            if step % (log_every * 5) == 0 or step == num_steps:
                success = "✓" if gap > 0 else "✗"
                print(f"\nStep {step:5d}: Bob={metrics['bob_mse']:.4f}, "
                      f"Eve={metrics['eve_mse']:.4f}, Gap={gap:+.4f} {success}")

    # Save final models
    torch.save({"alice": alice.state_dict(), "bob": bob.state_dict()}, "results/image/alice_bob.pt")
    torch.save(
        {"alice": alice.state_dict(), "bob": bob.state_dict(), "eve": eve.state_dict()},
        "results/image/alice_bob_eve.pt",
    )

    with open("results/image/training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final Bob MSE:        {log[-1]['bob_mse']:.4f} (target: < 0.05)")
    print(f"Final Eve MSE:        {log[-1]['eve_mse']:.4f} (target: > 0.3)")
    print(f"Final Gap (Eve-Bob):  {log[-1]['gap']:+.4f} (target: > 0.2)")
    print(f"Best Gap achieved:    {best_gap:+.4f}")
    print(f"\nSuccess: {log[-1]['gap'] > 0.2}")
    if log[-1]['gap'] < 0.2:
        print("Tip: Try increasing num_steps to 15000 or 20000")
    print("="*70)


if __name__ == "__main__":
    train_image_crypto()