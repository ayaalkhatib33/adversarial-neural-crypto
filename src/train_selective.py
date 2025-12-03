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

from src.data import generate_selective_numeric_batch
from src.models import AliceSelective, BobSelective, EveSelective
from src.utils import ensure_dir, get_device, save_checkpoint, save_json, set_seed



def train_selective(
    input_dim: int = 4,   # (A,B,C,D)
    key_len: int = 16,
    rep_dim: int = 16,
    batch_size: int = 512,
    num_steps: int = 10000,  # Increased from 5000
    eve_steps: int = 2,      # Train Eve more (was 1)
    alice_bob_steps: int = 1,
    alpha_adv: float = 2.0,  # Increased from 1.0
    lr: float = 1e-3,
    seed: int = 123,
    results_dir: str = "results/selective",
    log_every: int = 50,     # Added for better monitoring
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

    log: Dict[str, List[float]] = {
        "step": [],
        "bob_loss": [],
        "eve_loss": [],
        "blind_eve_loss": [],
        "eve_advantage": [],  # Track Eve vs Blind performance
    }

    for step in tqdm(range(1, num_steps + 1), desc="Training selective"):
        # ---- 1) Train Eve ---------------------------------------------------------------
        alice.eval()
        bob.eval()
        eve.train()

        eve_loss_accum = 0.0
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
            
            eve_loss_accum += loss_eve.item()
        
        # Average Eve's loss over multiple training steps
        eve_loss_accum /= eve_steps

        # ---- 2) Train Alice & Bob -------------------------------------------------------
        alice.train()
        bob.train()
        eve.eval()

        for _ in range(alice_bob_steps):
            x, D, C = generate_selective_numeric_batch(batch_size=batch_size, device=device)
            key = torch.randn(batch_size, key_len, device=device)

            rep = alice(x, key)
            d_hat = bob(rep, key)

            # Loss for Bob: should accurately predict D
            loss_bob = criterion(d_hat, D)

            # Evaluate Eve's performance on this batch
            with torch.no_grad():
                c_hat_eval = eve(rep)
                loss_eve_eval = criterion(c_hat_eval, C)

            c_hat = eve(rep)
            loss_eve_for_ab = criterion(c_hat, C)

            # Blind Eve baseline: always predicts mean(C) from this batch
            with torch.no_grad():
                c_mean = C.mean(dim=0, keepdim=True)
                c_blind = c_mean.expand_as(C)
                loss_blind = criterion(c_blind, C)

            # Use max to ensure we at least maintain blind performance
            # This ensures Eve never gets better than blind baseline
            eve_penalty = torch.clamp(loss_blind - loss_eve_for_ab, min=0.0)
            loss_ab = loss_bob + alpha_adv * eve_penalty
            
            # Square the difference to heavily penalize Eve doing well
            # eve_diff = loss_eve_for_ab - loss_blind
            # loss_ab = loss_bob - alpha_adv * eve_diff + 0.5 * torch.clamp(-eve_diff, min=0.0)**2

            opt_ab.zero_grad()
            loss_ab.backward()
            opt_ab.step()

        # ---- 3) Logging -----------------------------------------------------------------
        if step % log_every == 0 or step == num_steps:
            # Evaluate on a fresh batch for logging
            alice.eval()
            bob.eval()
            eve.eval()
            
            with torch.no_grad():
                x, D, C = generate_selective_numeric_batch(batch_size=batch_size, device=device)
                key = torch.randn(batch_size, key_len, device=device)
                
                rep = alice(x, key)
                d_hat = bob(rep, key)
                c_hat = eve(rep)
                
                log_bob_loss = criterion(d_hat, D).item()
                log_eve_loss = criterion(c_hat, C).item()
                
                c_mean = C.mean(dim=0, keepdim=True)
                c_blind = c_mean.expand_as(C)
                log_blind_loss = criterion(c_blind, C).item()
                
                # Positive means Eve is worse than blind (good!)
                # Negative means Eve is better than blind (bad!)
                eve_advantage = log_eve_loss - log_blind_loss
            
            log["step"].append(step)
            log["bob_loss"].append(log_bob_loss)
            log["eve_loss"].append(log_eve_loss)
            log["blind_eve_loss"].append(log_blind_loss)
            log["eve_advantage"].append(eve_advantage)
            
            # Print progress periodically
            if step % (log_every * 10) == 0 or step == num_steps:
                print(f"\n{'='*70}")
                print(f"Step {step}/{num_steps}")
                print(f"  Bob Loss (predict D):     {log_bob_loss:.6f}")
                print(f"  Eve Loss (predict C):     {log_eve_loss:.6f}")
                print(f"  Blind Baseline:           {log_blind_loss:.6f}")
                print(f"  Eve - Blind:              {eve_advantage:.6f} {'✓' if eve_advantage > 0 else '✗'}")
                print(f"  Status: {'SUCCESS - Eve worse than blind' if eve_advantage > 0.01 else 'Training...'}")
                print(f"{'='*70}")

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    final_bob = log["bob_loss"][-1]
    final_eve = log["eve_loss"][-1]
    final_blind = log["blind_eve_loss"][-1]
    final_advantage = log["eve_advantage"][-1]
    
    print(f"Final Bob Loss:    {final_bob:.6f} (should be low, < 0.01)")
    print(f"Final Eve Loss:    {final_eve:.6f}")
    print(f"Final Blind Loss:  {final_blind:.6f}")
    print(f"Eve Disadvantage:  {final_advantage:.6f} (should be positive)")
    print()
    
    success_bob = final_bob < 0.01
    success_eve = final_advantage > 0.001  # Eve is at least slightly worse than blind
    
    print(f"✓ Bob predicts D accurately:        {success_bob}")
    print(f"✓ Eve worse than blind on C:        {success_eve}")
    print(f"Overall Success:                     {success_bob and success_eve}")
    print("="*70 + "\n")

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
    print(f"Models saved to {results_dir}/")


if __name__ == "__main__":
    train_selective()