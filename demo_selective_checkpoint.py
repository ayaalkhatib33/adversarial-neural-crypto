# demo_selective_checkpoint.py
#
# Demo for the selective encryption experiment.
# Uses the trained checkpoint in results/selective/alice_bob_eve_selective.pt
#
# Shows:
#   - Original data (A, B, C, D)
#   - Bob's prediction of D (with key) - should be accurate
#   - Eve's prediction of C (without key) - should be no better than blind baseline

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data import generate_selective_numeric_batch
from src.models import AliceSelective, BobSelective, EveSelective

# ----------------------------------------------------------------------
# Global device
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """
    Load and demo the trained selective encryption models.
    
    Goal: Bob predicts D accurately, Eve cannot predict C better than blind guessing.
    """
    # ------------------------------------------------------------------
    # 1) HYPERPARAMETERS – MUST MATCH train_selective()
    # ------------------------------------------------------------------
    input_dim = 4      # (A, B, C, D)
    key_len = 16       # from train_selective
    rep_dim = 16       # from train_selective

    # ------------------------------------------------------------------
    # 2) Load checkpoint and rebuild Alice, Bob, Eve
    # ------------------------------------------------------------------
    ckpt_path = "results/selective/alice_bob_eve_selective.pt"
    
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"\nError: Checkpoint not found at {ckpt_path}")
        print("Please run training first: python -m src.train_selective")
        return

    alice = AliceSelective(
        input_dim=input_dim,
        key_len=key_len,
        rep_dim=rep_dim,
    ).to(device)

    bob = BobSelective(
        rep_dim=rep_dim,
        key_len=key_len,
    ).to(device)

    eve = EveSelective(
        rep_dim=rep_dim,
    ).to(device)

    # Load weights
    alice.load_state_dict(ckpt["alice_state_dict"])
    bob.load_state_dict(ckpt["bob_state_dict"])
    eve.load_state_dict(ckpt["eve_state_dict"])

    alice.eval()
    bob.eval()
    eve.eval()

    # ------------------------------------------------------------------
    # 3) Generate test data
    # ------------------------------------------------------------------
    batch_size = 100
    x, D, C = generate_selective_numeric_batch(batch_size=batch_size, device=device)
    # x: (batch, 4) containing [A, B, C, D]
    # D: (batch, 1) - the target Bob should predict
    # C: (batch, 1) - the sensitive value Eve should NOT predict

    key = torch.randn(batch_size, key_len, device=device)

    # ------------------------------------------------------------------
    # 4) Run Alice, Bob, Eve
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Alice encrypts
        rep = alice(x, key)
        
        # Bob decrypts to predict D (with key)
        d_hat = bob(rep, key)
        
        # Eve tries to predict C (without key)
        c_hat = eve(rep)
        
        # Blind baseline: always predict mean(C)
        c_mean = C.mean(dim=0, keepdim=True)
        c_blind = c_mean.expand_as(C)

    # ------------------------------------------------------------------
    # 5) Compute metrics
    # ------------------------------------------------------------------
    mse_bob = torch.mean((d_hat - D) ** 2).item()
    mse_eve = torch.mean((c_hat - C) ** 2).item()
    mse_blind = torch.mean((c_blind - C) ** 2).item()
    
    eve_advantage = mse_eve - mse_blind  # Positive is good (Eve worse than blind)

    print("\n" + "="*70)
    print("SELECTIVE ENCRYPTION DEMO")
    print("="*70)
    print("\nGoal:")
    print("  ✓ Bob (with key) should accurately predict D")
    print("  ✓ Eve (without key) should NOT predict C better than blind guessing")
    print("\nResults:")
    print(f"  Bob's MSE (predicting D):      {mse_bob:.6f} (target: < 0.01)")
    print(f"  Eve's MSE (predicting C):      {mse_eve:.6f}")
    print(f"  Blind Baseline MSE:            {mse_blind:.6f}")
    print(f"  Eve - Blind:                   {eve_advantage:+.6f} {'✓' if eve_advantage > 0 else '✗'}")
    print()
    print("Interpretation:")
    
    bob_success = mse_bob < 0.01
    eve_success = eve_advantage > 0.001
    
    if bob_success and eve_success:
        print("  ✓✓ SUCCESS! Selective encryption working perfectly!")
        print("     - Bob accurately recovers D")
        print("     - Eve cannot predict C better than blind guessing")
    elif bob_success and not eve_success:
        print("  ✓✗ Partial: Bob works but encryption failed")
        print("     - Bob accurately recovers D")
        print("     - Eve CAN predict C (encryption too weak)")
    elif not bob_success and eve_success:
        print("  ✗✓ Partial: Encryption works but Bob struggles")
        print("     - Bob cannot accurately recover D")
        print("     - Eve successfully blocked from C")
    else:
        print("  ✗✗ FAILED on both objectives")
    
    print("="*70 + "\n")

    # ------------------------------------------------------------------
    # 6) Detailed visualization
    # ------------------------------------------------------------------
    # Convert to numpy for easier plotting
    x_np = x.cpu().numpy()
    D_np = D.cpu().numpy().flatten()
    C_np = C.cpu().numpy().flatten()
    d_hat_np = d_hat.cpu().numpy().flatten()
    c_hat_np = c_hat.cpu().numpy().flatten()
    c_blind_np = c_blind.cpu().numpy().flatten()

    # Create comprehensive visualization
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Bob's predictions (D)
    ax1 = plt.subplot(3, 2, 1)
    ax1.scatter(D_np, d_hat_np, alpha=0.5, s=20)
    ax1.plot([D_np.min(), D_np.max()], [D_np.min(), D_np.max()], 'r--', label='Perfect prediction')
    ax1.set_xlabel('True D')
    ax1.set_ylabel("Bob's prediction")
    ax1.set_title(f"Bob's Performance (MSE: {mse_bob:.4f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bob's errors
    ax2 = plt.subplot(3, 2, 2)
    bob_errors = np.abs(d_hat_np - D_np)
    ax2.hist(bob_errors, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f"Bob's Error Distribution (Mean: {bob_errors.mean():.4f})")
    ax2.grid(True, alpha=0.3)
    
    # 3. Eve's predictions (C)
    ax3 = plt.subplot(3, 2, 3)
    ax3.scatter(C_np, c_hat_np, alpha=0.5, s=20, label='Eve')
    ax3.scatter(C_np, c_blind_np, alpha=0.5, s=20, label='Blind baseline', marker='x')
    ax3.plot([C_np.min(), C_np.max()], [C_np.min(), C_np.max()], 'r--', label='Perfect prediction')
    ax3.set_xlabel('True C')
    ax3.set_ylabel('Prediction')
    ax3.set_title(f"Eve's Performance (MSE: {mse_eve:.4f})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Eve's errors vs Blind
    ax4 = plt.subplot(3, 2, 4)
    eve_errors = np.abs(c_hat_np - C_np)
    blind_errors = np.abs(c_blind_np - C_np)
    ax4.hist(eve_errors, bins=30, alpha=0.5, label=f'Eve (mean: {eve_errors.mean():.4f})', edgecolor='black')
    ax4.hist(blind_errors, bins=30, alpha=0.5, label=f'Blind (mean: {blind_errors.mean():.4f})', edgecolor='black')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title("Eve vs Blind Baseline - Error Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Sample data table (first 10 samples)
    ax5 = plt.subplot(3, 2, 5)
    ax5.axis('tight')
    ax5.axis('off')
    
    num_samples = min(10, batch_size)
    table_data = []
    table_data.append(['#', 'A', 'B', 'C (true)', 'D (true)', 'D (Bob)', 'C (Eve)', 'C (Blind)'])
    
    for i in range(num_samples):
        row = [
            str(i+1),
            f"{x_np[i, 0]:.2f}",
            f"{x_np[i, 1]:.2f}",
            f"{C_np[i]:.2f}",
            f"{D_np[i]:.2f}",
            f"{d_hat_np[i]:.2f}",
            f"{c_hat_np[i]:.2f}",
            f"{c_blind_np[i]:.2f}",
        ]
        table_data.append(row)
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Color header row
    for i in range(8):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title("Sample Predictions (First 10)", pad=20)
    
    # 6. Summary metrics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    summary_text = f"""
    SUMMARY METRICS
    
    Bob (Predict D with key):
      MSE: {mse_bob:.6f}
      Status: {'✓ PASS' if bob_success else '✗ FAIL'}
    
    Eve (Predict C without key):
      MSE: {mse_eve:.6f}
      Blind MSE: {mse_blind:.6f}
      Difference: {eve_advantage:+.6f}
      Status: {'✓ PASS (worse than blind)' if eve_success else '✗ FAIL (better than blind)'}
    
    Overall:
      {'✓✓ SELECTIVE ENCRYPTION SUCCESS' if (bob_success and eve_success) else '✗ Needs improvement'}
    
    Explanation:
    • A, B, C, D are input values
    • Bob should predict D accurately (with key)
    • Eve should NOT predict C well (without key)
    • Selective encryption protects C while
      revealing D to authorized party (Bob)
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle("Selective Encryption Evaluation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()