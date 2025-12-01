import torch
from src.models import AliceSymmetric, BobSymmetric, EveSymmetric
from src.data import generate_symmetric_batch
from src.utils import get_device


def main():
    device = get_device()

    # Load the checkpoint saved by train_symmetric
    ckpt_path = "results/symmetric/alice_bob.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract architecture parameters from the checkpoint
    message_len = checkpoint["message_len"]
    key_len = checkpoint["key_len"]

    # Recreate the models with the same sizes used during training
    alice = AliceSymmetric(message_len=message_len, key_len=key_len).to(device)
    bob = BobSymmetric(message_len=message_len, key_len=key_len).to(device)
    eve = EveSymmetric(message_len=message_len).to(device)

    # Load trained weights
    alice.load_state_dict(checkpoint["alice_state_dict"])
    bob.load_state_dict(checkpoint["bob_state_dict"])
    # Eve is untrained here on purpose: she represents an attacker starting fresh

    # Generate a single random message/key pair
    plaintext, key = generate_symmetric_batch(
        batch_size=1,
        message_len=message_len,
        key_len=key_len,
        device=device,
    )

    plaintext = plaintext.to(device)
    key = key.to(device)

    # Run through the system
    ciphertext = alice(plaintext, key)
    bob_out = bob(ciphertext, key)
    eve_out = eve(ciphertext)

    # Move to CPU and format for printing
    pt = plaintext.cpu().numpy().round(3)
    ct = ciphertext.cpu().detach().numpy().round(3)
    bob_rec = bob_out.cpu().detach().numpy().round(3)
    eve_rec = eve_out.cpu().detach().numpy().round(3)

    print("\n=== Single Example Demonstration ===\n")
    print("Message length:", message_len, " Key length:", key_len, "\n")
    print("Plaintext:    ", pt)
    print("Ciphertext:   ", ct)
    print("Bob Recovers: ", bob_rec)
    print("Eve Recovers: ", eve_rec)
    print("\n(Goal: Bob ≈ Plaintext, Eve ≠ Plaintext)\n")
    print("====================================\n")


if __name__ == "__main__":
    main()
