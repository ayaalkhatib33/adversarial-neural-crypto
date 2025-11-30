# Adversarial Neural Cryptography – Symmetric Encryption with Neural Networks

This repository implements a neural-network-based symmetric encryption system inspired by Abadi and Andersen (2016).  
Unlike traditional cryptography, where algorithms are fixed (such as AES or DES), adversarial neural cryptography uses neural networks that learn how to encrypt and decrypt messages entirely from data.

Three neural networks are trained with competing objectives:

- **Alice**: Encrypts a plaintext message using a shared secret key.
- **Bob**: Decrypts the ciphertext using the same key.
- **Eve**: Attempts to intercept the ciphertext and reconstruct the plaintext without the key.

Alice and Bob collaborate to enable secure communication while Eve simultaneously attempts to break it. This adversarial setup forces Alice and Bob to discover an encryption strategy that is hard for Eve to decode.

---

## 1. Background

### Traditional Cryptography  
Relies on fixed mathematical rules and algorithms. Security is based on computational hardness assumptions.

### Neural Cryptography  
Instead of fixed rules, neural networks learn how to protect communication through training. Encryption is **learned**, not programmed.

### Core Concepts

#### Shared Key  
Alice and Bob both receive the same random key.  
Eve does not receive the key and must infer the plaintext from the ciphertext alone.

#### Adversarial Training Dynamics  
- Alice and Bob minimize Bob’s reconstruction error.  
- Eve minimizes her reconstruction error.  
- Alice and Bob also maximize Eve’s error through adversarial gradients.

#### Emergent Learned Encryption  
The system learns a transformation that:
- appears random to Eve,
- but is decodable by Bob with the correct key.

---

## 2. Project Structure

adversarial-neural-crypto/
│
├── results/
│ └── symmetric/
│ ├── alice_bob.pt # Trained Alice & Bob model
│ ├── alice_bob_eve.pt # Trained Alice, Bob, and Eve
│ └── training_log.json # Metrics recorded during training
│
├── src/
│ ├── data.py # Random data/key generators
│ ├── models.py # Definitions of Alice, Bob, Eve
│ ├── train_symmetric.py # Main adversarial training script
│ ├── train_selective.py # Optional alternative training script
│ ├── eval_eve.py # Optional evaluation/attack script
│ └── utils.py # Helper functions
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## 3. Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/ayaalkhatib33/adversarial-neural-crypto.git
cd adversarial-neural-crypto
Step 2 — Install dependencies
bash
Copy code
pip install -r requirements.txt
This installs:

PyTorch

NumPy

tqdm

Matplotlib (optional)

4. How to Run the Project
Train Alice, Bob, and Eve together
This trains the full adversarial system:

bash
Copy code
python src/train_symmetric.py
What this script does:

creates random plaintexts and keys

trains all three neural networks

logs training loss

saves model weights into results/symmetric/

Evaluate Eve separately (optional)
bash
Copy code
python src/eval_eve.py
This assesses how well Eve can break the learned encryption using saved models.

5. Training Output
After training completes, the following files are generated:

File	Description
alice_bob.pt	Alice & Bob model trained to communicate
alice_bob_eve.pt	Full model with Eve included
training_log.json	Loss curves and metrics over training steps

Example console output:

sql
Copy code
Starting train_symmetric main...
Training symmetric: 100% |██████████| 5000/5000 
Finished train_symmetric without error.
6. Technical Details
Loss Functions
Bob’s Loss (Reconstruction Error)
Bob tries to match the original plaintext:

ini
Copy code
L_bob = || plaintext – bob_output ||²
Eve’s Loss (Attack Error)
Eve attempts to decode the plaintext:

ini
Copy code
L_eve = || plaintext – eve_output ||²
Alice and Bob’s Total Objective

ini
Copy code
L_total = L_bob - L_eve
This encourages Alice and Bob to:

communicate well with each other

make Eve's predictions worse

Eve optimizes only L_eve.

7. Example Workflow
Install dependencies

Run train_symmetric.py

Inspect training_log.json

Load alice_bob.pt to test encryption/decryption

Optionally run Eve separately to assess model security

8. Limitations
This system is for research and education only.

The learned encryption is not secure for real-world cryptography.

This implementation does not guarantee resistance to modern cryptographic attacks.

9. References (Required for the Assignment)
Foundational Work
Abadi, M., & Andersen, D. G. (2016).
Learning to Protect Communications with Adversarial Neural Cryptography.
This is the core paper that introduced neural adversarial cryptography.

Contemporary Work Building on the Original Paper
Chen, Z., Yu, H., & Zhou, Z. (2023).
Neural Cryptography in Deep Learning: Improved Adversarial Encryption Networks.
This work expands and improves neural cryptographic architectures and adversarial training stability.
