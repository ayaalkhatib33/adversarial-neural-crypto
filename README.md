# Adversarial Neural Cryptography – Symmetric Encryption with Neural Networks

This repository implements a neural-network-based symmetric encryption system inspired by Abadi and Andersen (2016).
Unlike traditional cryptography, where algorithms are fixed (such as AES or DES), adversarial neural cryptography uses neural networks that learn how to encrypt and decrypt messages directly from data.

Three neural networks are trained with competing objectives:

* **Alice** encrypts a plaintext message using a shared secret key.
* **Bob** decrypts the ciphertext using the same key.
* **Eve** attempts to reconstruct the plaintext without access to the key.

Alice and Bob collaborate to enable secure communication while Eve simultaneously attempts to break it.

---

## 1. Background

### Traditional Cryptography

Relies on fixed mathematical rules and well-understood security assumptions.

### Neural Cryptography

The encryption operation is not hand-designed.
Neural networks learn their own encoding and decoding transformations.

### Shared Key

* Alice and Bob receive the same random key.
* Eve does not receive the key.

### Adversarial Objectives

* Alice and Bob minimize Bob’s reconstruction error.
* Eve minimizes her own reconstruction error.
* Alice and Bob attempt to maximize Eve’s error.

### Emergent Encryption

The networks discover an encoding that appears random to Eve but remains decodable to Bob.

---

## 2. Project Structure

```
adversarial-neural-crypto/
│
├── results/
│   ├── symmetric/
│   ├── selective/      # additional experiment outputs
│   ├── text/           # text-based experiment outputs
│   └── image/          # image (MNIST) experiment outputs
│
├── src/
│   ├── data.py
│   ├── models.py
│   ├── utils.py
│   ├── train_symmetric.py
│   ├── train_selective.py
│   ├── train_text.py
│   ├── train_image.py
│   └── eval_eve.py
│
├── requirements.txt
└── README.md
```

---

## 3. Installation

### Clone the repository

```
git clone https://github.com/ayaalkhatib33/adversarial-neural-crypto.git
cd adversarial-neural-crypto
```

### Install dependencies

```
pip install -r requirements.txt
```

This installs PyTorch, NumPy, tqdm, Matplotlib, and torchvision (used for the MNIST image experiment).

---

## 4. Running the Project

The project uses a package-based structure, so training scripts should be run with:

```
python -m src.<script_name>
```

### Symmetric Encryption Training

Train Alice, Bob, and Eve together:

```
python -m src.train_symmetric
```

This script:

* generates random plaintexts and keys
* trains all three networks
* saves outputs in `results/symmetric/`

### Evaluating Eve Separately

Measure Eve’s ability to break a trained cipher:

```
python -m src.eval_eve
```

This loads the saved model in `results/symmetric/` and produces an evaluation file.

### Selective Encryption (optional)

Trains a variant where some message components are treated as more sensitive:

```
python -m src.train_selective
```

Outputs are written to `results/selective/`.

### Text-Based Encryption (optional)

A small toy experiment using synthetic text sequences:

```
python -m src.train_text
```

Results appear in `results/text/`.

### Image-Based Encryption (MNIST)

Applies adversarial encryption to image data (requires torchvision):

```
python -m src.train_image
```

Results are created in `results/image/`.

---

## 5. Training Output

After symmetric training, the following files appear in `results/symmetric/`:

| File              | Description                      |
| ----------------- | -------------------------------- |
| alice_bob.pt      | Trained Alice & Bob model        |
| alice_bob_eve.pt  | Full model including Eve         |
| training_log.json | Recorded metrics and loss values |

Example output:

```
Starting train_symmetric main...
Training symmetric: 100% |██████████| 5000/5000
Finished train_symmetric without error.
```

---

## 6. Technical Details

### Bob’s Loss

```
L_bob = || plaintext – bob_output ||²
```

### Eve’s Loss

```
L_eve = || plaintext – eve_output ||²
```

### Alice & Bob’s Objective

```
L_total = L_bob - L_eve
```

Alice and Bob minimize this objective;
Eve minimizes only `L_eve`.

---

## 7. Example Workflow

1. Install dependencies
2. Run `python -m src.train_symmetric`
3. Inspect `training_log.json`
4. Use `alice_bob.pt` for evaluation
5. Optionally run:

   * `python -m src.eval_eve`
   * `python -m src.train_selective`
   * `python -m src.train_text`
   * `python -m src.train_image`

---

## 8. Limitations

* Intended for research and educational purposes
* Not secure for real-world cryptography
* Provides no formal security guarantees

---

## 9. References

### Foundational Work

Abadi, M., & Andersen, D. G. (2016).
*Learning to Protect Communications with Adversarial Neural Cryptography.*

### Contemporary Work

Chen, Z., Yu, H., & Zhou, Z. (2023).
*Neural Cryptography in Deep Learning: Improved Adversarial Encryption Networks.*
