# Adversarial Neural Cryptography – Symmetric Encryption with Neural Networks

This repository implements a neural-network-based symmetric encryption scheme inspired by the work of Abadi and Andersen (2016).  
Instead of using traditional cryptographic algorithms, neural networks learn to encrypt and decrypt messages through adversarial training.

Three neural networks are trained with competing goals:

- **Alice**: Encrypts a plaintext using a shared key.
- **Bob**: Uses the same shared key to decrypt Alice’s ciphertext.
- **Eve**: Attempts to intercept the ciphertext and recover the original plaintext without the key.

Alice and Bob are optimized to communicate successfully, while Eve is simultaneously optimized to break the communication. This adversarial setup forces Alice and Bob to discover an encryption strategy that is difficult for Eve to decode.

---------------------------------------------------------------------

## 1. Background

Traditional encryption algorithms such as AES or DES use fixed mathematical operations.  
In contrast, adversarial neural cryptography attempts to learn encryption rules automatically.

### Core Concepts

1. **Shared Key**  
   Alice and Bob both receive the same key as input. Eve does not receive the key.

2. **Adversarial Training**  
   - Alice and Bob minimize Bob’s reconstruction error.
   - Eve minimizes her own reconstruction error.
   - Alice and Bob also maximize Eve’s error.

3. **Learned Encryption**  
   The model learns a transformation that:
   - looks random to Eve,
   - but is decodable by Bob using the shared key.

---------------------------------------------------------------------

## 2. Project Structure

