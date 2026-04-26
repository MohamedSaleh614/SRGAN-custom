# Custom SRGAN - Super Resolution GAN

A custom implementation of **SRGAN** (Super-Resolution Generative Adversarial Network) built from scratch in PyTorch for image super-resolution (4x upscale).

This project includes the full pipeline: Residual-in-Residual Dense Blocks (RRDB), Generator, Discriminator, perceptual loss (VGG), and adversarial training.

## 📁 Project Structure
SRGAN-custom/
├── nn.py          # Generator (with RRDB blocks) + Discriminator
├── dataset.py     # Custom Image Dataset loader
├── loss.py        # VGG Perceptual Loss
├── train.py       # Training script (Adversarial + Perceptual + Pixel loss)
└── README.md

## ✨ Features

- Custom **RRDB** (Residual-in-Residual Dense Block) architecture
- Generator with skip connection (residual learning)
- Patch-based Discriminator
- Combines **Pixel Loss (L1)** + **Perceptual Loss (VGG19)** + **Adversarial Loss**
- Simple and clean training loop
- Built from scratch (no official ESRGAN/SRGAN repo dependency)

## 🛠️ Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pillow tqdm
