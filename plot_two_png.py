#!/usr/bin/env python3
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python plot_two_pngs_rgb.py <image1.png> <image2.png>")
    sys.exit(1)

img1_path = sys.argv[1]
img2_path = sys.argv[2]

# -----------------------------
# Load images as grayscale
# -----------------------------
img1 = np.array(Image.open(img1_path).convert("L"))
img2 = np.array(Image.open(img2_path).convert("L"))

# Ensure same shape
if img1.shape != img2.shape:
    print("Error: Images must have the same dimensions!")
    print(f"Image1: {img1.shape}, Image2: {img2.shape}")
    sys.exit(1)

# -----------------------------
# Build RGB composite (raw values)
# -----------------------------
R = img1.astype(np.uint8)
G = img2.astype(np.uint8)
B = np.zeros_like(R, dtype=np.uint8)

rgb = np.stack([R, G, B], axis=-1)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 10))
plt.imshow(rgb)
plt.title(f"Red={img1_path},  Green={img2_path}")
plt.axis('off')
plt.show()

