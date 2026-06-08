import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from skimage.metrics import (
    mean_squared_error,
    structural_similarity
)

# =====================================================
# 1. SETTINGS
# =====================================================
IMAGE_PATH = "IMG PATH"
RESIZE_DIM = (256, 256)

# Trained Kernels
kernel_h = np.array([
    [-0.02879315,  0.17715165, -0.10721943],
    [ 0.06097224, -0.14783184, -0.04105892],
    [ 0.14806044, -0.02384954,  0.04330644]
])

kernel_v = np.array([
    [ 0.00908672, -0.14445831,  0.14225422],
    [-0.2636892,  -0.18888885,  0.07509353],
    [-0.04386042, -0.05765314,  0.22489855]
])

# SNN Parameters
CONTRAST_THRESH = 35
V_TH = 0.1

# =====================================================
# 2. LOAD IMAGE
# =====================================================
img_raw = cv2.imread(IMAGE_PATH, 0)

if img_raw is None:
    raise FileNotFoundError(f"Image not found:\n{IMAGE_PATH}")

img_resized = cv2.resize(img_raw, RESIZE_DIM)

W, H_img = RESIZE_DIM
KH = kernel_h
KV = kernel_v

# =====================================================
# 3. SOFTWARE SNN
# =====================================================
soft_edge = np.zeros((H_img, W), dtype=np.float32)

start_snn = time.perf_counter()

for y in range(1, H_img - 1):
    for x in range(1, W - 1):

        center = int(img_resized[y, x])

        sum_h = 0.0
        sum_v = 0.0

        for ky in range(3):
            for kx in range(3):

                raw_pixel = int(
                    img_resized[y + ky - 1, x + kx - 1]
                )

                # Unsigned 8-bit wrap
                thresh_hi = (center + CONTRAST_THRESH) & 0xFF
                thresh_lo = (center - CONTRAST_THRESH) & 0xFF

                spike_pos = raw_pixel > thresh_hi
                spike_neg = raw_pixel < thresh_lo

                if spike_pos:
                    sum_h += KH[ky, kx]
                    sum_v += KV[ky, kx]

                elif spike_neg:
                    sum_h -= KH[ky, kx]
                    sum_v -= KV[ky, kx]

        abs_h = abs(sum_h)
        abs_v = abs(sum_v)

        winner_potential = (
            abs_h if abs_h > abs_v else abs_v
        )

        if winner_potential > V_TH:
            soft_edge[y, x] = winner_potential

snn_duration = time.perf_counter() - start_snn

# Normalize SNN Output
soft_edge_norm = cv2.normalize(
    soft_edge,
    None,
    0,
    1,
    cv2.NORM_MINMAX
)

# =====================================================
# 4. OpenCV SOBEL
# =====================================================
start_sobel = time.perf_counter()

img_float = img_resized.astype(np.float64) / 255.0

sobel_x = cv2.Sobel(
    img_float,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_y = cv2.Sobel(
    img_float,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

sobel_mag = cv2.normalize(
    sobel_mag,
    None,
    0,
    1,
    cv2.NORM_MINMAX
)

sobel_duration = time.perf_counter() - start_sobel

# =====================================================
# 5. METRICS
# =====================================================

# Normalize original image
original_norm = img_resized.astype(np.float32) / 255.0

# -------------------------
# MSE
# -------------------------
mse_soft_vs_sobel = mean_squared_error(
    soft_edge_norm,
    sobel_mag
)

mse_soft_vs_original = mean_squared_error(
    soft_edge_norm,
    original_norm
)

mse_sobel_vs_original = mean_squared_error(
    sobel_mag,
    original_norm
)

# -------------------------
# SSIM
# -------------------------
ssim_soft_vs_sobel = structural_similarity(
    soft_edge_norm,
    sobel_mag,
    data_range=1.0
)

ssim_soft_vs_original = structural_similarity(
    soft_edge_norm,
    original_norm,
    data_range=1.0
)

ssim_sobel_vs_original = structural_similarity(
    sobel_mag,
    original_norm,
    data_range=1.0
)

# =====================================================
# 6. PRINT RESULTS
# =====================================================
print("\n========== BENCHMARK REPORT ==========")

print(f"\nSW SNN Time:         {snn_duration * 1000:.2f} ms")
print(f"OpenCV Sobel Time:   {sobel_duration * 1000:.2f} ms")

print("\n------------- MSE -------------")
print(f"SNN vs Sobel:        {mse_soft_vs_sobel:.6f}")
print(f"SNN vs Original:     {mse_soft_vs_original:.6f}")
print(f"Sobel vs Original:   {mse_sobel_vs_original:.6f}")

print("\n------------ SSIM ------------")
print(f"SNN vs Sobel:        {ssim_soft_vs_sobel:.6f}")
print(f"SNN vs Original:     {ssim_soft_vs_original:.6f}")
print(f"Sobel vs Original:   {ssim_sobel_vs_original:.6f}")

print("\n----------- SPARSITY ----------")
print(
    f"SNN Sparsity: "
    f"{(1 - np.count_nonzero(soft_edge) / soft_edge.size) * 100:.2f}%"
)

print("\n======================================\n")

# =====================================================
# 7. VISUALIZATION
# =====================================================
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title("Original Input")
plt.imshow(img_resized, cmap='gray')
plt.axis('off')

# SNN Output
plt.subplot(1, 3, 2)
plt.title("SNN Output")
plt.imshow(soft_edge_norm, cmap='gray')
plt.axis('off')

# OpenCV Sobel
plt.subplot(1, 3, 3)
plt.title("OpenCV Sobel")
plt.imshow(sobel_mag, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
