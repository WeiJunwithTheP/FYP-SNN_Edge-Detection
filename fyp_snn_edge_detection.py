import numpy as np
import cv2
from brian2 import *
import matplotlib.pyplot as plt
from itertools import product
from skimage.metrics import mean_squared_error
import time
import os

# =====================================================
# USER SETTINGS
# =====================================================
# Using a relative path so it works on any computer
IMAGE_PATH = os.path.join("test_picture", "t14.png")
RESIZE_DIM = (256, 256) # Reduced size for faster demo, change back to (1080, 1080) for final
SIM_TIME = 200 * ms

prefs.codegen.target = "numpy"

# =====================================================
# LIF PARAMETERS (FPGA-friendly)
# =====================================================
tau = 15 * ms
v_rest = 0.0
v_reset = 0.0
v_th = 0.125
input_scale = 0.05

# =====================================================
# LOAD IMAGE
# =====================================================
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}. Please ensure the 'test_picture' folder exists.")

img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(cv2.resize(img, RESIZE_DIM), cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32) / 255.0

H, W = gray.shape
N = H * W
print("Image size:", gray.shape)

# =====================================================
# DELTA MODULATION (EVENT-DRIVEN ENCODING)
# =====================================================
dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
delta = dx + dy
delta /= delta.max() if delta.max() > 0 else 1.0

rates = delta.flatten() * 300 * Hz
input_group = PoissonGroup(N, rates=rates)

# =====================================================
# SOBEL KERNELS (8 DIRECTIONS)
# =====================================================
sobel_kernels = np.array([
    [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]],
    [[ 0,  1,  2], [-1,  0,  1], [-2, -1,  0]],
    [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
    [[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]],
    [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]],
    [[ 0, -1, -2], [ 1,  0, -1], [ 2,  1,  0]],
    [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]],
    [[-2, -1,  0], [-1,  0,  1], [ 0,  1,  2]]
], dtype=np.float32)

NUM_DIR = sobel_kernels.shape[0]
K = 3
OFFSET = 1

# =====================================================
# CONVOLUTION LAYER (LIF NEURONS)
# =====================================================
eqs = '''
dv/dt = (v_rest - v) / tau : 1 (unless refractory)
'''

conv_layer = NeuronGroup(
    N * NUM_DIR,
    eqs,
    threshold='v > v_th',
    reset='v = v_reset',
    refractory=2 * ms,
    method='linear'
)

num_neurons = conv_layer.N
print("Number of neurons:", num_neurons)

# =====================================================
# FIXED CONVOLUTIONAL SYNAPSES
# =====================================================
syn = Synapses(
    input_group,
    conv_layer,
    model='w : 1',
    on_pre='v_post += w'
)

pre_idx, post_idx, weights = [], [], []

for k in range(NUM_DIR):
    kernel = sobel_kernels[k]
    for y in range(OFFSET, H - OFFSET):
        for x in range(OFFSET, W - OFFSET):
            post = k * N + (y * W + x)
            for ky, kx in product(range(K), repeat=2):
                w = kernel[ky, kx]
                if w != 0:
                    py = y + ky - OFFSET
                    px = x + kx - OFFSET
                    pre = py * W + px
                    pre_idx.append(pre)
                    post_idx.append(post)
                    weights.append(w * input_scale)

syn.connect(i=pre_idx, j=post_idx)
syn.w = weights

num_synapses = len(weights)
print("Number of synapses:", num_synapses)

# =====================================================
# MONITOR
# =====================================================
spike_mon = SpikeMonitor(conv_layer)

# =====================================================
# RUN SIMULATION (TIMED SNN ONLY)
# =====================================================
run(1 * ms)  # warm-up

t_snn_start = time.perf_counter()
run(SIM_TIME)
t_snn_end = time.perf_counter()

snn_time = t_snn_end - t_snn_start

print(f"SNN simulation time: {snn_time:.6f} s")
print(f"Speed factor: {(SIM_TIME/second)/snn_time:.3f} ×")

# =====================================================
# SPIKE → FEATURE MAPS
# =====================================================
feature_maps = np.zeros((NUM_DIR, H, W))

for k in range(NUM_DIR):
    for i in range(N):
        feature_maps[k].flat[i] = spike_mon.count[k * N + i]

edge_map = np.max(feature_maps, axis=0)
edge_map /= edge_map.max() if edge_map.max() > 0 else 1.0

# =====================================================
# HIGH-PASS FILTER
# =====================================================
hp_kernel = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
], dtype=np.float32)

edge_hp = cv2.filter2D(edge_map, -1, hp_kernel)
edge_hp = np.clip(edge_hp, 0, None)
edge_hp /= edge_hp.max() if edge_hp.max() > 0 else 1.0

# =====================================================
# OPENCV SOBEL (TIMED)
# =====================================================
t_cv_start = time.perf_counter()

sobel_ref = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
sobel_ref = np.abs(sobel_ref)
sobel_ref /= sobel_ref.max() if sobel_ref.max() > 0 else 1.0

t_cv_end = time.perf_counter()
cv_time = t_cv_end - t_cv_start

print(f"OpenCV Sobel time: {cv_time:.6f} s")

# =====================================================
# MSE METRICS
# =====================================================
mse_sobel = mean_squared_error(sobel_ref, edge_hp)
mse_original = mean_squared_error(gray, edge_hp)

print(f"MSE (SNN vs Sobel): {mse_sobel:.6f}")
print(f"MSE (SNN vs Original): {mse_original:.6f}")

# =====================================================
# SUMMARY
# =====================================================
print("\n========== PERFORMANCE SUMMARY ==========")
print(f"Image size            : {H} x {W}")
print(f"Neurons               : {num_neurons}")
print(f"Synapses              : {num_synapses}")
print(f"SNN time              : {snn_time:.6f} s")
print(f"OpenCV Sobel time     : {cv_time:.6f} s")
print(f"MSE (SNN vs Sobel)    : {mse_sobel:.6f}")
print(f"MSE (SNN vs Original) : {mse_original:.6f}")
print("========================================")

# =====================================================
# DISPLAY
# =====================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("SNN Edge Map")
plt.imshow(edge_hp, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("OpenCV Sobel")
plt.imshow(sobel_ref, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
