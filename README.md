# FYP-SNN_Edge-Detection

# Bio-Inspired Edge Detection using Spiking Neural Networks (SNN)

This repository contains the implementation of an edge detection system using Leaky Integrate-and-Fire (LIF) neurons. The project explores neuromorphic computing techniques by utilizing the **Brian2** spiking neural network simulator to replicate the functionality of traditional Sobel filters.

## 🚀 Project Overview
Traditional computer vision algorithms are often computationally expensive for edge-computing devices. This project explores a **biomimetic approach**, using event-driven neural spikes to identify edges in high-resolution images.



### Key Features:
* **Delta Modulation:** Encodes image gradients into spiking events.
* **8-Directional Processing:** Uses 8 specialized Sobel kernels ($0^{\circ}$ to $315^{\circ}$) mapped to synaptic weights.
* **LIF Neuron Model:** Simulates biological membrane potential ($v$) to perform feature extraction.
* **Quantitative Validation:** Uses Mean Squared Error (MSE) to compare SNN performance against industry-standard OpenCV results.

---

## 🔬 Methodology & Tuning

### 1. Parameter Optimization
To achieve high-quality edge maps, the following parameters were tuned based on experimental performance:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **$V_{th}$** | 0.125 | Membrane threshold for spike generation. |
| **$\tau$** | 15 ms | Time constant for potential decay. |
| **Input Scale** | 0.05 | Scaling factor for synaptic weights. |
| **Simulation Time** | 200 ms | Biological time simulated per frame. |

### 2. Quantitative Evaluation
The system measures the **Mean Squared Error (MSE)** to validate accuracy. A lower MSE relative to the Sobel benchmark indicates successful neural reconstruction.

$$MSE = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i,j) - K(i,j)]^2$$

---

# 🛠️ Installation & Usage

### 1. Requirements
Ensure you have Python installed, then install the necessary dependencies:
```bash
pip install numpy opencv-python brian2 matplotlib scikit-image
