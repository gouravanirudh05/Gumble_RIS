# CNN-Based Visual Recognition for RIS Beamforming Optimization using Gumbel-Softmax

## Overview

This project implements a **Convolutional Neural Network (CNN) based visual recognition framework** for optimizing **Reconfigurable Intelligent Surface (RIS)** configurations in wireless communication systems. The approach interprets wireless channel matrices as **2D spatial representations**, enabling CNNs to learn propagation patterns similarly to visual recognition tasks.

Using these learned spatial features, the system predicts optimal **discrete RIS phase configurations** through a **Gumbel-Softmax relaxation**, allowing end-to-end differentiable optimization.

The primary objective is to **maximize capacity in RIS-assisted wireless communication systems**.

---

## Motivation

Reconfigurable Intelligent Surfaces are a key technology for **next-generation wireless networks (6G)**. An RIS consists of many passive elements capable of applying programmable phase shifts to reflected electromagnetic waves.

Optimizing RIS phase configurations is challenging because:

* Phase shifts are **discrete values**
* The optimization problem is **non-convex and combinatorial**
* Traditional optimization techniques scale poorly with large RIS sizes

This project addresses these challenges by combining:

* **CNN-based spatial feature extraction** from channel matrices
* **Gumbel-Softmax relaxation** for differentiable discrete optimization

---

## Key Idea

Wireless channel matrices exhibit **structured spatial patterns**, which can be treated similarly to images. A CNN can therefore learn meaningful spatial features from these matrices.

The system pipeline:

1. Represent channel matrices as spatial tensors
2. Use a CNN to extract propagation features
3. Predict phase selection probabilities
4. Use **Gumbel-Softmax** to approximate discrete phase selection
5. Compute achievable communication rate

The model is trained to **maximize spectral efficiency**.

---

## System Architecture

```
Channel Matrices
      ↓
CNN Feature Extractor
      ↓
Fully Connected Prediction Network
      ↓
Gumbel-Softmax Phase Selection
      ↓
RIS Configuration
      ↓
Achievable Rate Computation
```

The CNN learns to recognize spatial propagation structures that determine optimal RIS phase shifts.

---

## Channel Representation

Each training sample consists of channel matrices describing signal propagation through the environment:

* Base Station → RIS channel
* RIS → User channel
* Direct Base Station → User channel

These matrices are converted into **multi-channel 2D tensors**, enabling CNN processing analogous to image inputs.

---

## CNN Model

The CNN acts as a spatial feature extractor that identifies patterns in wireless channel responses.

The network outputs **phase selection logits** for each RIS element.

---

## Gumbel-Softmax Relaxation

RIS elements typically support a limited set of discrete phase shifts.

To enable gradient-based optimization, we use the **Gumbel-Softmax trick** which approximates categorical sampling with a differentiable formulation:

[
y_i = \frac{\exp((\log(\pi_i) + g_i)/\tau)}{\sum_j \exp((\log(\pi_j) + g_j)/\tau)}
]

Where:

* (g_i) is sampled Gumbel noise
* (\tau) is a temperature parameter controlling discreteness

During inference, the **argmax phase value** is selected for each RIS element.

---

## Objective Function

The system optimizes **spectral efficiency**:

[
R = \log_2(1 + \text{SNR})
]

Where the Signal-to-Noise Ratio depends on the effective channel created by the RIS configuration.

Training objective:

```
Loss = -Mean Achievable Rate
```

Maximizing achievable rate encourages the model to learn optimal phase configurations.

---

## Experiments

Experiments are conducted using channel datasets for different RIS sizes.

| RIS Elements | Dataset             |
| ------------ | ------------------- |
| 4            | RIS_Channels_4.mat  |
| 16           | RIS_Channels_16.mat |
| 64           | RIS_Channels_64.mat |

Performance improves as the number of RIS elements increases due to higher beamforming gain.

---


## Installation

Clone the repository:

```bash
git clone https://github.com/gouravanirudh05/Gumble_RIS
cd Gumble_RIS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Technologies Used

* Python
* PyTorch
* Convolutional Neural Networks (CNN)
* Gumbel-Softmax relaxation
* NumPy
* SciPy
* MATLAB dataset processing

---

## Research Significance

This project demonstrates how **CNN-based visual recognition techniques can be applied to wireless communication systems** by treating channel matrices as spatial data.

The approach bridges **computer vision methodologies and wireless system optimization**, contributing to research directions in **AI-native wireless networks and intelligent radio environments for 6G**.

---

## Implemented by

**Gourav Anirudh**
**Arnav Oruganty**
