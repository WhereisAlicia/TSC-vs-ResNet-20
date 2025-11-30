# TSC-vs-ResNet-20

# Image Classification via Flattened Time Series

This repository contains the official implementation code for the Master's Thesis: **"Image Classification via Flattened Time Series: A Comparative Study of TSC Models and CNN Baselines"**.

## 📖 Project Overview
This project investigates the feasibility of applying Time Series Classification (TSC) models to image classification tasks. specifically, it transforms 2D **Fashion-MNIST** images into 1D sequences using **Hilbert space-filling curves** and compares the performance of state-of-the-art TSC models against a standard 2D CNN baseline.

**Key Metrics:** Accuracy, F1-Score, Training Time, and Inference Time.

## 📂 Project Structure

The repository is organized as follows. Each script corresponds to a specific model implementation:

| Model | Script Name | Description |
| :--- | :--- | :--- |
| **Hydra** | `train_hydra_gpu.py` | Implementation of the Hydra model (GPU version). |
| **Quant** | `train_quant.py` | Implementation of the Quant model (CPU version). |
| **InceptionTime** | `inceptiontime_torch_final.py` | PyTorch implementation of the InceptionTime ensemble. |
| **ResNet-20** | `fashionmnist-final.py` | The 2D CNN Baseline (ResNet-20). |

> **Note regarding Hyperparameter Tuning:**
> In `train_hydra_gpu.py` and `train_quant.py`, the code sections related to grid search and hyperparameter tuning have been commented out to keep the main execution flow clean. You can uncomment these sections to reproduce the tuning process.

## 🚀 Getting Started

### Requirements
* Requirement-TSC for Hydra, Quant, and InceptionTime
* Requirement-CNN for ResNet-20

## 🔗 Acknowledgements & References

This project builds upon open-source implementations. We acknowledge the authors of the original repositories:

ResNet-20 Implementation:

Based on the implementation by https://github.com/zhunzhong07/Random-Erasing


Hydra & Quant Implementations:

Based on the official code by https://github.com/angus924/aaltd2024

Some modifications were made to adapt this study.
Please refer to their repository for deeper insights into the model structures.

📄 License
This project is part of a Master's Thesis at Tilburg University. The code is available for educational and research purposes.
