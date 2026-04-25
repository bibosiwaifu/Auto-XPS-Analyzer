# Auto-XPS-Analyzer: Physics-Informed 1D-CNN Analysis

**Auto-XPS-Analyzer** is an automated pipeline designed for materials science researchers to analyze XPS Survey Scans. By leveraging a **One-Dimensional Convolutional Neural Network (1D-CNN)**, this tool automatically identifies 13 common elements and predicts their Atomic Percentages ($At\%$) directly from raw experimental data.

---

## 🌟 Key Features

### 1. Physics-Informed Data Augmentation
To overcome small sample sizes, this project implements a custom augmentation algorithm that simulates:
* **Energy Shifts:** Randomly shifts binding energies to simulate charging effects.
* **Gaussian Noise:** Simulates different Signal-to-Noise Ratios (SNR).
* **Gain Fluctuation:** Simulates variations in detector sensitivity.
This process expanded the dataset to **1,200 spectra**, significantly enhancing model robustness.

### 2. 1D-CNN Deep Learning Architecture
Our **1D-CNN** uses convolutional kernels to scan the spectrum and recognize continuous **Peak Shapes**, ensuring higher accuracy in element identification compared to traditional ML models.

### 3. End-to-End Web UI
Integrated with **Gradio**, providing a user-friendly interface for researchers to upload `.txt` files and receive quantitative results in seconds.

---

## 📊 Model Performance

* **Target Elements:** 13 (Al, C, Ca, Cl, Fe, K, N, Na, O, P, S, Si, Zn)
* **Training Platform:** Google Colab (NVIDIA T4 GPU)
* **Accuracy:** Achieved a **Mean Absolute Error ($MAE$) of ~1.37 $At\%$**.

---

## 📂 Project Structure

```text
Auto-XPS-Analyzer/
├── data/                    # Raw experimental data & Label.csv
├── models/                  # Pre-trained 1D-CNN model (.h5)
├── src/                     # Source scripts
├── requirements.txt         # Dependency list
└── README.md                # Project documentation
