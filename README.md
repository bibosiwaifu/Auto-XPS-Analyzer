🚀 Auto-XPS-Analyzer: Physics-Informed 1D-CNN Analysis
Auto-XPS-Analyzer is an automated pipeline designed for materials science researchers to analyze XPS Survey Scans. By leveraging a One-Dimensional Convolutional Neural Network (1D-CNN), this tool automatically identifies 13 common elements and predicts their Atomic Percentages (At %) directly from raw experimental data.
🌟 Key Features
1. Physics-Informed Data Augmentation
In real-world research, high-quality labeled datasets are often small. To overcome the limitation of having only 12 original samples, this project implements a custom augmentation algorithm that simulates:
 Energy Shifts: Randomly shifts binding energies to simulate charging effects and instrument calibration variances.
 Gaussian Noise: Introduces random noise to simulate different Signal-to-Noise Ratios (SNR).
 Gain Fluctuation: Simulates variations in beam intensity and detector sensitivity.This process expanded the dataset to 1,200 spectra, significantly enhancing model robustness.
2. 1D-CNN Deep Learning Architecture
Unlike traditional machine learning models (e.g., Random Forests) that treat energy points as independent variables, our 1D-CNN acts like a human expert. It uses convolutional kernels to scan the spectrum and recognize continuous Peak Shapes, ensuring high accuracy in element identification.3. End-to-End Web UIIntegrated with Gradio, the project provides a user-friendly interface. Researchers can upload a raw .txt file and receive a quantitative elemental analysis in seconds without writing a single line of code.
📊 Model Performance
 Target Elements: 13 (Al, C, Ca, Cl, Fe, K, N, Na, O, P, S, Si, Zn)
 Training Platform: Google Colab (NVIDIA T4 GPU)
 Accuracy: Achieved a Mean Absolute Error (MAE) of ~1.37 At %. This accuracy level is sufficient for rapid screening and qualitative-to-quantitative analysis in routine laboratory work.
📂 Project Structure
PlaintextAuto-XPS-Analyzer/
├── data/                    # Raw experimental data & Label.csv
├── models/                  # Pre-trained 1D-CNN model (.h5)
├── src/                     
│   ├── preprocess.py        # Raw .txt parsing & normalization
│   ├── augment.py           # Physics-informed augmentation script
│   ├── train_cnn.py         # 1D-CNN training pipeline
│   └── app_colab.py         # Gradio Web UI script
├── requirements.txt         # Dependency list
└── README.md                # Project documentation
🛠️ Installation & Usage
1. Clone the RepositoryBashgit clone https://github.com/your-username/Auto-XPS-Analyzer.git
cd Auto-XPS-Analyzer
2. Install DependenciesBashpip install -r requirements.txt
3. Run the AnalyzerLaunch the Gradio interface:Bashpython src/app_colab.py
Note: For users on Apple Silicon Macs experiencing "mutex lock" errors, running the script in Google Colab is recommended.
🔬 Scientific Workflow
1. Data Ingestion: Automatically strips metadata and extracts Binding Energy vs. Intensity arrays.
2. Preprocessing: Normalizes intensities to a $[0, 1]$ range to eliminate discrepancies between different scan batches.
3. 1D-CNN Inference: Sliding kernels identify characteristic peaks across the 0-1200 eV range.Q
4. uantification: Predicts Atomic Percentages and visualizes results via an interactive bar chart.
👤 Author 
A Sina 
 Postgraduate Researcher in Materials Science & Membrane Technology.
 Interests: AI for Materials (MatSci AI), Battery Technology, and XPS Automation.