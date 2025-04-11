# 🧠 CAE-IF: Credit Card Fraud Detection using Convolutional Autoencoder and Isolation Forest

This project implements an **unsupervised anomaly detection pipeline** using a Convolutional Autoencoder (CAE) followed by Isolation Forest (IF). It is structured as a modular Python package:

```
📁 cae_if_fraud_detection/
├── config.py         # Global settings and seed
├── data_loader.py    # Load and preprocess dataset
├── model.py          # CAE model architecture and functions
├── utils.py          # Metric functions
└── main.py           # Main script to train CAE and run IF
```

## 🚀 How to Run

1. Clone the repository and install requirements:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python main.py
```

The CAE is trained using 5-fold cross-validation. The best-performing encoder is then used to generate latent features for anomaly detection using Isolation Forest.

## ⚠️ Notice

This project is shared for **review purposes only**.  
Usage, reproduction, or distribution is strictly prohibited without the author's **written consent**.

© 2024 Zahra Salekshahrezaee. All rights reserved.  

