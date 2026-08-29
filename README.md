# ChemTrans-CVAE: Chemical Transformer Conditional Variational Autoencoder for Targeted Molecular Generation and VEGFR-2 Inhibitor Design.
# Overview
ChemTrans-CVAE, a hybrid framework that integrates a pre-trained Transformer architecture with a Conditional Variational Autoencoder to enable high-fidelity, property-constrained molecular optimization. By leveraging the Self-referencing Embedded Strings(SELFIES) representation to enforce chemical validity, the model is pre-trained on the generic chemical space of the Molecular Sets (MOSES) dataset and subsequently fine-tuned to target Vascular Endothelial Growth Factor Receptor-2 (VEGFR-2). We demonstrate that ChemTrans-CVAE achieves 100% structural validity and outperforms baseline models in novelty and scaffold exploration. To ensure therapeutic relevance, we implement a comprehensive downstream validation pipeline: generated entities are filtered through a robust QSAR model and subjected to structure-based docking using GNINA. This workflow successfully identifies a diverse cluster of high-potency candidates with superior predicted binding affinities. Our results establish ChemTrans-CVAE as an effective end-to-end strategy that unifies deep generative learning with structure-based validation to accelerate the discovery of novel kinase inhibitors.
<img width="949" height="425" alt="image" src="https://github.com/user-attachments/assets/94ba41ab-57eb-48fd-abf4-4670af84a3b8" />

This repository contains the complete codebase for:

* Training the ChemTrans-CVAE generative model
* Generating novel molecules with controlled properties
* QSAR model training for activity prediction
* Molecular descriptor calculation and filtering

# Repository Structure 
'
├── VEGFR2_autoregressive-scaled-conditional-tra...ipynb
│   └── Main model building and training notebook
│
├── ChemTrans_CVAE_Model_ipynb
│   └── Load trained model and generate new molecules
│
├── QSAR-vegfr-2.ipynb
│   └── QSAR model training for VEGFR-2 activity prediction
│
├── descriptors_for_MOSES_Test.ipynb
│   └── Calculating molecular descriptors using RDKit and Mordred
│
├── filter_descriptors.ipynb
│   └── Descriptor filtering pipeline (correlation, variance, normalization)
│
└── README.md
    └── This file
'
# Data Preparation
## 1. Download MOSES Dataset
```
wget https://github.com/molecularsets/moses/raw/master/data/train.csv
wget https://github.com/molecularsets/moses/raw/master/data/test.csv
```

## 2. Calculate Molecular Descriptors
Run 'descriptors_for_MOSES_Test.ipynb' to:
* Parse SELFIES/SMILES strings
* Calculate descriptors using RDKit and Mordred
* Output descriptor matrices for downstream use

## 3. Filter Descriptors
Run 'filter_descriptors.ipynb' to:
* Remove non-numeric and duplicate features
* Eliminate highly correlated descriptors (>0.95)
* Normalize using Min-Max scaling
* Apply variance filtering (≤0.015 for generation, ≤0.01 for QSAR)

# Training the Model
## 1. Training ChemTrans-CVAE
Run 'VEGFR2_autoregressive-scaled-conditional-transformers-vae.py' to:
* Build the encoder (pre-trained Transformer + descriptor conditioning)
* Build the decoder (autoregressive Transformer with memory conditioning)
* Pre-train on MOSES dataset
* Fine-tune on target-specific dataset (e.g., VEGFR-2 inhibitors)
* Loss function: L_total = L_recon + β·L_KL

## 2. QSAR Model Training
Run 'QSAR-vegfr-2.ipynb' to:
* Train on 13,406 compounds (12,400 train / 1,006 test)
* Use 234 curated descriptors (variance > 0.01)
* Multi-feature fusion: Transformer embeddings + Morgan fingerprints + MACCS keys + descriptors
* Training objective: MSE loss for pIC50 prediction

# Generating Molecules
After training, use 'ChemTrans_CVAE_Model_.ipynb' to:
1. Load Model and Checkpoint
2. Generate New Molecules
  - Configure sampling parameters:
    - Focused: temperature=1.0, top_k=100, top_p=1.0
    - Diverse: temperature=1.2, top_k=100, top_p=0.95
  
