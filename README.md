
# Spectra Prediction for Polymethine-Based Fluorophores
A machine learning-based toolkit for predicting spectral properties for polymethine-based fluorophores, including absorption wavelength, emission wavelength, and molar absorptivity.

## Overview
This project employs molecular fingerprints and machine learning algorithms to predict molecular spectral properties:

- **Absorption Wavelength
- **Emission Wavelength
- **Molar Absorptivity

## Basic Usage
1. Generate Molecular Features

from src.featurization import MolecularFeaturizer

# Generate molecular fingerprints
featurizer = MolecularFeaturizer()
feature_df = featurizer.create_feature_dataframe(
    smiles_list=df['SMILES'].tolist(), 
    target_data=df, 
    target_type='ABS'
)

2. Feature Selection

from src.feature_selection import run_feature_selection_pipeline

# Run complete feature selection pipeline
run_feature_selection_pipeline('data/processed/absorption_features.xlsx', 'ABS')

3. Model Training

from src.model_training import ModelTrainer

# Train and evaluate all models
trainer = ModelTrainer()
results = trainer.train_and_evaluate_all_models(
    'data/processed/absorption_features_rfecv.xlsx', 
    'ABS', 
    n_splits=10
)

4. Make Predictions

from src.predict import predict_absorption, save_predictions

# Simple prediction
predictions = predict_absorption('dataset/absorption_fps.xlsx')
save_predictions(predictions, 'dataset/absorption_fps.xlsx', 'results/predictions.xlsx')


## Methodology
1. Molecular Representation
MACCS Keys: 166-bit structural keys

Morgan Fingerprints: 1024-bit circular fingerprints (radius=2)

RDKit Fingerprints: 1024-bit topological fingerprints

Solvent Descriptors: Et(30), SP, SdP, SA, SB parameters

2. Feature Selection Pipeline
Variance Threshold: Remove low-variance features (threshold=0.01)

Pearson Correlation: Filter by correlation with target (threshold=0.15)

Mutual Information: Select informative features (threshold=0.01)

Recursive Feature Elimination: Optimal feature subset selection

3. Machine Learning Models
K-Nearest Neighbors (KNN)

Kernel Ridge Regression (KRR)

Support Vector Regression (SVR)

Random Forest (RF)

XGBoost (XGB)

LightGBM (LGB)

4. Model Evaluation
10-fold Stratified Cross Validation

Metrics: R², RMSE, MAE

Automatic Hyperparameter Tuning


## Data Format
Input Data Format
ID,SMILES,SOLVENT,Et(30),SP,SdP,SA,SB,ABS
1,CCCC[N+]1=...,MeCN,45.6,0.645,0.974,0.044,0.286,758
2,CCCC[N+]1=...,MeCN,45.6,0.645,0.974,0.044,0.286,817

Output Prediction Format
ID,SMILES,SOLVENT,Predicted_ABS
1,CCCC[N+]1=...,MeCN,745.2
2,CCCC[N+]1=...,MeCN,812.8

































