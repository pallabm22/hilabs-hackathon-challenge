# hilabs-hackathon-challenge

# 🩺 Healthcare Risk Prediction – Feature Analysis & Modeling

This repository focuses on **healthcare data analysis** and **risk score prediction** using correlation analysis, feature engineering, dimensionality reduction (PCA), and machine learning models.  

The project integrates multiple healthcare data sources — care, diagnosis, visits, and patient information — to identify the most significant predictors of patient risk.

---

## 📂 Project Structure

├── Care_Analysis/
│ ├── care.ipynb # Care data analysis
│ ├── risk_care.ipynb # Merging care data with risk table and correlation analysis
│
├── Data_preprocessing/
│ ├── Correlation_matrix_analysis.ipynb # Feature correlation computation with risk_score
│ ├── Dataset/ # Contains base datasets used for merging
│ ├── merging_training_dataset.ipynb # Final merged dataset creation for training
│
├── Diagnosis_Analysis/
│ ├── diagnosis.ipynb # Diagnosis-specific data exploration
│ ├── risk_diagnosis.ipynb # Diagnosis-risk dataset merging and analysis
│
├── Experiments_models/
│ ├── ML_models.ipynb # ML models without PCA
│ ├── NN_model.ipynb # Neural network baseline
│ ├── ML_models_with_hyperparameter_tuning.ipynb # Grid/Random Search optimization
│
├── Model/
│ ├── main_training_model_pca.ipynb # Main PCA-based model training
│ ├── training_testing_model_pca_pipeline.ipynb # PCA pipeline for training/testing
│
├── Patient_Analysis/
│ ├── patient.ipynb # Patient-level data insights
│ ├── patient_risk.ipynb # Patient data merged with risk for correlation
│
├── Pcms_hackathon_data/
│ ├── training_data_raw.csv # Raw training dataset
│ ├── testing_data_raw.csv # Raw testing dataset
│
├── Risk_Analysis/
│ ├── risk_analysis.ipynb # Risk table understanding and feature alignment
│
├── Test/
│ ├── processing_test_dataset.ipynb # Test data preprocessing
│ ├── processed_test_dataset.csv # Cleaned test dataset
│ ├── merge_test_dataset.ipynb # Merging test dataset with risk data
│
├── Visits_Analysis/
│ ├── visit.ipynb # Visit data analysis
│ ├── risk_visit.ipynb # Visit-risk correlation study
│
└── README.md # Project documentation



---

## ⚙️ Environment Setup

You can replicate the working environment by creating a **Conda environment** and installing the required packages:

```bash
conda create --name Hilabs python=3.10
conda activate Hilabs

pip install ipykernel
python -m ipykernel install --user --name=Hilabs --display-name "Hilabs"

pip install pandas
pip install numpy
pip install scikit-learn
pip install tensorflow

---


## Methodology Overview
1. Data Acquisition & Preprocessing

Multiple datasets (care, diagnosis, visits, patient) were processed and merged with the risk table.

Missing values were handled and categorical columns encoded as required.

Each merged dataset was analyzed independently to identify significant predictors of risk_score.

2. Correlation Matrix Analysis

Computed feature-to-risk_score correlations across all merged datasets.

Identified and selected top positively correlated features as key indicators of patient risk.

3. Feature Selection & PCA

Selected top-performing features based on correlation ranking.

Applied Principal Component Analysis (PCA) to reduce feature dimensionality.

Extracted 21 principal components out of 42 features, preserving 95% cumulative variance.

4. Model Development

Baseline models: Linear Regression, Random Forest, XGBoost, and Neural Networks.

With PCA: Trained models using reduced PCA feature set for improved efficiency.

Without PCA: Compared performance to measure variance preservation effectiveness.

Hyperparameter Tuning: Used GridSearchCV and RandomizedSearchCV for optimization.

5. Evaluation Metrics

Models evaluated using:

R² Score

RMSE (Root Mean Square Error)

MAE (Mean Absolute Error)

📊 Key Insights

High correlation features (e.g., type_SCREENING, num_care_events, num_chronic_conditions) were strong predictors of risk.

PCA effectively reduced redundancy while retaining predictive strength.

Combined risk analysis across domains improved the model’s ability to generalize to unseen patient profiles.
