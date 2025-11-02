# 🩺 Healthcare Risk Prediction
## _Proactive Patient Risk Scoring with ML & PCA_

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-green.svg)]()
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A data-driven system for predicting **patient risk scores** by integrating multi-domain healthcare data, performing correlation-based feature engineering, applying **PCA**, and training machine learning & deep learning models.

- Clean & merge healthcare datasets  
- Engineer care + diagnosis + patient + visit features  
- Perform correlation analysis & PCA  
- Train ML & NN models for risk score prediction  

---

## Features

-  Multi-source healthcare dataset integration  
-  Advanced preprocessing & feature engineering  
-  Correlation-based feature selection  
-  ML + Neural Network training pipelines  
-  PCA for 95% variance retention  
-  Model evaluation using R², MAE, RMSE  
-  Pipeline for both training & inference  


---

## Why This Matters

Healthcare systems often react **after** patients deteriorate.  
This project enables **proactive intervention** by predicting risk early — using structured medical data + machine learning.



### 📂 Project File Structure

```text
├── Care_Analysis/
│   ├── care.ipynb                         # Care data analysis
│   └── risk_care.ipynb                    # Care data merged with risk table
├── training_data_preprocessing/
│   ├── Correlation_matrix_analysis        # Correlation computation with risk_score
│   ├── Dataset/                           # Contains base datasets for merging
│   └── merging_training_dataset.ipynb     # Final merged dataset for model training
│
├── Diagnosis_Analysis/
│   ├── diagnosis.ipynb                    # Diagnosis-specific data exploration
│   └── risk_diagnosis.ipynb               # Diagnosis-risk dataset merging and analysis
│
├── Experiments_models/
│   ├── NN_Pred_without_PCA                                  # ML models without PCA 
│   ├── NN_model.ipynb                                       # Neural Network baseline without PCA 
│   └── ML_Models_Pred_params_trainging.ipynb                # Grid/Random Search optimization without PCA 
│
├── Model/
|   ├── Model_Training_with_PCA.ipynb      # PCA-based model training
│   └── Model_Training_Testing_PCA.ipynb   # Main PCA pipeline for training-testing pipeline
│
├── Patient_Analysis/
│   ├── patient.ipynb                      # Patient-level data insights
│   └── patient_risk.ipynb                 # Patient data merged with risk for correlation analysis
│
├── Pcms_hackathon_data/
│   ├── Train                              # Raw training dataset
│   └── Test                               # Raw testing dataset
│
├── Risk_Analysis/
│   └── risk_analysis.ipynb                # Risk table structure and analysis
│
├── test_data_processing/
│   ├── processing_test_dataset       # Test data preprocessing
│   ├── processed_test_dataset        # Cleaned test dataset
│   └── merge_test_dataset.ipynb           # Merging test dataset with risk table
│
├── Visits_Analysis/
│   ├── visit.ipynb                        # Visit data exploration
│   └── risk_visit.ipynb                   # Visit-risk relationship analysis
│
|── requirement.txt                         
└── README.md                              # Project documentation


```
### Tech

This project leverages modern ML & data tools:

-  Python 3.10  
-  Pandas & NumPy — data processing  
-  scikit-learn — ML + PCA + pipelines  
-  TensorFlow — neural network baseline  
-  Matplotlib & Seaborn — visualizations   

---


### 🔄 Workflow & Methodology

#### **1. Data Preprocessing & Integration**
- Extracted multiple features for each dataset
- Merged multiple datasets with a central `risk_score` table  
- Cleaned missing values and encoded categorical features  
- Standardized fields across healthcare domains  

#### **2. Correlation Analysis & Feature Selection**
- Computed correlations with `risk_score`
- Identified top predictive features  
- Selected strongest positively correlated variables  

#### **3. Dimensionality Reduction with PCA**
- Applied Principal Component Analysis  
- Reduced 42 high-correlation features to **21 principal components**
- Preserved **95% of variance**  

#### **4. Model Development & Evaluation**
- Baseline models: *Linear Regression, Random Forest, XGBoost, Neural Networks*  
- PCA-enhanced ML pipelines for efficiency & interpretability  
- Hyperparameter tuning: *GridSearchCV / RandomizedSearchCV*  
- Metrics used: **R²**, **RMSE**, **MAE**  

---

### Key Insights

- PCA successfully removed redundancy while retaining predictive performance  
- Cross-domain data fusion improved model generalization  

---



### 📎 Reference Visuals

## **Correlation Matrix After & Before Irrelavant Feature Removal**
![Before and After Correlation](https://github.com/pallabm22/hilabs-hackathon-challenge/blob/main/After-Before%20preprocess-corr.png)


## **Correlation Matrix After Irrelavant Feature Removal**
<img src="https://github.com/pallabm22/hilabs-hackathon-challenge/blob/main/corr-matrix.png" width="700">



## **Model Summary (Pipeline & Architecture)**
<img src="https://github.com/pallabm22/hilabs-hackathon-challenge/blob/main/model_summary.png" width="500">



## Installation

```sh
conda create --name hilabs python=3.10
conda activate hilabs
pip install pandas numpy scikit-learn tensorflow ipykernel
python -m ipykernel install --user --name=Hilabs --display-name "Hilabs"
