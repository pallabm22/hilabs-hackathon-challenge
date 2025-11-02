# 🩺 Healthcare Risk Prediction
## _Proactive Patient Risk Scoring with ML & PCA_

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-green.svg)]()
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A data-driven system for predicting **patient risk scores** by integrating multi-domain healthcare data, performing correlation-based feature engineering, applying **PCA**, and training machine learning & deep learning models.

- Clean & merge healthcare datasets  
- Engineer clinical + visit + care + diagnosis features  
- Perform correlation analysis & PCA  
- Train ML & NN models for risk score prediction  
- ✨ Actionable insights for smarter healthcare ✨

---

## Features

- 🔗 Multi-source healthcare dataset integration  
- 🧼 Advanced preprocessing & feature engineering  
- 📊 Correlation-based feature selection  
- 🧠 ML + Neural Network training pipelines  
- 📉 PCA for 95% variance retention  
- ✅ Model evaluation using R², MAE, RMSE  
- 📦 Pipeline for both training & inference  

> Healthcare risk prediction that balances **accuracy + interpretability**

---

## Why This Matters

Healthcare systems often react **after** patients deteriorate.  
This project enables **proactive intervention** by predicting risk early — using structured medical data + machine learning.

> “Turning medical data into intelligent care decisions.”

---

## Tech

This project leverages modern ML & data tools:

- 🐍 Python 3.10  
- 📦 Pandas & NumPy — data processing  
- 🤖 scikit-learn — ML + PCA + pipelines  
- 🧠 TensorFlow — neural network baseline  
- 📊 Matplotlib & Seaborn — visualizations  
- 🧮 Graphviz / Pydot — model pipelines  

---

### 📎 Reference Visuals

**Before & After Preprocessing Correlation**
![Before and After Correlation](https://github.com/pallabm22/hilabs-hackathon-challenge/blob/main/After-Before%20preprocess-corr.png)

**Correlation Matrix After Feature Removal**
<img src="https://github.com/pallabm22/hilabs-hackathon-challenge/blob/main/corr-matrix.png" width="700">

**Model Summary (Pipeline & Architecture)**
<img src="https://github.com/pallabm22/hilabs-hackathon-challenge/blob/main/model_summary.png" width="500">



## Installation

You need **Conda (recommended)** or Python 3.10+.

```sh
conda create --name hilabs python=3.10
conda activate hilabs
pip install pandas numpy scikit-learn tensorflow pydot graphviz ipykernel
python -m ipykernel install --user --name=Hilabs --display-name "Hilabs"
