# London Weather Temperature Predictor

> Machine Learning model for daily mean temperature prediction in London (1979-2020)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-green.svg)]()

---

## 🎯 Project Overview

This project implements a **Random Forest regression model** to predict daily mean temperature in London using meteorological features, achieving:

- **RMSE:** 2.784°C [95% CI: 2.713-2.853°C]
- **Improvement over baseline:** +24.6% (Seasonal Naive)
- **No temporal leakage:** Uses only features available before prediction

**Key Deliverables:**
- ✅ Production-ready model (`models/random_forest_v1.pkl`)
- ✅ Comprehensive Model Card (`model_card.md`)
- ✅ Deployment guide (`README_deployment.md`)
- ✅ Monitoring criteria (`results/monitoring_criteria.csv`)

---

## 📊 Model Performance

### Test Set Metrics (2015-2020, n=3,061 days)

| Metric | Value | 95% Confidence Interval |
|--------|-------|------------------------|
| **RMSE** | **2.784°C** | [2.713, 2.853]°C |
| **MAE** | **2.208°C** | [2.148, 2.267]°C |
| **R²** | **0.759** | [0.747, 0.771] |

### Performance by Season

| Season | RMSE (°C) | Status |
|--------|-----------|--------|
| Autumn | 2.627 | ✅ Best |
| Summer | 2.722 | ✅ Good |
| Spring | 2.883 | ⚠️ Acceptable |
| Winter | 2.894 | ⚠️ Acceptable |

### Limitations

⚠️ **Extreme temperatures (> 25°C):** RMSE degrades to 6.410°C (only 26 samples)  
⚠️ **Cold temperatures (< 5°C):** RMSE = 3.701°C (33% worse than average)  
✅ **Optimal range (5-15°C):** RMSE = 2.472°C (56% of dataset)

---

## 🗂️ Project Structure

```
project/
├── excersises2.ipynb              # Complete analysis pipeline
├── data/
│   ├── raw/                       # Original dataset
│   └── processed/                 # Processed features
├── models/                        # Trained models
│   ├── random_forest_v1.pkl       # Production (RMSE: 2.784°C)
│   ├── xgboost_v1.pkl             # Benchmark
│   └── ridge_v1.pkl               # Benchmark
├── results/                       # Metrics and analysis
│   ├── final_analysis.csv
│   ├── bootstrap_results.csv
│   ├── error_analysis.csv
│   └── monitoring_criteria.csv
├── [01-24]_*.png                  # Visualizations
├── model_card.md
├── README_deployment.md
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository_url>
cd assestment_cientifico_de_datos

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Load Production Model

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/random_forest_v1.pkl')

# Feature order (MUST match training)
FEATURES = [
    'cloud_cover', 'sunshine', 'global_radiation', 'pressure',
    'precipitation', 'snow_depth', 'season_winter', 'season_spring',
    'season_summer', 'day_of_year', 'year_normalized'
]

# Prepare input
X_new = pd.DataFrame([[5, 3.2, 120.5, 1013.2, 0.0, 0.0, 0, 1, 0, 90, 0.5]], 
                     columns=FEATURES)

# Predict
y_pred = model.predict(X_new)
print(f"Predicted temperature: {y_pred:.2f}°C")
```

#### 2. Run Analysis Notebook

```bash
jupyter notebook excersises2.ipynb
```

---

## 📖 Documentation

- **Model Card:** [`model_card.md`](model_card.md)
- **Deployment Guide:** [`README_deployment.md`](README_deployment.md)
- **Monitoring:** [`results/monitoring_criteria.csv`](results/monitoring_criteria.csv)

---

## 🔬 Technical Highlights

### Rigorous ML Practices

✅ **Temporal validation:** TimeSeriesSplit (5 folds, no random split)  
✅ **Baseline comparison:** Seasonal Naive (RMSE = 3.690°C)  
✅ **Confidence intervals:** Bootstrap (1000 iterations, 95% CI)  
✅ **Leakage detection:** v1 vs v2 documented

### Ethical Trade-off

**Question:** Why not use `max_temp` and `min_temp` features (65% better performance)?

**Answer:** They exhibit **temporal leakage** (same-day measurements).

| Version | RMSE | Status |
|---------|------|--------|
| **v1 (production)** | **2.784°C** | ✅ No leakage |
| v2 (invalid) | 0.966°C | ❌ Temporal leakage |

**Decision:** Accept 188% more error to ensure model integrity.  
**Documentation:** See `model_card.md` § Ethical Considerations

---

## 🛠️ Technology Stack

- **Language:** Python 3.8+
- **ML Framework:** scikit-learn 1.3.0, XGBoost 1.7.6
- **Data:** pandas 2.0.3, numpy 1.24.3
- **Visualization:** matplotlib 3.7.2, seaborn 0.12.2
- **Environment:** Jupyter Notebook

---

## 📈 Next Steps

### Production Deployment
1. Review Model Card (`model_card.md`)
2. Set up monitoring (`results/monitoring_criteria.csv`)
3. Deploy using `README_deployment.md`

### Model Maintenance
- **Retraining:** Every 6 months
- **Alert threshold:** RMSE > 3.18°C
- **Critical threshold:** RMSE > 3.48°C

---

## 👤 Author

**Ismael Alejandro Carreño Diaz**  
Bioinformático | Candidato a Científico de Datos  
Walmart / Stefanini México

---

## 🙏 Acknowledgments

- **Dataset:** London Weather (1979-2020)
- **Methodology:** Model Card framework (Mitchell et al., 2019)
- **Guidelines:** Data Science Standards

---

**Last Updated:** May 24, 2026  
**Model Version:** 1.0.0  
**Status:** ✅ Production-Ready
