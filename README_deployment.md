# Deployment Guide: Random Forest Temperature Predictor v1

## Quick Start

### Prerequisites
```bash
Python 3.8+
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
```

### Installation
```bash
pip install scikit-learn==1.3.0 pandas==2.0.3 numpy==1.24.3 joblib==1.3.2
```

---

## Model Loading

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/random_forest_v1.pkl')

# Expected features (ORDER MATTERS)
FEATURE_ORDER = [
    'cloud_cover',
    'sunshine',
    'global_radiation',
    'pressure',
    'precipitation',
    'snow_depth',
    'season_winter',
    'season_spring',
    'season_summer',
    'day_of_year',
    'year_normalized'
]
```

---

## Inference Example

```python
# Prepare input data
data = {
    'cloud_cover': 5,
    'sunshine': 3.2,
    'global_radiation': 120.5,
    'pressure': 1013.2,
    'precipitation': 0.0,
    'snow_depth': 0.0,
    'season_winter': 0,
    'season_spring': 1,
    'season_summer': 0,
    'day_of_year': 90,
    'year_normalized': 0.5
}

X_input = pd.DataFrame([data], columns=FEATURE_ORDER)

# Predict
y_pred = model.predict(X_input)

print(f"Predicted mean temperature: {y_pred:.2f}°C")
```

---

## Performance Monitoring

### Expected Performance (Test Set 2015-2020)
- **RMSE:** 2.783°C [95% CI: 2.713-2.853°C]
- **MAE:** 2.208°C
- **R²:** 0.7588

### Alert Thresholds
| Metric | Alert | Critical | Action |
|--------|-------|----------|--------|
| RMSE | > 3.18°C | > 3.48°C | Investigate / Retrain |
| MAE | > 2.51°C | > 2.81°C | Investigate / Retrain |
| R² | < 0.70 | < 0.65 | Retrain Urgently |

---

## Support

**Model Owner:** Bioinformatician - Walmart/Stefanini Data Scientist Candidate  
**Documentation:** `model_card.md`  
**Performance Metrics:** `results/final_analysis.csv`  
**Error Analysis:** `results/error_analysis.csv`

---

**Last Updated:** May 24, 2026  
**Model Version:** 1.0.0  
**Status:** PRODUCTION-READY
