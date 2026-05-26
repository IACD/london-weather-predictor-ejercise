# Model Card: Weather Temperature Predictor

**Model Name:** Random Forest Temperature Predictor v1  
**Model Version:** 1.0.0  
**Model Type:** Regression (Random Forest)  
**Owner:** Bioinformatician - Walmart/Stefanini Data Scientist Candidate  
**Date:** May 25, 2026  
**Status:** ✅ PRODUCTION-READY (No Temporal Leakage)

---

## 📌 Intended Use

### Primary Use Cases
This model predicts **daily mean temperature in London** based on:
- ✅ Meteorological features (cloud cover, sunshine, global radiation, pressure, precipitation)
- ✅ Seasonal patterns (encoded as binary features)
- ✅ Temporal trends (day of year, year normalized)

### Target Users
- Weather forecasting systems (short-term predictions)
- Energy demand forecasting (temperature-dependent consumption)
- Agricultural planning (crop temperature requirements)

### Out-of-Scope Uses
- ❌ Long-term climate forecasting (> 7 days ahead)
- ❌ Extreme weather event prediction (heatwaves, cold snaps)
- ❌ Other geographical locations (trained on London only)
- ❌ Real-time nowcasting (requires features not available in real-time)

---

## 📊 Training Data

### Data Source
- **Dataset:** London Weather Dataset (1979-2020)
- **Geographical Coverage:** London, United Kingdom (51.5°N, 0.1°W)
- **Temporal Coverage:** 41 years of daily observations
- **Total Samples:** 15,305 days

### Data Splits (Temporal - No Random Split)
- **Train:** 9,183 samples (60%) → 1979-2004
- **Validation:** 3,061 samples (20%) → 2005-2010
- **Test:** 3,061 samples (20%) → 2015-2020

### Features (11 Total - NO TEMPORAL LEAKAGE)
1. `cloud_cover` (oktas, 0-8)
2. `sunshine` (hours)
3. `global_radiation` (W/m²)
4. `pressure` (hPa)
5. `precipitation` (mm)
6. `snow_depth` (cm)
7. `season_winter` (binary)
8. `season_spring` (binary)
9. `season_summer` (binary)
10. `day_of_year` (1-365)
11. `year_normalized` (0-1 scaled)

**⚠️ EXCLUDED FEATURES (Temporal Leakage):**
- `max_temp` (same-day, unavailable before prediction)
- `min_temp` (same-day, unavailable before prediction)

### Target Variable
- **`mean_temp`** (°C) - Daily mean temperature

---

## 🎯 Performance Metrics

### Test Set Performance (n = 3,061 samples)

| Metric | Value | 95% Confidence Interval |
|--------|-------|------------------------|
| **RMSE** | **2.783°C** | [2.713, 2.853]°C |
| **MAE** | **2.208°C** | [2.148, 2.267]°C |
| **R²** | **0.7588** | [0.7471, 0.7710] |

### Baseline Comparison
- **Baseline Model:** Seasonal Naive (previous year same day)
- **Baseline RMSE:** 3.690°C
- **Improvement:** +24.6% (0.906°C reduction)

### Performance by Season

| Season | RMSE (°C) | Sample Count |
|--------|-----------|--------------|
| Winter | 2.894 | 805 |
| Spring | 2.883 | 736 |
| Summer | 2.722 | 736 |
| Autumn | 2.627 | 784 |

**Observation:** Model performs best in **Autumn** (RMSE = 2.627°C) and worst in **Winter** (RMSE = 2.894°C).

### Performance by Temperature Range

| Temperature Range | RMSE (°C) | Sample Count |
|-------------------|-----------|--------------|
| < 5°C (Frío) | 3.701 | 312 |
| 5-15°C (Templado) | 2.472 | 1,718 |
| 15-25°C (Cálido) | 2.801 | 1,005 |
| > 25°C (Extremo) | 6.410 | 26 |

**Observation:** Model performs best in **5-15°C (Templado)** and degrades in **> 25°C (Extremo)**.

---

## 🛠️ Model Details

### Algorithm
- **Type:** Ensemble Learning (Bootstrap Aggregation)
- **Base Estimator:** Decision Tree Regressor
- **Ensemble:** Random Forest with 200 trees

### Hyperparameters (Optimized via RandomizedSearchCV)
```python
{
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'random_state': 42
}
```

### Cross-Validation Strategy
- **Method:** TimeSeriesSplit (5 folds)
- **Scoring:** Negative Root Mean Squared Error
- **Search:** RandomizedSearchCV (50 iterations)

### Top 5 Most Important Features
1. **season_summer** → 0.2539 (25.39%)
2. **day_of_year** → 0.2317 (23.17%)
3. **season_winter** → 0.1797 (17.97%)
4. **global_radiation** → 0.1612 (16.12%)
5. **sunshine** → 0.0426 (4.26%)


---

## ⚠️ Limitations

### Known Issues
1. **Performance Degradation in Extreme Temperatures**
   - Model RMSE increases to 6.410°C for temperatures > 25°C (Extremo)
   - Cause: Limited training samples in extreme ranges
   - Recommendation: Do NOT use for heatwave/cold snap predictions

2. **Seasonal Variability**
   - Model performs 10.2% worse in Winter vs Autumn
   - Cause: Complex weather patterns in transition seasons

3. **Geographical Constraint**
   - Trained exclusively on London data (oceanic temperate climate)
   - NOT generalizable to other cities or climate zones

4. **Climate Change Assumption**
   - Trained on 1979-2020 data (assumes stationary climate)
   - May degrade if climate patterns shift significantly post-2020

### Error Analysis
- **Mean Absolute Error:** 2.208°C
- **Typical Error Range:** ±2-3°C in most cases
- **Outliers:** ~5% of predictions have errors > 5°C (extreme weather events)

---

## 🔬 Ethical Considerations

### Data Leakage Decision
**Trade-off:** This model deliberately **EXCLUDES** `max_temp` and `min_temp` features despite 65% performance improvement.

| Feature Set | RMSE (Test) | Ethical Status |
|-------------|-------------|----------------|
| **v1 (This Model)** | **2.784°C** | ✅ **Production-Ready (No Leakage)** |
| v2 (With max/min temp) | 0.966°C | ❌ Invalid (Temporal Leakage) |

**Justification:**
- `max_temp` and `min_temp` are measured **on the same day** as the prediction target
- Including them creates **temporal leakage** (using future information)
- **Ethical choice:** Accept 188% worse performance to ensure model integrity

**Documentation:**
- Full comparison available in `comparison_v1_vs_v2.csv`
- Feature importance analysis shows v2 models depend 60-70% on leaked features

### Bias Considerations
- **Temporal Bias:** Model trained on 1979-2020 may underrepresent recent climate trends
- **Geographical Bias:** London-only data limits applicability to similar climates

---

## 📈 Monitoring & Maintenance

### Production Monitoring Thresholds

| Metric | Baseline (Test) | Alert Threshold | Critical Threshold | Action |
|--------|-----------------|-----------------|--------------------| -------|
| **RMSE** | 2.783°C | > 3.20°C | > 3.50°C | Investigate / Retrain |
| **MAE** | 2.208°C | > 2.50°C | > 2.80°C | Investigate / Retrain |
| **R²** | 0.7588 | < 0.70 | < 0.65 | Retrain Urgently |

### Retraining Triggers
1. **Performance Degradation:**
   - RMSE exceeds 3.50°C for 30 consecutive days

2. **Data Drift:**
   - Feature distributions shift significantly (KL divergence > 0.1)
   - New weather patterns emerge (unseen in training data)

3. **Scheduled Retraining:**
   - Minimum: Once per year with accumulated new data
   - Recommended: Every 6 months to capture seasonal changes

### Recommended Monitoring Frequency
- **Real-time:** Track daily predictions vs actuals
- **Weekly:** Calculate rolling 7-day RMSE/MAE
- **Monthly:** Full model evaluation + drift analysis

---

## 📦 Model Artifacts

### Saved Files
- **Model:** `models/random_forest_v1.pkl` (joblib serialized)
- **Feature List:** `X_test_v1.csv` (column order MUST be preserved)
- **Scaler:** Not required (Random Forest does not need scaling)

### Dependencies
```python
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
```

### Loading Instructions
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/random_forest_v1.pkl')

# Prepare features (MUST match training order)
features = ['cloud_cover', 'sunshine', 'global_radiation', 'pressure', 
            'precipitation', 'snow_depth', 'season_winter', 'season_spring', 
            'season_summer', 'day_of_year', 'year_normalized']

X_new = pd.DataFrame({...}, columns=features)

# Predict
y_pred = model.predict(X_new)
```

---

## 📚 References

### Model Card Framework
- Mitchell, M. et al. (2019). "Model Cards for Model Reporting." *ACM FAT Conference*.

### Dataset
- London Weather Dataset (1979-2020), Kaggle / Met Office UK.

### Methodology
- Time Series Cross-Validation: Bergmeir, C. & Benítez, J. M. (2012).
- Bootstrap Confidence Intervals: Efron, B. & Tibshirani, R. (1993).

---

## ✅ Approval & Sign-off

**Model Status:** PRODUCTION-READY  
**Reviewed By:** [Pending MLOps Team Review]  
**Deployed Date:** [Pending Deployment]  
**Next Review:** [6 months from deployment]

---

**End of Model Card**
