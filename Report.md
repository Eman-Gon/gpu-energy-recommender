cd ~/csc466/gpu-energy-recommender

# Create the REPORT.md
cat > REPORT.md << 'EOF'
# GPU Energy-Aware Workload Recommendation System
## Exploratory Data Analysis Report

**Author:** Steven  
**Course:** CSC-466 Machine Learning  
**Institution:** California Polytechnic State University, San Luis Obispo  
**Date:** November 2025

---

## Executive Summary

This report presents a comprehensive exploratory data analysis for a GPU workload recommendation system that optimizes scheduling based on electricity prices. By analyzing 90 days of ERCOT Texas energy market data merged with GPU cluster utilization metrics, we identified a **69% cost reduction opportunity**.

**Key Finding:** GPU workloads scheduled during off-peak hours (midnight-8 AM) achieve **3.28x better cost efficiency** than business hour scheduling.

---

## 1. Dataset Overview

### 1.1 What is the Dataset?

This project combines two datasets:

1. **ERCOT Electricity Prices**
   - 90 days of hourly electricity prices from Texas grid
   - Settlement Point: HB_NORTH (Houston area)
   - Price range: $15-$772/MWh
   - Based on real ERCOT market patterns

2. **GPU Cluster Utilization**
   - 100-GPU cluster metrics
   - Power consumption, job counts, utilization
   - Realistic patterns based on NVIDIA specifications

### 1.2 Why This Dataset?

**Real-World Impact:** Data centers consume 1-2% of global electricity. GPU clusters are particularly power-intensive. By intelligently scheduling deferrable workloads (AI training, batch processing) during low-price periods, organizations can:
- ✅ Reduce operating costs by 48-69%
- ✅ Support renewable energy integration
- ✅ Lower carbon footprint

**Novel Combination:** Combining ERCOT energy market data with GPU workload optimization is a novel approach not extensively studied.

### 1.3 Dataset Statistics

| Attribute | Value |
|-----------|-------|
| **Total Records** | 2,161 hourly observations |
| **Time Span** | 90 days (Aug 14 - Nov 12, 2025) |
| **Features** | 20 (14 raw + 6 engineered) |
| **Missing Values** | 0 (0.0%) ✅ |
| **Target Balance** | 50.0% / 50.0% (perfect) ✅ |
| **Data Quality** | Excellent |

---

## 2. What Did We Learn from EDA?

### 2.1 Massive Cost Savings Opportunity

![Target Variable Analysis](eda/target_variable_analysis.png)

**Key Discovery:**
- **Efficient hours:** $0.46/hour average cost
- **Inefficient hours:** $1.51/hour average cost  
- **69% cost reduction** by smart scheduling
- **3.28x better** jobs-per-dollar efficiency

### 2.2 Clear Temporal Patterns

![Daily and Weekly Patterns](eda/daily_weekly_patterns.png)

**Hourly Efficiency Patterns:**

✅ **Best Hours (60-85% efficient):**
- Midnight - 8:00 AM (off-peak)
- 10:00 PM - 11:00 PM (late evening)

❌ **Worst Hours (26-38% efficient):**
- 9:00 AM - 9:00 PM (business hours)
- 4:00 PM - 5:00 PM (peak demand)

**Insight:** Off-peak hours have 52% cheaper electricity and 69% lower operating costs!

### 2.3 Price Volatility Analysis

![Electricity Price Time Series](eda/electricity_prices_timeseries.png)

**Price Patterns Discovered:**
- Average price: $58/MWh
- Off-peak average: $35/MWh
- Peak average: $80/MWh
- **2% extreme spike events** (>$270/MWh) - grid stress periods

**24-hour moving average** smooths daily fluctuations and reveals weekly trends.

### 2.4 Feature Relationships

![Correlation Heatmap](eda/correlation_heatmap.png)

**Strong Predictors of Efficiency:**
- `power_consumption_kw`: -0.545 ⭐
- `gpu_utilization_pct`: -0.524 ⭐
- `hourly_cost_usd`: -0.503 ⭐
- `price_mwh`: -0.432 ⭐
- `is_business_hours`: -0.382 ⭐

**Negative correlations mean:** Lower values = More likely efficient  
(Lower price, lower utilization during cheap hours = high efficiency)

### 2.5 Cost Efficiency Relationships

![Cost Efficiency Analysis](eda/cost_efficiency_analysis.png)

**Left plot:** Price vs Utilization (colored by hourly cost)
- Shows that high utilization during high-price hours = expensive

**Right plot:** Price vs Jobs per Dollar (colored by hour)
- Late night/early morning hours (blue) cluster in high-efficiency region
- Business hours (yellow/red) cluster in low-efficiency region

### 2.6 Statistical Distributions

![Distributions](eda/distributions.png)

**Key Observations:**
- **Price distribution:** Positively skewed (occasional extreme spikes)
- **Utilization:** Nearly normal distribution around 53% mean
- **Cost:** Right-skewed, most hours <$2, outliers up to $16
- **Jobs per dollar:** Highly right-skewed, median = 124

---

## 3. What Issues or Open Questions Remain?

### 3.1 Challenges Identified

**1. Extreme Price Spikes (2% of hours)**
- **Issue:** Prices occasionally jump 5-10× normal
- **Solution:** Implement robust scaling, add "extreme event" feature, penalty during spikes

**2. Temporal Dependencies**
- **Issue:** Time series data, not independent samples
- **Solution:** Use time-aware train/test split, time series cross-validation

**3. Feature Multicollinearity**
- **Issue:** High correlation between GPU metrics (r > 0.9)
- **Solution:** Feature selection with L1 regularization, use tree-based models

**4. Limited Temporal Coverage**
- **Issue:** Only 90 days (one season)
- **Solution:** Focus on hourly/daily patterns, collect more data for production

**5. Cold Start Problem**
- **Issue:** How to recommend for new/unprecedented conditions?
- **Solution:** Use time features for generalization, fallback to price-only heuristics

### 3.2 Data Limitations

**Simulated GPU Data:**
- GPU utilization is synthetic (not from real production cluster)
- Patterns based on realistic data center studies
- **Justification:** Serves well for proof-of-concept; can retrain on real data

**Single Geographic Zone:**
- Only Houston (HB_NORTH) settlement point
- **Generalization:** Approach works for other ERCOT zones and markets

**No Job Priorities:**
- All jobs treated equally (no urgent vs deferrable distinction)
- **Future Work:** Add priority weights and deadline constraints

---

## 4. Target Variable: `is_efficient_time`

### 4.1 Definition
```python
is_efficient_time = (jobs_per_dollar > median(jobs_per_dollar))
```

**Threshold:** 124 jobs per dollar

### 4.2 Characteristics

| Metric | Efficient (1) | Inefficient (0) | Difference |
|--------|---------------|-----------------|------------|
| **Electricity Price** | $37.54/MWh | $78.66/MWh | 52% cheaper |
| **Hourly Cost** | $0.46 | $1.51 | 69% cheaper |
| **GPU Utilization** | 43.3% | 63.4% | 20 pp lower |
| **Jobs per Dollar** | 271.04 | 82.58 | 3.28x better |

**Perfect Balance:** 1,080 efficient / 1,081 inefficient (50/50 split)

---

## 5. Feature Engineering

### 5.1 Engineered Features

Created **6 new features:**

1. **`price_category`** - Low/Medium/High price bins
2. **`is_business_hours`** - Binary flag (8 AM - 6 PM weekdays)
3. **`is_peak_hours`** - Binary flag (2 PM - 6 PM)
4. **`utilization_level`** - Low/Medium/High utilization bins
5. **`is_efficient_time`** - **TARGET VARIABLE** (binary)
6. **`price_rolling_mean_24h`** - 24-hour moving average price

### 5.2 Feature Importance Preview

For classification models, expected important features:
1. `price_mwh` (strong negative correlation)
2. `hour` (clear temporal pattern)
3. `power_consumption_kw` (proxy for cost)
4. `is_business_hours` (captures efficiency window)
5. `gpu_utilization_pct` (workload indicator)

---

## 6. Next Steps

### 6.1 Model Development Plan

**Phase 2: Classification Models**
- Train Random Forest, XGBoost, Logistic Regression
- Predict `is_efficient_time` (0 or 1)
- Target metrics: >80% precision, >75% recall, >77% F1

**Phase 3: Recommendation System**
- Content-based filtering using job + time features
- Optimization layer for workload-to-window assignment
- Real-time decision support: "Schedule now?" → Yes/No

### 6.2 Expected Results

**Model Performance:**
- High accuracy expected (strong predictive signals)
- Clear feature importance (interpretable)
- Time-based validation (no data leakage)

**Business Impact:**
- 40-50% reduction in GPU energy costs
- Annual savings: $9,000-$90,000 (depending on cluster size)
- Improved sustainability metrics

---

## 7. Conclusion

This EDA demonstrates:

✅ **High-quality dataset** - Zero missing values, perfect balance  
✅ **Clear patterns** - 69% cost savings opportunity identified  
✅ **Strong predictive signals** - Multiple features correlate with efficiency  
✅ **Actionable insights** - Avoid 9 AM-9 PM scheduling  
✅ **Novel approach** - Energy market + GPU optimization  

The dataset is **ready for classification modeling** with high expected accuracy due to clear temporal patterns and strong feature correlations.

**Bottom Line:** By scheduling GPU workloads during off-peak hours (midnight-8 AM), organizations can achieve **3.28x better cost efficiency** and reduce operating costs by **69%**.

---

## References

1. ERCOT - Electric Reliability Council of Texas - https://www.ercot.com/
2. NREL - Data Center Energy Report - https://www.nrel.gov/
3. NVIDIA - Data Center GPU Documentation - https://www.nvidia.com/

---

**Files Generated:**
- 6 visualizations (300 DPI PNG)
- 4 datasets (raw + processed CSV)
- Enhanced dataset with 20 features
- Complete Python scripts for reproducibility

**GitHub Repository:** [Link to be added]
EOF

echo "✅ REPORT.md created!"