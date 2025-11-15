# GPU Energy-Aware Workload Recommendation System
## Exploratory Data Analysis Report

**Author:** Steven  
**Course:** CSC-466 Machine Learning  
**Date:** November 2025

---

## Executive Summary

This analysis combines 90 days of Texas electricity prices with GPU data center usage patterns. We found that scheduling GPU jobs during off-peak hours (midnight-8 AM) costs 69% less than peak hours. The data shows clear patterns that can be used to build a machine learning classifier.

---

## 1. What is This Dataset and Why Did We Choose It?

### Dataset Overview

**Electricity Prices**
- Hourly prices from Texas power grid
- 90 days of data (Aug 14 - Nov 12, 2025)
- Prices range from $15 to $772 per megawatt-hour

**GPU Data Center Usage**
- 100-GPU cluster simulation
- Tracks power use, jobs running, and utilization
- Based on real NVIDIA A100 GPU specs

| What We Have | Amount |
|--------------|--------|
| Total hours of data | 2,161 |
| Features (variables) | 20 |
| Missing data | 0 |
| Efficient vs inefficient split | 50% / 50% |

### Why This Matters

GPU data centers use massive amounts of electricity. Many GPU tasks (like training AI models or running simulations) don't need to happen immediately - they can wait for cheaper electricity times. This project builds a system to predict when to run these jobs to save money.

---

## 2. What Did We Learn?

### Finding 1: Huge Cost Difference Between Time Windows

<p align="center">
  <img src="target_variable_analysis.png" width="800" alt="Cost Comparison">
</p>

| Metric | Cheap Hours | Expensive Hours |
|--------|-------------|-----------------|
| Electricity price | $38/MWh | $79/MWh |
| Cost per hour | $0.46 | $1.51 |
| Jobs per dollar | 271 | 82 |

**Key takeaway:** Running jobs at night is 69% cheaper and completes 3.28× more work per dollar.

### Finding 2: Clear Time Patterns

<p align="center">
  <img src="daily_weekly_patterns.png" width="800" alt="Time Patterns">
</p>

**Best times to run jobs:**
- Midnight to 8 AM (cheapest)
- 10 PM to 11 PM

**Worst times:**
- 9 AM to 9 PM (expensive)
- 4 PM to 5 PM (most expensive)

### Finding 3: What Predicts Efficiency

<p align="center">
  <img src="correlation_heatmap.png" width="700" alt="Feature Relationships">
</p>

**Strongest predictors of cheap vs expensive hours:**
- Electricity price (obviously)
- Time of day
- How much power GPUs are using
- Whether it's business hours

These aren't just obvious - they show HOW MUCH each factor matters, which helps build a smarter model.

### Finding 4: It's Not Just Price OR Time - It's Both

<p align="center">
  <img src="cost_efficiency_analysis.png" width="800" alt="Price and Time Together">
</p>

Sometimes night hours are expensive (price spikes). Sometimes day hours are cheap (low demand days). A simple rule like "always run at night" isn't good enough - we need machine learning to handle the complexity.

### Finding 5: Electricity Prices Are Volatile

<p align="center">
  <img src="electricity_prices_timeseries.png" width="800" alt="Price Over Time">
</p>

- Average price: $58/MWh
- But prices swing wildly
- 2% of hours have EXTREME spikes (5-10× normal price)

These spikes happen during grid stress (equipment failures, heat waves, etc.). The system needs to avoid scheduling during these events.

### Finding 6: Data Distributions

<p align="center">
  <img src="distributions.png" width="800" alt="Data Distributions">
</p>

Shows how electricity prices, GPU usage, costs, and efficiency are distributed. Most hours are normal, but there are outliers we need to handle carefully.

---

## 3. What Problems Did We Find?

**Problem 1: Extreme Price Spikes**
- 2% of hours have crazy expensive prices
- Solution: Flag these events and never schedule during them

**Problem 2: Time Series Data**
- Each hour isn't independent (patterns repeat daily/weekly)
- Solution: Test the model on future data, not random data

**Problem 3: Related Features**
- Some variables are highly related (power use ↔ GPU utilization)
- Solution: Keep the most important ones, drop redundant ones

**Problem 4: Only 90 Days of Data**
- Missing winter/summer extremes
- Solution: Focus on time-of-day patterns that work year-round

**Problem 5: What About Weird Situations?**
- How do we handle never-before-seen conditions?
- Solution: If the model isn't confident, use simple backup rules

### Data Limitations

- GPU data is simulated (but realistic)
- Only covers Houston area
- Treats all jobs as flexible (reality: some jobs are urgent)

---

## 4. The Target Variable

### What We're Trying to Predict

```python
efficiency = jobs_completed / cost
is_efficient = (efficiency > median)
```

We label each hour as "efficient" (good time to run jobs) or "inefficient" (bad time).

**Why this works:**
- We predict efficiency FROM price and time
- We don't cluster ON efficiency (that would be circular)
- Clean 50/50 split (balanced data)

---

## 5. Features We Created

We engineered 6 new features from the raw data:

1. **Price category** - Low/Medium/High buckets
2. **Business hours flag** - Is it 8 AM-6 PM on a weekday?
3. **Peak hours flag** - Is it 2-6 PM (highest demand)?
4. **Utilization level** - Low/Medium/High GPU usage
5. **Efficiency label** - Our TARGET (what we predict)
6. **24-hour price average** - Smooths out short-term spikes

**What goes into the model:**
- Electricity price
- Hour of day
- Day of week
- Business hours flag
- Peak hours flag
- Power consumption
- 24-hour price average

**What we exclude:**
- Cost (it's just price × power)
- Efficiency metric (that's what we're predicting)
- Redundant GPU metrics

---

## 6. Next Steps

### Build Models

**Simple models first:**
- Logistic Regression
- Decision Tree

**Better models:**
- Random Forest
- XGBoost

### Test Properly

- Use time-based testing (train on old data, test on new data)
- Target: 80%+ precision, 75%+ recall

### Build Recommendation System

**How it works:**
- Input: Current price, time, GPU state
- Output: "Run now" or "Wait"

**Decision rules:**
- Model says 70%+ confident → Run now
- Model says 30-70% confident → Wait and check again
- Model says <30% confident → Don't run

### Expected Results

- 78-85% accuracy
- 40-50% cost reduction
- $100K-$120K annual savings for 100-GPU cluster

---

## 7. Conclusion

**What we proved:**
- Data is high quality (no missing values, balanced classes)
- Clear patterns exist (night is 69% cheaper)
- Strong predictive signals (correlations up to -0.545)
- Problem is solvable with machine learning

**Ready to build:**
- Classification models
- Recommendation system
- Real-time scheduling tool

---

## All Visualizations

<p align="center">
  <img src="simple_scatter_efficiency.png" width="400" alt="Efficiency Scatter">
  <img src="boxplots_comparison.png" width="400" alt="Box Plots">
</p>

<p align="center">
  <img src="kmeans_elbow_plot.png" width="400" alt="Clustering">
  <img src="pca_analysis.png" width="400" alt="PCA">
</p>

<p align="center">
  <img src="hourly_averages.png" width="400" alt="Hourly Patterns">
  <img src="weekly_patterns.png" width="400" alt="Weekly Patterns">
</p>

**Deliverables:** 12+ visualizations, 4 datasets, reproducible code