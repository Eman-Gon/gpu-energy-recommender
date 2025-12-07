# GPU Energy-Aware Workload Scheduling: A Machine Learning Approach

**Emanuel Gonzalez**  
California Polytechnic State University, San Luis Obispo  
CSC-466 Knowledge Discovery from Data  
December 2025

## The Idea

Running GPU workloads in data centers costs money, and electricity prices change dramatically throughout the day. I show that machine learning is needed to find the best times to schedule work. I trained four models on 2,161 hours of electricity price and GPU usage data, achieving 97.4% accuracy. This beats simple rules by 19-33 percentage points. I found that 31% of daytime hours are actually efficient, with the best daytime hours being 68× better than the worst nighttime hours.

## Intro

Large GPU clusters use huge amounts of electricity. In Texas (ERCOT market), prices range from $15/MWh to $772/MWh throughout the day. For a 10,000-GPU cluster, poor scheduling can waste over $2,000 per hour.

The simple approach of "just run jobs at night when electricity is cheap" has problems. Nighttime can be expensive during high demand, and daytime sometimes offers good opportunities. These patterns are too complex for simple rules.

**Question:** Can machine learning find better scheduling patterns than simple rules?


### Dataset

I combined two data sources over 90 days:
- **Electricity Prices:** Real hourly prices from Houston, Texas ($15-$772/MWh)
- **GPU Usage:** Simulated 10,000-GPU cluster data (300W per GPU)

Total: 2,161 hourly observations, no missing data.

### Features and Target

**17 Features:** electricity price, total cost, power usage, GPU utilization, active jobs, hour of day, day of week, weekday/weekend, business hours, peak hours, and advanced combinations.

**Target:** Hours labeled "efficient" (1) or "inefficient" (0) based on jobs completed per dollar spent. Median threshold: 124 jobs/dollar. Perfectly balanced: 1,080 efficient, 1,081 inefficient.

### Models

Four algorithms using 75/25 train/test split: Logistic Regression, Decision Tree, Random Forest (100 trees), and XGBoost.

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **Logistic Regression** | **97.4%** | **97.7%** | **97.0%** | **97.4%** |
| XGBoost | 97.2% | 98.1% | 96.3% | 97.2% |
| Random Forest | 96.3% | 97.3% | 95.1% | 96.2% |
| Decision Tree | 96.3% | 95.2% | 97.4% | 96.3% |

Logistic Regression performed best with only 14 errors out of 541 test samples (2.6% error rate).

### Comparison to Simple Rules

| Strategy | Accuracy |
|----------|----------|
| "Run at night (0-8am)" | 64.2% |
| "Run when price < $44/MWh" | 78.4% |
| **ML (Logistic Regression)** | **97.4%** |

Machine learning improves 19-33 points over simple rules.

### Why Simple Rules Fail

![Price Anomalies](results/plots/chart1_price_anomalies.png)  
*Figure 1: Green stars show 128 cheap daytime opportunities simple rules miss. Red X's show 40 expensive nighttime hours wrongly scheduled.*

**The "Run at Night" Rule makes 774 mistakes:**
- **567 missed opportunities:** Efficient daytime hours ignored (avg: $41.60/MWh, 45% util, 104 jobs)
- **207 bad schedules:** Expensive nighttime hours used (avg: $49.51/MWh, 57% util, 66 jobs)

**Key Findings:**
- 29% of nighttime hours are inefficient
- 31% of daytime hours are efficient
- Weekends differ: 66% Saturday, 71% Sunday daytime efficient
- **Best daytime is 68× better than worst nighttime** (5,900 vs 86 jobs/dollar)

### Feature Importance

![Feature Importance](results/plots/comparison.png)  
*Figure 2: Economic and workload features dominate.*

**Top 5 Features:**
1. `hourly_cost_usd` (34%) - Price × utilization
2. `active_jobs` (19%) - Work available
3. `price_mwh` (16%) - Electricity price
4. `power_consumption_kw` (9%) - Energy use
5. `gpu_utilization_pct` (6%) - GPU busy-ness

Economic features: 50%, workload features: 31%. Proves complexity beyond "just check price."

### Validation

![Confusion Matrices](results/plots/confusion_matrices.png)  
*Figure 3: All models achieve >95% accuracy.*

McNemar's test confirms results:
- **P-value:** <0.001 (highly significant)
- **95% CI (ML):** 96.5%-98.4%
- **95% CI (Baseline):** 60.3%-68.0%

Non-overlapping intervals prove ML is definitively better.

### Complexity Score: 9/9

1. Daytime efficiency 31% (>20% threshold) - 3 points
2. Nighttime variability 71% (<85% threshold) - 3 points
3. ML improvement 19 points (>15 threshold) - 3 points

## Discussion

### Why ML Works

GPU scheduling needs machine learning because:
1. **Complex relationships:** Efficiency = price × utilization × jobs
2. **Context matters:** Same hour varies by conditions
3. **Many features:** 17 features interact beyond simple rules

All four algorithms >96% proves patterns are real.

### Limitations

- 90 days might miss seasonal patterns
- Simulated GPU data, not real cluster
- Default model settings, no optimization
- Could add weather/grid data

### Future Work

1. Deep learning (LSTM/Transformers) for time patterns
2. Cost-sensitive learning for expensive errors
3. Multi-class: "run now" vs "wait 2-4hrs" vs "wait until night"
4. SHAP explainability for operators
5. Reinforcement learning for sequential decisions

## Conclusion

This proves GPU scheduling needs machine learning. Models achieve 97.4% accuracy, beating simple rules by 19-33 points. Finding 31% of daytime efficient—with 68× variance—shows complexity beyond "run at night."

Statistical tests (p<0.001), 9/9 complexity score, and four algorithms agreeing prove real patterns. Economic factors dominate (50%) but workload/time add value (40%).

Simple rules make 774 mistakes: miss 567 good daytime hours, wrongly schedule 207 expensive nighttime hours. For large GPU operations, ML saves money through better scheduling: 97% accuracy, 19-33 point improvement, statistically proven.