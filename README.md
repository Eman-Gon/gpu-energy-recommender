## Project Overview

This project develops a machine learning recommendation system that suggests optimal GPU workload scheduling by analyzing ERCOT Texas electricity prices and GPU cluster utilization patterns. 

**Goal:** Minimize energy costs by 69% while maintaining system efficiency.

### Key Results

- 69% cost reduction during efficient hours ($1.51 → $0.46/hour)
- 52% cheaper electricity during off-peak times
- 3.28x better jobs-per-dollar efficiency
- Perfect data quality (0 missing values, 50/50 class balance)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone [your-repo-url]
cd gpu-energy-recommender

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Navigate to EDA folder
cd eda

# Generate datasets
python data_collection.py

# Run main exploratory data analysis
python eda.py

# Run target variable analysis
python Analyze_target_variable.py

# Generate additional visualizations
python additional_visualizations.py
```

### Results

All analysis results are in the `eda/` folder:
- `eda_summary.md` - Complete 7-section analysis report
- `*.png` - 12+ professional visualizations (300 DPI)
- `*.csv` - Generated datasets (raw + enhanced)

For detailed findings, see [eda/eda_summary.md](eda/eda_summary.md)

---

## Key Findings

<p align="center">
  <img src="eda/target_variable_analysis.png" width="850" alt="Target Variable Analysis">
</p>

### Optimal Scheduling Windows

**Schedule GPU jobs:**
- Midnight - 8:00 AM (60-85% efficient)
- 10:00 PM - 11:00 PM (70-75% efficient)

**Avoid scheduling:**
- 9:00 AM - 9:00 PM (26-38% efficient)
- 4:00 PM - 5:00 PM (peak hours, most expensive)

---

## Dataset

| Metric | Value |
|--------|-------|
| **Time Period** | 90 days (Aug 14 - Nov 12, 2025) |
| **Records** | 2,161 hourly observations |
| **Features** | 20 (14 raw + 6 engineered) |
| **Missing Values** | 0 |
| **Target Balance** | 50.0% / 50.0% |

**Data Sources:**
1. ERCOT electricity prices (Texas energy market)
2. GPU cluster utilization (100-GPU cluster, realistic patterns)

---


## Technologies Used

- **Python 3.10+**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib** - Visualizations
- **seaborn** - Statistical graphics
- **scikit-learn** - Feature engineering

---

## Next Steps

1. **Model Development** - Train Random Forest, XGBoost, Logistic Regression classifiers
2. **Evaluation** - Target >80% precision, >75% recall
3. **Recommendation System** - Build real-time decision support
4. **Business Impact** - Deploy for 40-50% energy cost reduction

---

## Course Information

**Course:** CSC-466 Machine Learning  
**Institution:** California Polytechnic State University, San Luis Obispo  
**Quarter:** Fall 2025  
**Assignment:** Final Project Milestone - Data Collection & EDA

---

## License

This project is for educational purposes as part of CSC-466 coursework.
