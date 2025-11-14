cd ~/csc466/gpu-energy-recommender

cat > README.md << 'EOF'
# GPU Energy-Aware Workload Recommendation System

**A data-driven machine learning system for optimizing GPU workload scheduling based on real-time electricity prices.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-EDA_Complete-success.svg)]()

---

## 🎯 Project Overview

This project develops a machine learning recommendation system that suggests optimal GPU workload scheduling by analyzing ERCOT Texas electricity prices and GPU cluster utilization patterns. 

**Goal:** Minimize energy costs by **69%** while maintaining system efficiency.

### Key Results

- 💰 **69% cost reduction** during efficient hours ($1.51 → $0.46/hour)
- ⚡ **52% cheaper electricity** during off-peak times
- 📊 **3.28x better** jobs-per-dollar efficiency
- ✅ **Perfect data quality** (0 missing values, 50/50 class balance)

---

## 📊 Key Findings

<p align="center">
  <img src="eda/target_variable_analysis.png" width="850" alt="Target Variable Analysis">
</p>

### Optimal Scheduling Windows

✅ **Schedule GPU jobs:**
- Midnight - 8:00 AM (60-85% efficient)
- 10:00 PM - 11:00 PM (70-75% efficient)

❌ **Avoid scheduling:**
- 9:00 AM - 9:00 PM (26-38% efficient)
- 4:00 PM - 5:00 PM (peak hours, most expensive)

---

## 📈 Dataset

| Metric | Value |
|--------|-------|
| **Time Period** | 90 days (Aug 14 - Nov 12, 2025) |
| **Records** | 2,161 hourly observations |
| **Features** | 20 (14 raw + 6 engineered) |
| **Missing Values** | 0 ✅ |
| **Target Balance** | 50.0% / 50.0% ✅ |

**Data Sources:**
1. ERCOT electricity prices (Texas energy market)
2. GPU cluster utilization (100-GPU cluster, realistic patterns)

---

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.10
pandas, numpy, matplotlib, seaborn, scikit-learn
```

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gpu-energy-recommender.git
cd gpu-energy-recommender

# Install dependencies
pip install -r requirements.txt
```

### Run EDA
```bash
cd eda

# Generate datasets
python data_collection.py

# Run exploratory data analysis
python eda.py

# Analyze target variable
python analyze_target_variable.py
```

### Output

After running, you'll have:
- 📊 6 high-resolution visualizations (PNG, 300 DPI)
- 📁 4 datasets (CSV format)
- 📈 Enhanced dataset with 20 features

---

## 📁 Project Structure
```
gpu-energy-recommender/
├── README.md                           # This file
├── REPORT.md                           # Full EDA report
├── requirements.txt                    # Dependencies
└── eda/
    ├── data_collection.py              # Data generation
    ├── eda.py                          # Main EDA script
    ├── analyze_target_variable.py      # Target analysis
    ├── eda_summary.md                  # Written summary
    ├── *.csv                           # 4 datasets
    └── *.png                           # 6 visualizations
```

---

## 📊 Visualizations

### 1. Target Variable Analysis
Shows the 69% cost savings opportunity with clear efficient vs inefficient patterns.

### 2. Daily & Weekly Patterns
Reveals hourly and weekly efficiency patterns - best times are midnight-8 AM.

### 3. Electricity Price Time Series
90-day price evolution with moving average, showing volatility and patterns.

### 4. Correlation Heatmap
Feature relationships - strong predictors include price, utilization, and cost.

### 5. Cost Efficiency Analysis
Price vs utilization relationships, colored by cost and time-of-day.

### 6. Distributions
Statistical distributions of all key variables.

[See full visualizations in REPORT.md](REPORT.md)

---

## 🎓 Academic Context

**Course:** CSC-466 Machine Learning  
**Institution:** California Polytechnic State University, San Luis Obispo  
**Author:** Steven  
**Date:** November 2025  
**Type:** Final Project - Classification & Recommendation System

---

## 🔬 Methodology

### Classification Problem

**Input Features:**
- Electricity price ($/MWh)
- GPU utilization (%)
- Hour of day, day of week
- Power consumption (kW)
- Business/peak hour flags

**Target Variable:**
- `is_efficient_time` (binary: 0 = inefficient, 1 = efficient)

**Approach:**
1. Train classification models (Random Forest, XGBoost, Logistic Regression)
2. Predict optimal scheduling windows
3. Build recommendation system

### Expected Performance

| Metric | Target |
|--------|--------|
| Precision | >80% |
| Recall | >75% |
| F1-Score | >77% |

---

## 🔮 Roadmap

- ✅ **Phase 1: EDA** - Complete (Nov 2025)
  - Data collection and cleaning
  - 6 visualizations created
  - Feature engineering (20 features)
  - Written report completed

- 🔄 **Phase 2: Model Building** - In Progress
  - Train classification models
  - Evaluate performance
  - Feature importance analysis

- 📅 **Phase 3: Recommendation System** - Planned
  - Build recommendation engine
  - Real-time decision support
  - Cost-benefit analysis

- 📅 **Phase 4: Final Deliverables** - Planned
  - 3-page technical paper
  - Slide deck presentation
  - Class presentation

---

## 💡 Key Insights

### Cost Comparison

| Time Period | Electricity Price | Hourly Cost | Efficiency |
|-------------|------------------|-------------|------------|
| **Efficient** | $37.54/MWh | $0.46/hr | 271 jobs/$ |
| **Inefficient** | $78.66/MWh | $1.51/hr | 83 jobs/$ |
| **Difference** | 52% cheaper | 69% cheaper | 3.28x better |

### Business Impact

For a 100-GPU cluster:
- **Annual savings:** ~$9,000
- **Monthly savings:** ~$750
- **Mechanism:** Schedule deferrable workloads during off-peak hours

For larger deployments (500-1000 GPUs):
- **Annual savings:** $45,000-$90,000

---

## 📚 Documentation

- **[REPORT.md](REPORT.md)** - Full EDA report with visualizations
- **[eda/eda_summary.md](eda/eda_summary.md)** - Detailed analysis summary
- **Code comments** - All scripts are well-documented

---

## 🤝 Contributing

This is an academic project for CSC-466. Ideas for extensions:
- Real ERCOT API integration
- Multi-zone price forecasting
- Job priority and deadline constraints
- LSTM/GRU models for price prediction
- Cloud provider integration (AWS, GCP, Azure)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 📧 Contact

**Steven**  
Cal Poly San Luis Obispo  
Computer Science Department

Questions? Open an issue on GitHub!

---

## 🙏 Acknowledgments

- ERCOT for public electricity market data
- NVIDIA for GPU specifications
- Cal Poly CSC-466 course staff
- Scikit-learn, Pandas, Matplotlib communities

---

## 📖 References

1. ERCOT - [Electric Reliability Council of Texas](https://www.ercot.com/)
2. NREL - [Data Center Energy Report](https://www.nrel.gov/docs/fy21osti/77364.pdf)
3. NVIDIA - [Data Center Documentation](https://www.nvidia.com/en-us/data-center/)

---

<p align="center">
  <i>Building sustainable AI infrastructure through intelligent scheduling</i> 🌱⚡
</p>

<p align="center">
  <b>⭐ Star this repo if you find it useful!</b>
</p>
EOF

echo "✅ README.md created!"