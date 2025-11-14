# Paste the entire script here
"""
Comprehensive Analysis of is_efficient_time Target Variable
Shows what makes an hour "efficient" vs "inefficient"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("ANALYZING TARGET VARIABLE: is_efficient_time")
print("=" * 80)

# Load the enhanced dataset
df = pd.read_csv('merged_data_enhanced.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\nDataset loaded: {df.shape}")
print(f"Total records: {len(df)}")

# ==========================================
# 1. UNDERSTAND THE TARGET VARIABLE
# ==========================================

print("\n" + "=" * 80)
print("1. TARGET VARIABLE DISTRIBUTION")
print("=" * 80)

efficient_count = (df['is_efficient_time'] == 1).sum()
inefficient_count = (df['is_efficient_time'] == 0).sum()

print(f"\nEfficient hours (label = 1): {efficient_count} ({efficient_count/len(df)*100:.1f}%)")
print(f"Inefficient hours (label = 0): {inefficient_count} ({inefficient_count/len(df)*100:.1f}%)")
print(f"Balance ratio: {min(efficient_count, inefficient_count) / max(efficient_count, inefficient_count):.3f}")

# Show the threshold
median_jobs_per_dollar = df['jobs_per_dollar'].median()
print(f"\nThreshold (median jobs_per_dollar): {median_jobs_per_dollar:.2f}")
print(f"  - If jobs_per_dollar > {median_jobs_per_dollar:.2f} → Efficient (1)")
print(f"  - If jobs_per_dollar ≤ {median_jobs_per_dollar:.2f} → Inefficient (0)")

# ==========================================
# 2. COMPARE EFFICIENT VS INEFFICIENT HOURS
# ==========================================

print("\n" + "=" * 80)
print("2. EFFICIENT vs INEFFICIENT CHARACTERISTICS")
print("=" * 80)

efficient = df[df['is_efficient_time'] == 1]
inefficient = df[df['is_efficient_time'] == 0]

comparison = pd.DataFrame({
    'Metric': [
        'Electricity Price ($/MWh)',
        'GPU Utilization (%)',
        'Active Jobs',
        'Power Consumption (kW)',
        'Hourly Cost ($)',
        'Jobs per Dollar'
    ],
    'Efficient (1)': [
        f"${efficient['price_mwh'].mean():.2f}",
        f"{efficient['gpu_utilization_pct'].mean():.1f}%",
        f"{efficient['active_jobs'].mean():.0f}",
        f"{efficient['power_consumption_kw'].mean():.1f} kW",
        f"${efficient['hourly_cost_usd'].mean():.2f}",
        f"{efficient['jobs_per_dollar'].mean():.2f}"
    ],
    'Inefficient (0)': [
        f"${inefficient['price_mwh'].mean():.2f}",
        f"{inefficient['gpu_utilization_pct'].mean():.1f}%",
        f"{inefficient['active_jobs'].mean():.0f}",
        f"{inefficient['power_consumption_kw'].mean():.1f} kW",
        f"${inefficient['hourly_cost_usd'].mean():.2f}",
        f"{inefficient['jobs_per_dollar'].mean():.2f}"
    ]
})

print("\n" + comparison.to_string(index=False))

# Calculate differences
price_diff = ((efficient['price_mwh'].mean() - inefficient['price_mwh'].mean()) / 
              inefficient['price_mwh'].mean() * 100)
cost_diff = ((efficient['hourly_cost_usd'].mean() - inefficient['hourly_cost_usd'].mean()) / 
             inefficient['hourly_cost_usd'].mean() * 100)

print(f"\n📊 KEY INSIGHTS:")
print(f"  • Efficient hours have {abs(price_diff):.1f}% {'LOWER' if price_diff < 0 else 'HIGHER'} electricity prices")
print(f"  • Efficient hours have {abs(cost_diff):.1f}% {'LOWER' if cost_diff < 0 else 'HIGHER'} operating costs")
print(f"  • Efficient hours achieve {efficient['jobs_per_dollar'].mean() / inefficient['jobs_per_dollar'].mean():.2f}x more jobs per dollar")

# ==========================================
# 3. SHOW SAMPLE HOURS
# ==========================================

print("\n" + "=" * 80)
print("3. SAMPLE EFFICIENT HOURS")
print("=" * 80)

efficient_sample = efficient.nlargest(5, 'jobs_per_dollar')[
    ['timestamp', 'hour', 'price_mwh', 'active_jobs', 'hourly_cost_usd', 'jobs_per_dollar']
]
print("\nTop 5 Most Efficient Hours:")
print(efficient_sample.to_string(index=False))

print("\n" + "=" * 80)
print("4. SAMPLE INEFFICIENT HOURS")
print("=" * 80)

inefficient_sample = inefficient.nsmallest(5, 'jobs_per_dollar')[
    ['timestamp', 'hour', 'price_mwh', 'active_jobs', 'hourly_cost_usd', 'jobs_per_dollar']
]
print("\nTop 5 Least Efficient Hours:")
print(inefficient_sample.to_string(index=False))

# ==========================================
# 5. HOURLY PATTERNS
# ==========================================

print("\n" + "=" * 80)
print("5. HOURLY EFFICIENCY PATTERNS")
print("=" * 80)

hourly_efficiency = df.groupby('hour')['is_efficient_time'].agg(['mean', 'sum', 'count'])
hourly_efficiency.columns = ['Efficiency_Rate', 'Efficient_Count', 'Total_Hours']
hourly_efficiency['Efficiency_Pct'] = hourly_efficiency['Efficiency_Rate'] * 100

print("\nHours Most Likely to be Efficient:")
print(hourly_efficiency.nlargest(5, 'Efficiency_Pct')[['Efficiency_Pct', 'Efficient_Count', 'Total_Hours']])

print("\nHours Least Likely to be Efficient:")
print(hourly_efficiency.nsmallest(5, 'Efficiency_Pct')[['Efficiency_Pct', 'Efficient_Count', 'Total_Hours']])

# ==========================================
# 6. VISUALIZATIONS
# ==========================================

print("\n" + "=" * 80)
print("6. CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Jobs per Dollar Distribution by Label
axes[0, 0].hist(efficient['jobs_per_dollar'], bins=50, alpha=0.6, label='Efficient (1)', color='green', edgecolor='black')
axes[0, 0].hist(inefficient['jobs_per_dollar'], bins=50, alpha=0.6, label='Inefficient (0)', color='red', edgecolor='black')
axes[0, 0].axvline(median_jobs_per_dollar, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {median_jobs_per_dollar:.2f}')
axes[0, 0].set_xlabel('Jobs per Dollar', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Target Variable: Jobs per Dollar Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Hourly Efficiency Rate
axes[0, 1].bar(hourly_efficiency.index, hourly_efficiency['Efficiency_Pct'], 
               color=['green' if x > 50 else 'red' for x in hourly_efficiency['Efficiency_Pct']], 
               alpha=0.7, edgecolor='black')
axes[0, 1].axhline(50, color='blue', linestyle='--', linewidth=2, label='50% threshold')
axes[0, 1].set_xlabel('Hour of Day', fontsize=12)
axes[0, 1].set_ylabel('% Efficient', fontsize=12)
axes[0, 1].set_title('Efficiency Rate by Hour of Day', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Price vs Label
price_by_label = df.groupby('is_efficient_time')['price_mwh'].mean()
axes[1, 0].bar(['Inefficient (0)', 'Efficient (1)'], price_by_label.values, 
               color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 0].set_ylabel('Average Electricity Price ($/MWh)', fontsize=12)
axes[1, 0].set_title('Average Electricity Price by Label', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(price_by_label.values):
    axes[1, 0].text(i, v + 1, f'${v:.2f}', ha='center', fontsize=12, fontweight='bold')

# Plot 4: Cost vs Label
cost_by_label = df.groupby('is_efficient_time')['hourly_cost_usd'].mean()
axes[1, 1].bar(['Inefficient (0)', 'Efficient (1)'], cost_by_label.values, 
               color=['red', 'green'], alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('Average Hourly Cost ($)', fontsize=12)
axes[1, 1].set_title('Average Operating Cost by Label', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(cost_by_label.values):
    axes[1, 1].text(i, v + 0.05, f'${v:.2f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('target_variable_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: target_variable_analysis.png")
plt.close()

# ==========================================
# 7. FEATURE IMPORTANCE FOR CLASSIFICATION
# ==========================================

print("\n" + "=" * 80)
print("7. FEATURES CORRELATED WITH is_efficient_time")
print("=" * 80)

# Calculate correlation with target
correlations = df[[
    'price_mwh', 'gpu_utilization_pct', 'active_jobs', 'power_consumption_kw',
    'hourly_cost_usd', 'hour', 'day_of_week', 'is_weekend', 
    'is_business_hours', 'is_peak_hours', 'is_efficient_time'
]].corr()['is_efficient_time'].drop('is_efficient_time').sort_values(ascending=False)

print("\nCorrelation with is_efficient_time:")
for feature, corr in correlations.items():
    direction = "positive" if corr > 0 else "negative"
    strength = "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.1 else "weak"
    print(f"  {feature:25s}: {corr:+.3f}  ({strength} {direction})")

# ==========================================
# 8. CLASSIFICATION HINTS
# ==========================================

print("\n" + "=" * 80)
print("8. WHAT THIS MEANS FOR CLASSIFICATION")
print("=" * 80)

print("""
Your classification model will learn to predict is_efficient_time (0 or 1) using these patterns:

✅ GOOD PREDICTORS (strong correlation):
   • jobs_per_dollar (by definition - this is what creates the label)
   • price_mwh (lower price → more likely efficient)
   • hourly_cost_usd (lower cost → more likely efficient)
   • hour (certain hours are consistently efficient)

⚠️  MODERATE PREDICTORS:
   • is_peak_hours (peak hours are less efficient)
   • is_business_hours (business hours affect efficiency)
   • gpu_utilization_pct (utilization patterns matter)

📊 MODEL TRAINING EXAMPLE:
   X = df[['price_mwh', 'hour', 'day_of_week', 'gpu_utilization_pct', ...]]
   y = df['is_efficient_time']
   
   model.fit(X, y)
   
   # Predict: Is 2:00 AM on Tuesday efficient?
   prediction = model.predict([[35.0, 2, 1, 50.0, ...]])
   # Output: 1 (Yes, it's efficient!)

💡 RECOMMENDATION SYSTEM:
   If model predicts 1 → "✅ Good time to schedule GPU jobs"
   If model predicts 0 → "❌ Wait for better conditions"
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\n✅ Created: target_variable_analysis.png")
print("📊 Review the visualizations to understand your target variable!")
print("\n🚀 Next step: Build classification models (Random Forest, XGBoost)")