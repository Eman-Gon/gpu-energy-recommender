import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


print("\n1. Loading datasets")
ercot_df = pd.read_csv('ercot_prices.csv')
gpu_df = pd.read_csv('gpu_utilization.csv')
merged_df = pd.read_csv('merged_data.csv')

for df in [ercot_df, gpu_df, merged_df]:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"- ERCOT Data: {ercot_df.shape}")
print(f"- GPU Data: {gpu_df.shape}")
print(f"- Merged Data: {merged_df.shape}")


print("\n2. Data Quality Assessment")
print(f"- Missing values: {merged_df.isnull().sum().sum()}")
print(f"- Duplicate rows: {merged_df.duplicated().sum()}")
print(f"- Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")

print("\n3. Statistical Summary:")
print(merged_df[['price_mwh', 'gpu_utilization_pct', 'hourly_cost_usd']].describe())

print("\n4. Creating features")
merged_df['price_category'] = pd.cut(
    merged_df['price_mwh'],
    bins=[0, 40, 70, 1000],
    labels=['Low', 'Medium', 'High']
)

merged_df['is_business_hours'] = (
    (merged_df['hour'] >= 8) &
    (merged_df['hour'] <= 18) &
    (merged_df['day_of_week'] < 5)
).astype(int)

merged_df['is_peak_hours'] = (
    (merged_df['hour'] >= 14) &
    (merged_df['hour'] <= 18)
).astype(int)

merged_df['utilization_level'] = pd.cut(
    merged_df['gpu_utilization_pct'],
    bins=[0, 30, 60, 100],
    labels=['Low', 'Medium', 'High']
)

merged_df['is_efficient_time'] = (
    merged_df['jobs_per_dollar'] >
    merged_df['jobs_per_dollar'].median()
).astype(int)

print(f"-Created 5 new features")
print(f"-Target variable balance: {merged_df['is_efficient_time'].value_counts().to_dict()}")

print("\n5. Creating visualizations")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(merged_df['timestamp'], merged_df['price_mwh'],
        alpha=0.5, linewidth=0.8, label='Hourly Price')

merged_df['price_ma24'] = merged_df['price_mwh'].rolling(window=24, center=True).mean()
ax.plot(merged_df['timestamp'], merged_df['price_ma24'],
        color='red', linewidth=2, label='24h Moving Average')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price ($/MWh)', fontsize=12)
ax.set_title('ERCOT Electricity Prices Over Time', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('electricity_prices_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

hourly_price = merged_df.groupby('hour')['price_mwh'].agg(['mean', 'std'])
axes[0, 0].bar(hourly_price.index, hourly_price['mean'], color='steelblue', alpha=0.7)
axes[0, 0].errorbar(
    hourly_price.index,
    hourly_price['mean'],
    yerr=hourly_price['std'],
    fmt='none',
    ecolor='black',
    capsize=3,
    alpha=0.5
)
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Average Price ($/MWh)')
axes[0, 0].set_title('Electricity Price by Hour of Day')
axes[0, 0].grid(True, alpha=0.3)

hourly_util = merged_df.groupby('hour')['gpu_utilization_pct'].agg(['mean', 'std'])
axes[0, 1].bar(hourly_util.index, hourly_util['mean'], color='coral', alpha=0.7)
axes[0, 1].errorbar(
    hourly_util.index,
    hourly_util['mean'],
    yerr=hourly_util['std'],
    fmt='none',
    ecolor='black',
    capsize=3,
    alpha=0.5
)
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Average GPU Utilization (%)')
axes[0, 1].set_title('GPU Utilization by Hour of Day')
axes[0, 1].grid(True, alpha=0.3)

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
weekly_price = merged_df.groupby('day_of_week')['price_mwh'].mean()
axes[1, 0].bar(range(7), weekly_price.values, color='mediumseagreen', alpha=0.7)
axes[1, 0].set_xticks(range(7))
axes[1, 0].set_xticklabels(day_names)
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('Average Price ($/MWh)')
axes[1, 0].set_title('Electricity Price by Day of Week')
axes[1, 0].grid(True, alpha=0.3)

weekly_cost = merged_df.groupby('day_of_week')['hourly_cost_usd'].mean()
axes[1, 1].bar(range(7), weekly_cost.values, color='orchid', alpha=0.7)
axes[1, 1].set_xticks(range(7))
axes[1, 1].set_xticklabels(day_names)
axes[1, 1].set_xlabel('Day of Week')
axes[1, 1].set_ylabel('Average Hourly Cost ($)')
axes[1, 1].set_title('GPU Energy Cost by Day of Week')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('daily_weekly_patterns.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(12, 8))
corr_cols = [
    'price_mwh', 'gpu_utilization_pct', 'active_jobs', 'active_gpus',
    'power_consumption_kw', 'hourly_cost_usd', 'jobs_per_dollar',
    'hour', 'day_of_week', 'is_weekend'
]
corr_matrix = merged_df[corr_cols].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter = axes[0].scatter(
    merged_df['price_mwh'],
    merged_df['gpu_utilization_pct'],
    c=merged_df['hourly_cost_usd'],
    cmap='viridis',
    alpha=0.6,
    s=20
)
axes[0].set_xlabel('Electricity Price ($/MWh)', fontsize=12)
axes[0].set_ylabel('GPU Utilization (%)', fontsize=12)
axes[0].set_title('Price vs Utilization (colored by hourly cost)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label='Hourly Cost ($)')

scatter2 = axes[1].scatter(
    merged_df['price_mwh'],
    merged_df['jobs_per_dollar'],
    c=merged_df['hour'],
    cmap='twilight',
    alpha=0.6,
    s=20
)
axes[1].set_xlabel('Electricity Price ($/MWh)', fontsize=12)
axes[1].set_ylabel('Jobs per Dollar', fontsize=12)
axes[1].set_title('Jobs per Dollar vs Price', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Hour of Day')

plt.tight_layout()
plt.savefig('cost_efficiency_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].hist(merged_df['price_mwh'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(
    merged_df['price_mwh'].mean(),
    color='red',
    linestyle='--',
    linewidth=2,
    label=f"Mean: ${merged_df['price_mwh'].mean():.2f}"
)
axes[0, 0].set_xlabel('Price ($/MWh)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Electricity Price Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(merged_df['gpu_utilization_pct'], bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(
    merged_df['gpu_utilization_pct'].mean(),
    color='red',
    linestyle='--',
    linewidth=2,
    label=f"Mean: {merged_df['gpu_utilization_pct'].mean():.1f}%"
)
axes[0, 1].set_xlabel('GPU Utilization (%)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('GPU Utilization Distribution')
axes[0, 1].legend()
axes[0, 1]()