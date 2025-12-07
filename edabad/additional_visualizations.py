"""
Additional Simple Visualizations for CSC-466 Project
Creates clean, professional plots matching class presentation style
Author: Steven
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


print("CREATING ADDITIONAL VISUALIZATIONS")


df = pd.read_csv('merged_data_enhanced.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\nDataset loaded: {df.shape}")

print("\n1. Creating scatter plot: Price vs Utilization")

fig, ax = plt.subplots(figsize=(10, 6))

efficient = df[df['is_efficient_time'] == 1]
inefficient = df[df['is_efficient_time'] == 0]

ax.scatter(inefficient['price_mwh'], inefficient['gpu_utilization_pct'], 
           c='red', alpha=0.5, s=30, label='Inefficient (0)', edgecolors='darkred')
ax.scatter(efficient['price_mwh'], efficient['gpu_utilization_pct'], 
           c='green', alpha=0.5, s=30, label='Efficient (1)', edgecolors='darkgreen')

ax.set_xlabel('Electricity Price ($/MWh)', fontsize=12, fontweight='bold')
ax.set_ylabel('GPU Utilization (%)', fontsize=12, fontweight='bold')
ax.set_title('Efficiency Pattern: Price vs GPU Utilization', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_scatter_efficiency.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: simple_scatter_efficiency.png")
plt.close()


print("\n2. Creating box plots comparison")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

data_price = [inefficient['price_mwh'], efficient['price_mwh']]
bp1 = axes[0].boxplot(data_price, labels=['Inefficient (0)', 'Efficient (1)'],
                       patch_artist=True, widths=0.6)
bp1['boxes'][0].set_facecolor('lightcoral')
bp1['boxes'][1].set_facecolor('lightgreen')
axes[0].set_ylabel('Price ($/MWh)', fontsize=11, fontweight='bold')
axes[0].set_title('Electricity Price Distribution', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

data_cost = [inefficient['hourly_cost_usd'], efficient['hourly_cost_usd']]
bp2 = axes[1].boxplot(data_cost, labels=['Inefficient (0)', 'Efficient (1)'],
                       patch_artist=True, widths=0.6)
bp2['boxes'][0].set_facecolor('lightcoral')
bp2['boxes'][1].set_facecolor('lightgreen')
axes[1].set_ylabel('Hourly Cost ($)', fontsize=11, fontweight='bold')
axes[1].set_title('Operating Cost Distribution', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

data_jpd = [inefficient['jobs_per_dollar'], efficient['jobs_per_dollar']]
bp3 = axes[2].boxplot(data_jpd, labels=['Inefficient (0)', 'Efficient (1)'],
                       patch_artist=True, widths=0.6)
bp3['boxes'][0].set_facecolor('lightcoral')
bp3['boxes'][1].set_facecolor('lightgreen')
axes[2].set_ylabel('Jobs per Dollar', fontsize=11, fontweight='bold')
axes[2].set_title('Efficiency Metric Distribution', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('boxplots_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: boxplots_comparison.png")
plt.close()

print("\n3. Creating K-Means elbow plot")

features_for_clustering = df[['price_mwh', 'gpu_utilization_pct', 'hour', 'power_consumption_kw']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_for_clustering)
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12, fontweight='bold')
ax.set_title('K-Means Elbow Method for Optimal Clusters', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

ax.annotate('Elbow Point', xy=(3, inertias[2]), xytext=(5, inertias[2] + 1000),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('kmeans_elbow_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: kmeans_elbow_plot.png")
plt.close()


print("\n4. Creating PCA scree plot")

features_for_pca = df[['price_mwh', 'gpu_utilization_pct', 'active_jobs', 
                        'power_consumption_kw', 'hourly_cost_usd', 'hour', 
                        'day_of_week', 'is_weekend', 'is_business_hours', 
                        'is_peak_hours']].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_for_pca)

pca = PCA()
pca.fit(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(pca.explained_variance_)+1), 
             pca.explained_variance_, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Principal Component', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Explained Variance', fontsize=11, fontweight='bold')
axes[0].set_title('Scree Plot - Explained Variance', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

cumvar = pca.explained_variance_ratio_.cumsum() * 100
axes[1].plot(range(1, len(cumvar)+1), cumvar, 'ro-', linewidth=2, markersize=8)
axes[1].axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% threshold')
axes[1].axhline(y=90, color='blue', linestyle='--', linewidth=2, label='90% threshold')
axes[1].set_xlabel('Principal Component', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Cumulative Variance Explained (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: pca_analysis.png")
plt.close()


print("\n5. Creating hourly average comparison")

hourly_avg = df.groupby('hour').agg({
    'price_mwh': 'mean',
    'hourly_cost_usd': 'mean',
    'jobs_per_dollar': 'mean'
}).reset_index()

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].bar(hourly_avg['hour'], hourly_avg['price_mwh'], 
            color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Average Price ($/MWh)', fontsize=11, fontweight='bold')
axes[0].set_title('Average Electricity Price by Hour', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(hourly_avg['hour'], hourly_avg['hourly_cost_usd'], 
            color='coral', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Average Cost ($)', fontsize=11, fontweight='bold')
axes[1].set_title('Average Operating Cost by Hour', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

colors = ['green' if x > 124 else 'red' for x in hourly_avg['jobs_per_dollar']]
axes[2].bar(hourly_avg['hour'], hourly_avg['jobs_per_dollar'], 
            color=colors, alpha=0.7, edgecolor='black')
axes[2].axhline(y=124, color='blue', linestyle='--', linewidth=2, label='Median (124)')
axes[2].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Jobs per Dollar', fontsize=11, fontweight='bold')
axes[2].set_title('Average Efficiency by Hour', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hourly_averages.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: hourly_averages.png")
plt.close()

print("\n6. Creating weekly pattern plot")

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_avg = df.groupby('day_of_week').agg({
    'price_mwh': 'mean',
    'hourly_cost_usd': 'mean',
    'gpu_utilization_pct': 'mean'
}).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(weekly_avg['day_of_week'], weekly_avg['price_mwh'], 
        'o-', linewidth=2, markersize=8, label='Avg Price ($/MWh)', color='steelblue')
ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Price ($/MWh)', fontsize=12, fontweight='bold', color='steelblue')
ax.tick_params(axis='y', labelcolor='steelblue')
ax.set_xticks(range(7))
ax.set_xticklabels(day_names, rotation=45, ha='right')
ax.set_title('Weekly Pattern: Price and Utilization', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(weekly_avg['day_of_week'], weekly_avg['gpu_utilization_pct'], 
         's-', linewidth=2, markersize=8, label='Avg Utilization (%)', color='coral')
ax2.set_ylabel('Average GPU Utilization (%)', fontsize=12, fontweight='bold', color='coral')
ax2.tick_params(axis='y', labelcolor='coral')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('weekly_patterns.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: weekly_patterns.png")
plt.close()

print("VISUALIZATION CREATION COMPLETE")

print("\nGenerated 6 additional clean visualizations:")
print("  1. simple_scatter_efficiency.png - Price vs Utilization (colored by efficiency)")
print("  2. boxplots_comparison.png - Box plots comparing efficient vs inefficient")
print("  3. kmeans_elbow_plot.png - K-Means elbow method for clustering")
print("  4. pca_analysis.png - PCA scree plot and cumulative variance")
print("  5. hourly_averages.png - Bar charts of hourly averages")
print("  6. weekly_patterns.png - Line plot of weekly patterns")
print("\nAll visualizations saved in current directory!")
print("These match the clean, professional style from class examples.")