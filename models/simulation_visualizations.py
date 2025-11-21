"""
Simulation Visualizations - Show ML vs Rule-Based Scheduling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*70)
print("SIMULATION VISUALIZATION GENERATOR")
print("="*70)

# Load model and data
model = joblib.load('../results/logistic_regression_model.pkl')
scaler = joblib.load('../results/scaler.pkl')
df = pd.read_csv('../eda/merged_data_enhanced.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

with open('../results/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# Use actual test data for realistic simulation
split_idx = int(len(df) * 0.75)
test_data = df.iloc[split_idx:split_idx+168].copy()  # One week

print(f"\nSimulating {len(test_data)} hours (1 week)")

# Simulate decisions
ml_decisions = []
rule_decisions = []
actual_efficiency = []

for idx, row in test_data.iterrows():
    # ML decision
    features = [row[f] for f in feature_names]
    features_scaled = scaler.transform([features])
    ml_pred = model.predict(features_scaled)[0]
    ml_prob = model.predict_proba(features_scaled)[0][1]
    
    # Rule decision (night OR cheap)
    rule_pred = 1 if (row['hour'] < 8 or row['price_mwh'] < 45) else 0
    
    # Actual efficiency
    actual = row['is_efficient_time']
    
    ml_decisions.append(ml_pred)
    rule_decisions.append(rule_pred)
    actual_efficiency.append(actual)

test_data['ml_decision'] = ml_decisions
test_data['rule_decision'] = rule_decisions
test_data['actual_efficient'] = actual_efficiency

# Calculate outcomes
test_data['ml_correct'] = (test_data['ml_decision'] == test_data['actual_efficient'])
test_data['rule_correct'] = (test_data['rule_decision'] == test_data['actual_efficient'])

# ============================================================================
# VISUALIZATION 1: Decision Timeline
# ============================================================================
print("\n[1/4] Creating decision timeline...")

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Actual efficiency
colors = ['red' if x == 0 else 'green' for x in test_data['actual_efficient']]
axes[0].bar(range(len(test_data)), test_data['actual_efficient'], 
            color=colors, alpha=0.6, edgecolor='black')
axes[0].set_ylabel('Actual\nEfficiency', fontsize=11, fontweight='bold')
axes[0].set_ylim([-0.1, 1.1])
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(['Inefficient', 'Efficient'])
axes[0].set_title('One Week Simulation: ML vs Rule-Based Scheduling', 
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# ML decisions
colors_ml = ['green' if ml == actual else 'red' 
             for ml, actual in zip(test_data['ml_decision'], test_data['actual_efficient'])]
axes[1].bar(range(len(test_data)), test_data['ml_decision'], 
            color=colors_ml, alpha=0.6, edgecolor='black')
axes[1].set_ylabel('ML Model\nDecision', fontsize=11, fontweight='bold')
axes[1].set_ylim([-0.1, 1.1])
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['Wait', 'Schedule'])
axes[1].grid(True, alpha=0.3, axis='x')

# Rule decisions
colors_rule = ['green' if rule == actual else 'red' 
               for rule, actual in zip(test_data['rule_decision'], test_data['actual_efficient'])]
axes[2].bar(range(len(test_data)), test_data['rule_decision'], 
            color=colors_rule, alpha=0.6, edgecolor='black')
axes[2].set_ylabel('Rule-Based\nDecision', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Hour', fontsize=12, fontweight='bold')
axes[2].set_ylim([-0.1, 1.1])
axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(['Wait', 'Schedule'])
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../results/plots/simulation_timeline.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: simulation_timeline.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Accuracy Over Time
# ============================================================================
print("\n[2/4] Creating accuracy comparison...")

# Calculate rolling accuracy
window = 24
test_data['ml_rolling_acc'] = test_data['ml_correct'].rolling(window, min_periods=1).mean() * 100
test_data['rule_rolling_acc'] = test_data['rule_correct'].rolling(window, min_periods=1).mean() * 100

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(range(len(test_data)), test_data['ml_rolling_acc'], 
        'g-', linewidth=2.5, label='ML Model', marker='o', markersize=4)
ax.plot(range(len(test_data)), test_data['rule_rolling_acc'], 
        'r-', linewidth=2.5, label='Rule-Based', marker='s', markersize=4)

ax.set_xlabel('Hour', fontsize=12, fontweight='bold')
ax.set_ylabel('Rolling Accuracy (%) - 24hr Window', fontsize=12, fontweight='bold')
ax.set_title('Accuracy Over Time: ML vs Rule-Based Scheduling', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([50, 105])

# Add overall accuracy annotations
ml_overall = test_data['ml_correct'].mean() * 100
rule_overall = test_data['rule_correct'].mean() * 100

ax.axhline(ml_overall, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(rule_overall, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

ax.text(len(test_data)*0.02, ml_overall+2, f'ML Overall: {ml_overall:.1f}%', 
        fontsize=11, fontweight='bold', color='green')
ax.text(len(test_data)*0.02, rule_overall-3, f'Rule Overall: {rule_overall:.1f}%', 
        fontsize=11, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('../results/plots/simulation_accuracy_over_time.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: simulation_accuracy_over_time.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Cost Savings Analysis
# ============================================================================
print("\n[3/4] Creating cost savings analysis...")

# Calculate costs
test_data['ml_cost'] = np.where(test_data['ml_decision'] == 1, 
                                 test_data['hourly_cost_usd'], 0)
test_data['rule_cost'] = np.where(test_data['rule_decision'] == 1, 
                                   test_data['hourly_cost_usd'], 0)

test_data['ml_cumulative_cost'] = test_data['ml_cost'].cumsum()
test_data['rule_cumulative_cost'] = test_data['rule_cost'].cumsum()

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(range(len(test_data)), test_data['rule_cumulative_cost'], 
        'r-', linewidth=2.5, label='Rule-Based Cost', marker='s', markersize=4)
ax.plot(range(len(test_data)), test_data['ml_cumulative_cost'], 
        'g-', linewidth=2.5, label='ML-Based Cost', marker='o', markersize=4)

ax.fill_between(range(len(test_data)), 
                test_data['ml_cumulative_cost'], 
                test_data['rule_cumulative_cost'],
                where=(test_data['rule_cumulative_cost'] >= test_data['ml_cumulative_cost']),
                color='green', alpha=0.2, label='Cost Savings')

ax.set_xlabel('Hour', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Cost ($)', fontsize=12, fontweight='bold')
ax.set_title('Cost Comparison: ML vs Rule-Based Scheduling (1 Week)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

# Final savings annotation
final_savings = test_data['rule_cumulative_cost'].iloc[-1] - test_data['ml_cumulative_cost'].iloc[-1]
savings_pct = (final_savings / test_data['rule_cumulative_cost'].iloc[-1]) * 100

ax.text(len(test_data)*0.5, test_data['rule_cumulative_cost'].max()*0.9,
        f'Total Savings: ${final_savings:.2f} ({savings_pct:.1f}%)',
        fontsize=13, fontweight='bold', color='green',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))

plt.tight_layout()
plt.savefig('../results/plots/simulation_cost_savings.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: simulation_cost_savings.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: Error Type Analysis
# ============================================================================
print("\n[4/4] Creating error analysis...")

# Categorize decisions
def categorize_decision(row):
    if row['ml_decision'] == row['actual_efficient']:
        if row['ml_decision'] == 1:
            return 'True Positive'
        else:
            return 'True Negative'
    else:
        if row['ml_decision'] == 1:
            return 'False Positive'
        else:
            return 'False Negative'

def categorize_rule(row):
    if row['rule_decision'] == row['actual_efficient']:
        if row['rule_decision'] == 1:
            return 'True Positive'
        else:
            return 'True Negative'
    else:
        if row['rule_decision'] == 1:
            return 'False Positive'
        else:
            return 'False Negative'

test_data['ml_category'] = test_data.apply(categorize_decision, axis=1)
test_data['rule_category'] = test_data.apply(categorize_rule, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ML error breakdown
ml_counts = test_data['ml_category'].value_counts()
colors_error = ['green', 'lightgreen', 'salmon', 'red']
axes[0].bar(range(len(ml_counts)), ml_counts.values, 
            color=colors_error[:len(ml_counts)], alpha=0.8, edgecolor='black', linewidth=2)
axes[0].set_xticks(range(len(ml_counts)))
axes[0].set_xticklabels(ml_counts.index, rotation=45, ha='right')
axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[0].set_title(f'ML Model Decisions (Accuracy: {ml_overall:.1f}%)', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(ml_counts.values):
    axes[0].text(i, v + 2, str(v), ha='center', fontsize=11, fontweight='bold')

# Rule error breakdown
rule_counts = test_data['rule_category'].value_counts()
axes[1].bar(range(len(rule_counts)), rule_counts.values, 
            color=colors_error[:len(rule_counts)], alpha=0.8, edgecolor='black', linewidth=2)
axes[1].set_xticks(range(len(rule_counts)))
axes[1].set_xticklabels(rule_counts.index, rotation=45, ha='right')
axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[1].set_title(f'Rule-Based Decisions (Accuracy: {rule_overall:.1f}%)', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(rule_counts.values):
    axes[1].text(i, v + 2, str(v), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/plots/simulation_error_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: simulation_error_analysis.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("SIMULATION SUMMARY")
print("="*70)

print(f"\nML Model:")
print(f"  Accuracy: {ml_overall:.1f}%")
print(f"  Total Cost: ${test_data['ml_cumulative_cost'].iloc[-1]:.2f}")
print(f"  Jobs Scheduled: {test_data['ml_decision'].sum()}")

print(f"\nRule-Based:")
print(f"  Accuracy: {rule_overall:.1f}%")
print(f"  Total Cost: ${test_data['rule_cumulative_cost'].iloc[-1]:.2f}")
print(f"  Jobs Scheduled: {test_data['rule_decision'].sum()}")

print(f"\nSavings:")
print(f"  Cost Saved: ${final_savings:.2f} ({savings_pct:.1f}%)")
print(f"  Weekly savings extrapolated to year: ${final_savings * 52:.2f}")

print("\n✅ ALL VISUALIZATIONS CREATED!")
print("\nGenerated files:")
print("  1. simulation_timeline.png")
print("  2. simulation_accuracy_over_time.png")
print("  3. simulation_cost_savings.png")
print("  4. simulation_error_analysis.png")