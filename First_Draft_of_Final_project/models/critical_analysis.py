"""
Critical Analysis: Validates this is not a trivial "run at night" problem
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../../eda/merged_data_enhanced.csv')

print("CRITICAL ANALYSIS: IS THIS JUST 'RUN AT NIGHT'?\n")

# Daytime efficiency analysis
daytime = df[df['hour'].between(9, 17)]
daytime_efficient = (daytime['is_efficient_time'] == 1).sum()
daytime_total = len(daytime)
daytime_pct = (daytime_efficient / daytime_total * 100) if daytime_total > 0 else 0

print("[1] DAYTIME EFFICIENCY ANALYSIS")
print(f"Daytime hours (9am-5pm): {daytime_total} total")
print(f"  Efficient: {daytime_efficient} ({daytime_pct:.1f}%)")
print(f"  Inefficient: {daytime_total - daytime_efficient} ({100-daytime_pct:.1f}%)\n")

# Nighttime efficiency analysis
nighttime = df[df['hour'].isin(range(0, 8))]
nighttime_efficient = (nighttime['is_efficient_time'] == 1).sum()
nighttime_total = len(nighttime)
nighttime_pct = (nighttime_efficient / nighttime_total * 100) if nighttime_total > 0 else 0

print("[2] NIGHTTIME EFFICIENCY ANALYSIS")
print(f"Nighttime hours (midnight-8am): {nighttime_total} total")
print(f"  Efficient: {nighttime_efficient} ({nighttime_pct:.1f}%)")
print(f"  Inefficient: {nighttime_total - nighttime_efficient} ({100-nighttime_pct:.1f}%)\n")

# Rule-based vs ML comparison
df['rule_night'] = df['hour'].isin(range(0, 8)).astype(int)
night_rule_accuracy = (df['rule_night'] == df['is_efficient_time']).mean()

median_price = df['price_mwh'].median()
df['rule_price'] = (df['price_mwh'] < median_price).astype(int)
price_rule_accuracy = (df['rule_price'] == df['is_efficient_time']).mean()

df['rule_util'] = (df['gpu_utilization_pct'] < 60).astype(int)
util_rule_accuracy = (df['rule_util'] == df['is_efficient_time']).mean()

ml_accuracy = 0.974

print("[3] RULE-BASED VS ML COMPARISON")
print(f"'Run at night (0-8am)': {night_rule_accuracy*100:.1f}%")
print(f"'Run when price < ${median_price:.0f}/MWh': {price_rule_accuracy*100:.1f}%")
print(f"'Run when util < 60%': {util_rule_accuracy*100:.1f}%")
print(f"ML Model: {ml_accuracy*100:.1f}%")
print(f"Improvement: {(ml_accuracy - max(night_rule_accuracy, price_rule_accuracy, util_rule_accuracy))*100:.1f} percentage points\n")

# Error analysis
night_rule_errors = df[df['rule_night'] != df['is_efficient_time']]
false_positives = night_rule_errors[night_rule_errors['rule_night'] == 1]
false_negatives = night_rule_errors[night_rule_errors['rule_night'] == 0]

print("[4] ERROR ANALYSIS")
print(f"'Run at night' rule makes {len(night_rule_errors)} errors:")
print(f"  False Positives (inefficient nights): {len(false_positives)}")
if len(false_positives) > 0:
    print(f"    Avg price: ${false_positives['price_mwh'].mean():.2f}/MWh")
    print(f"    Avg util: {false_positives['gpu_utilization_pct'].mean():.1f}%")
print(f"  False Negatives (efficient days): {len(false_negatives)}")
if len(false_negatives) > 0:
    print(f"    Avg price: ${false_negatives['price_mwh'].mean():.2f}/MWh")
    print(f"    Avg util: {false_negatives['gpu_utilization_pct'].mean():.1f}%\n")

# Complexity score
score = 0
improvement = (ml_accuracy - night_rule_accuracy) * 100

print("[5] COMPLEXITY VALIDATION")
if daytime_pct > 20:
    score += 3
    print(f" [+3] Significant daytime efficiency ({daytime_pct:.1f}% > 20%)")
if nighttime_pct < 85:
    score += 3
    print(f" [+3] Nighttime variability ({nighttime_pct:.1f}% < 85%)")
if improvement > 15:
    score += 3
    print(f" [+3] Large ML improvement ({improvement:.1f} > 15 points)")

print(f"\nCOMPLEXITY SCORE: {score}/9")

if score >= 7:
    print("\nVERDICT: REAL ML value - NOT trivial!")
    print("Discovering nuanced patterns beyond 'run at night'\n")
elif score >= 4:
    print("\nVERDICT: BORDERLINE - Needs stronger justification\n")
else:
    print("\nVERDICT: Trivial - Just 'run at night'\n")