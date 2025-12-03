"""
Analysis: Is this project just "run at night" or is there real ML value?
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../../eda/merged_data_enhanced.csv')

print("CRITICAL ANALYSIS: IS THIS JUST 'RUN AT NIGHT'?")

print("\n[1] DAYTIME EFFICIENCY ANALYSIS")


daytime = df[df['hour'].between(9, 17)]  # 9am - 5pm
daytime_efficient = (daytime['is_efficient_time'] == 1).sum()
daytime_total = len(daytime)
daytime_pct = (daytime_efficient / daytime_total * 100) if daytime_total > 0 else 0

print(f"Daytime hours (9am-5pm):")
print(f"  Total: {daytime_total} hours")
print(f"  Efficient: {daytime_efficient} hours ({daytime_pct:.1f}%)")
print(f"  Inefficient: {daytime_total - daytime_efficient} hours ({100-daytime_pct:.1f}%)")

if daytime_pct > 10:
    print(f" GOOD: {daytime_pct:.1f}% of daytime is efficient NOT just 'run at night'!")
elif daytime_pct > 0:
    print(f" BORDERLINE: Only {daytime_pct:.1f}% of daytime is efficient")
else:
    print(f" PROBLEM: 0% of daytime is efficient this IS just 'run at night'")

print("\n[2] NIGHTTIME EFFICIENCY ANALYSIS")

nighttime = df[df['hour'].isin(range(0, 8))]  # Midnight - 8am
nighttime_efficient = (nighttime['is_efficient_time'] == 1).sum()
nighttime_total = len(nighttime)
nighttime_pct = (nighttime_efficient / nighttime_total * 100) if nighttime_total > 0 else 0

print(f"Nighttime hours (midnight-8am):")
print(f"  Total: {nighttime_total} hours")
print(f"  Efficient: {nighttime_efficient} hours ({nighttime_pct:.1f}%)")
print(f"  Inefficient: {nighttime_total - nighttime_efficient} hours ({100-nighttime_pct:.1f}%)")

if nighttime_pct < 90:
    print(f" GOOD: Only {nighttime_pct:.1f}% of nighttime is efficient shows nuance!")
elif nighttime_pct < 100:
    print(f" BORDERLINE: {nighttime_pct:.1f}% of nighttime is efficient")
else:
    print(f" PROBLEM: 100% of nighttime is efficient confirms 'run at night'")

print("\n[3] RULE-BASED VS ML COMPARISON")

df['rule_night'] = df['hour'].isin(range(0, 8)).astype(int)
night_rule_accuracy = (df['rule_night'] == df['is_efficient_time']).mean()

median_price = df['price_mwh'].median()
df['rule_price'] = (df['price_mwh'] < median_price).astype(int)
price_rule_accuracy = (df['rule_price'] == df['is_efficient_time']).mean()

df['rule_util'] = (df['gpu_utilization_pct'] < 60).astype(int)
util_rule_accuracy = (df['rule_util'] == df['is_efficient_time']).mean()
ml_accuracy = 0.976

print(f"Rule 1 'Always run at night (midnight-8am)': {night_rule_accuracy*100:.1f}%")
print(f"Rule 2 'Run when price < ${median_price:.0f}/MWh': {price_rule_accuracy*100:.1f}%")
print(f"Rule 3 'Run when GPU util < 60%': {util_rule_accuracy*100:.1f}%")
print(f"\nYour ML Model: {ml_accuracy*100:.1f}%")
print(f"\nImprovement over best rule: {(ml_accuracy - max(night_rule_accuracy, price_rule_accuracy, util_rule_accuracy))*100:.1f} percentage points")

if night_rule_accuracy > 0.90:
    print("\n WARNING: 'Run at night' rule is >90% accurate project might be trivial")
elif night_rule_accuracy > 0.80:
    print("\n CAUTION: 'Run at night' rule is >80% accurate need strong justification")
else:
    print("\n EXCELLENT: Simple rules fail ML is clearly needed!")

print("\n[4] ERROR ANALYSIS - WHERE DO SIMPLE RULES FAIL?")

night_rule_errors = df[df['rule_night'] != df['is_efficient_time']]
print(f"\n'Run at night' rule makes {len(night_rule_errors)} errors:")

false_positives = night_rule_errors[night_rule_errors['rule_night'] == 1]
print(f"  - {len(false_positives)} False Positives: Night hours that were INEFFICIENT")
if len(false_positives) > 0:
    print(f"    Average price: ${false_positives['price_mwh'].mean():.2f}/MWh")
    print(f"    Average utilization: {false_positives['gpu_utilization_pct'].mean():.1f}%")
    print(f"    Average jobs: {false_positives['active_jobs'].mean():.0f}")

false_negatives = night_rule_errors[night_rule_errors['rule_night'] == 0]
print(f"  - {len(false_negatives)} False Negatives: Day hours that were EFFICIENT")
if len(false_negatives) > 0:
    print(f"    Average price: ${false_negatives['price_mwh'].mean():.2f}/MWh")
    print(f"    Average utilization: {false_negatives['gpu_utilization_pct'].mean():.1f}%")
    print(f"    Average jobs: {false_negatives['active_jobs'].mean():.0f}")


print("\n[5] SPECIFIC EXAMPLES")

print("\nExample: Efficient DAYTIME hours (ML says YES, simple rule says NO):")
efficient_daytime = df[(df['is_efficient_time'] == 1) & (df['hour'].between(9, 17))]
if len(efficient_daytime) > 0:
    sample = efficient_daytime.head(3)[['timestamp', 'hour', 'price_mwh', 'gpu_utilization_pct', 'active_jobs', 'jobs_per_dollar']]
    print(sample.to_string(index=False))
else:
    print("  None found - this IS just 'run at night'")

print("\nExample: Inefficient NIGHTTIME hours (ML says NO, simple rule says YES):")
inefficient_nighttime = df[(df['is_efficient_time'] == 0) & (df['hour'].isin(range(0, 8)))]
if len(inefficient_nighttime) > 0:
    sample = inefficient_nighttime.head(3)[['timestamp', 'hour', 'price_mwh', 'gpu_utilization_pct', 'active_jobs', 'jobs_per_dollar']]
    print(sample.to_string(index=False))
else:
    print("  None found ALL nighttime is efficient")

print("FINAL thoughs")


score = 0

# Scoring system
if daytime_pct > 20:
    score += 3
    print(" [+3] Significant daytime efficiency (>20%)")
elif daytime_pct > 5:
    score += 2
    print(" [+2] Some daytime efficiency (>5%)")
else:
    score += 0
    print(" [+0] Almost no daytime efficiency")

if nighttime_pct < 85:
    score += 3
    print(" [+3] Nighttime is not always efficient (<85%)")
elif nighttime_pct < 95:
    score += 2
    print(" [+2] Most nighttime is efficient (<95%)")
else:
    score += 0
    print(" [+0] Nearly all nighttime is efficient")

improvement = (ml_accuracy - night_rule_accuracy) * 100
if improvement > 15:
    score += 3
    print(f" [+3] ML improves >15 points over 'night' rule ({improvement:.1f} points)")
elif improvement > 5:
    score += 2
    print(f" [+2] ML improves >5 points over 'night' rule ({improvement:.1f} points)")
else:
    score += 1
    print(f" [+1] ML barely improves over 'night' rule ({improvement:.1f} points)")

print(f"\nTOTAL SCORE: {score}/9")
print()

if score >= 7:
    print(" VERDICT: Your project has REAL ML value NOT trivial!")
    print(" You're discovering nuanced patterns beyond 'run at night'")
elif score >= 4:
    print(" VERDICT: Your project is BORDERLINE")
    print(" Need to emphasize what ML learns that rules miss")
else:
    print(" VERDICT: Your project IS trivial it's just 'run at night'")
    print(" Consider: changing labeling method or adding complexity")