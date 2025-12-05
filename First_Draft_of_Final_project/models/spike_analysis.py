"""
Spike Analysis - Why are some daytime hours efficient?
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../../eda/merged_data_enhanced.csv')

print("="*80)
print("ANALYZING DAYTIME EFFICIENCY SPIKES")
print("="*80)

# Get efficient daytime hours
daytime = df[df['hour'].between(9, 17)]
efficient_daytime = daytime[daytime['is_efficient_time'] == 1]
inefficient_daytime = daytime[daytime['is_efficient_time'] == 0]

print(f"\nDaytime hours (9am-5pm): {len(daytime)} total")
print(f"  Efficient: {len(efficient_daytime)} ({len(efficient_daytime)/len(daytime)*100:.1f}%)")
print(f"  Inefficient: {len(inefficient_daytime)} ({len(inefficient_daytime)/len(daytime)*100:.1f}%)")

# =============================================================================
# QUESTION 1: What makes efficient daytime different?
# =============================================================================
print("\n" + "="*80)
print("WHAT MAKES EFFICIENT DAYTIME HOURS DIFFERENT?")
print("="*80)

print("\n--- EFFICIENT Daytime Hours ---")
print(f"Average price:        ${efficient_daytime['price_mwh'].mean():.2f}/MWh")
print(f"Average utilization:  {efficient_daytime['gpu_utilization_pct'].mean():.1f}%")
print(f"Average active jobs:  {efficient_daytime['active_jobs'].mean():.0f}")
print(f"Average hourly cost:  ${efficient_daytime['hourly_cost_usd'].mean():.2f}")

print("\n--- INEFFICIENT Daytime Hours ---")
print(f"Average price:        ${inefficient_daytime['price_mwh'].mean():.2f}/MWh")
print(f"Average utilization:  {inefficient_daytime['gpu_utilization_pct'].mean():.1f}%")
print(f"Average active jobs:  {inefficient_daytime['active_jobs'].mean():.0f}")
print(f"Average hourly cost:  ${inefficient_daytime['hourly_cost_usd'].mean():.2f}")

print("\n--- DIFFERENCE ---")
print(f"Price difference:     ${inefficient_daytime['price_mwh'].mean() - efficient_daytime['price_mwh'].mean():.2f}/MWh")
print(f"Utilization diff:     {inefficient_daytime['gpu_utilization_pct'].mean() - efficient_daytime['gpu_utilization_pct'].mean():.1f}%")
print(f"Jobs diff:            {efficient_daytime['active_jobs'].mean() - inefficient_daytime['active_jobs'].mean():.0f}")

# =============================================================================
# QUESTION 2: Is there a time-of-day pattern?
# =============================================================================
print("\n" + "="*80)
print("TIME-OF-DAY PATTERN IN DAYTIME")
print("="*80)

daytime_by_hour = daytime.groupby('hour').agg({
    'is_efficient_time': 'mean',
    'price_mwh': 'mean',
    'gpu_utilization_pct': 'mean',
    'active_jobs': 'mean'
})

print("\nEfficiency by hour (9am-5pm):")
for hour, row in daytime_by_hour.iterrows():
    print(f"  {hour:02d}:00 - {row['is_efficient_time']*100:5.1f}% efficient | "
          f"Price: ${row['price_mwh']:5.1f} | Util: {row['gpu_utilization_pct']:4.1f}% | "
          f"Jobs: {row['active_jobs']:4.0f}")

# =============================================================================
# QUESTION 3: Is there a day-of-week pattern?
# =============================================================================
print("\n" + "="*80)
print("DAY-OF-WEEK PATTERN IN DAYTIME")
print("="*80)

days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

daytime = daytime.copy()
daytime['day_name'] = daytime['day_of_week'].map(days)
daytime_by_day = daytime.groupby('day_name').agg({
    'is_efficient_time': 'mean',
    'price_mwh': 'mean',
    'gpu_utilization_pct': 'mean'
})

print("\nEfficiency by day of week:")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for day in day_order:
    if day in daytime_by_day.index:
        row = daytime_by_day.loc[day]
        print(f"  {day:10s} - {row['is_efficient_time']*100:5.1f}% efficient | "
              f"Price: ${row['price_mwh']:5.1f} | Util: {row['gpu_utilization_pct']:4.1f}%")

# =============================================================================
# QUESTION 4: Are efficient daytime hours clustered or random?
# =============================================================================
print("\n" + "="*80)
print("CLUSTERING ANALYSIS")
print("="*80)

# Look at temporal autocorrelation
df_sorted = df.sort_values('timestamp')
df_sorted['next_hour_efficient'] = df_sorted['is_efficient_time'].shift(-1)

# For daytime hours, if current hour is efficient, is next hour also efficient?
daytime_sorted = df_sorted[df_sorted['hour'].between(9, 17)].copy()
persistence = daytime_sorted[daytime_sorted['is_efficient_time'] == 1]['next_hour_efficient'].mean()

print(f"\nIf a daytime hour is efficient, probability next hour is also efficient: {persistence:.1%}")

if persistence > 0.6:
    print("→ Efficient hours CLUSTER together (not random)")
else:
    print("→ Efficient hours are SCATTERED (more random)")

# =============================================================================
# QUESTION 5: Show specific examples
# =============================================================================
print("\n" + "="*80)
print("SPECIFIC EXAMPLES OF EFFICIENT DAYTIME HOURS")
print("="*80)

print("\nTop 5 most efficient daytime hours:")
top_efficient = efficient_daytime.nlargest(5, 'jobs_per_dollar')[
    ['timestamp', 'hour', 'day_of_week', 'price_mwh', 'gpu_utilization_pct', 
     'active_jobs', 'jobs_per_dollar']
]
print(top_efficient.to_string(index=False))

# =============================================================================
# FINAL VERDICT
# =============================================================================
print("\n" + "="*80)
print("VERDICT: ARE DAYTIME SPIKES RANDOM OR PATTERNED?")
print("="*80)

price_diff = inefficient_daytime['price_mwh'].mean() - efficient_daytime['price_mwh'].mean()
util_diff = inefficient_daytime['gpu_utilization_pct'].mean() - efficient_daytime['gpu_utilization_pct'].mean()

if price_diff > 10:
    print(f"✅ PRICE PATTERN: Efficient daytime is ${price_diff:.1f}/MWh cheaper")
    print("   → Not random - driven by electricity market dips")
elif util_diff > 10:
    print(f"✅ UTILIZATION PATTERN: Inefficient daytime has {util_diff:.1f}% higher utilization")
    print("   → Not random - driven by workload density")
else:
    print("⚠️  WEAK PATTERN: Differences are small")
    print("   → May be partially random or complex interaction")

if persistence > 0.6:
    print(f"✅ TEMPORAL CLUSTERING: {persistence:.1%} persistence")
    print("   → Efficient windows last multiple hours")
else:
    print(f"⚠️  LOW CLUSTERING: {persistence:.1%} persistence")
    print("   → Efficient hours are isolated events")

# Check if specific hours are consistently efficient
most_efficient_hour = daytime_by_hour['is_efficient_time'].idxmax()
efficiency_at_best_hour = daytime_by_hour.loc[most_efficient_hour, 'is_efficient_time']

if efficiency_at_best_hour > 0.6:
    print(f"✅ HOUR-OF-DAY PATTERN: {most_efficient_hour:02d}:00 is {efficiency_at_best_hour*100:.0f}% efficient")
    print("   → Specific times are reliably good")
else:
    print(f"⚠️  NO HOUR-OF-DAY PATTERN: Best hour is only {efficiency_at_best_hour:.1%} efficient")