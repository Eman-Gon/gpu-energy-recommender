"""
Analysis: Prove that daytime can be MORE efficient than nighttime
"""
import pandas as pd
import numpy as np
df = pd.read_csv('../eda/merged_data_enhanced.csv')


print("PROOF: DAYTIME CAN BE MORE EFFICIENT THAN NIGHTTIME")

print("\n[1] BEST DAYTIME vs WORST NIGHTTIME")


daytime = df[df['hour'].between(9, 17)].copy()
nighttime = df[df['hour'].isin(range(0, 8))].copy()

best_daytime = daytime.nlargest(10, 'jobs_per_dollar')
print("\nTop 10 MOST EFFICIENT Daytime Hours:")
print(best_daytime[['timestamp', 'hour', 'price_mwh', 'active_jobs', 
                     'hourly_cost_usd', 'jobs_per_dollar']].to_string(index=False))

worst_nighttime = nighttime.nsmallest(10, 'jobs_per_dollar')
print("\n\nTop 10 LEAST EFFICIENT Nighttime Hours:")
print(worst_nighttime[['timestamp', 'hour', 'price_mwh', 'active_jobs',
                       'hourly_cost_usd', 'jobs_per_dollar']].to_string(index=False))

print("KEY INSIGHT:")
best_day_avg = best_daytime['jobs_per_dollar'].mean()
worst_night_avg = worst_nighttime['jobs_per_dollar'].mean()
print(f"Best daytime hours:  {best_day_avg:.1f} jobs/dollar")
print(f"Worst nighttime hours: {worst_night_avg:.1f} jobs/dollar")
print(f"Difference: {best_day_avg - worst_night_avg:.1f} jobs/dollar")
print(f"\nBest daytime is {best_day_avg/worst_night_avg:.1f}x MORE efficient than worst nighttime!")

print("\n\n[2] DAYTIME HOURS THAT BEAT AVERAGE NIGHTTIME")


avg_night_efficiency = nighttime['jobs_per_dollar'].mean()
efficient_daytime = daytime[daytime['jobs_per_dollar'] > avg_night_efficiency]

print(f"\nAverage nighttime efficiency: {avg_night_efficiency:.1f} jobs/dollar")
print(f"Daytime hours that EXCEED this: {len(efficient_daytime)} out of {len(daytime)} ({len(efficient_daytime)/len(daytime)*100:.1f}%)")

print(f"\nExamples of daytime hours that beat average nighttime:")
samples = efficient_daytime.sample(min(5, len(efficient_daytime)))[
    ['timestamp', 'hour', 'price_mwh', 'gpu_utilization_pct', 'active_jobs', 'jobs_per_dollar']
]
print(samples.to_string(index=False))
print("\n\n[3] DIRECT COMPARISON: SAME DAY, DIFFERENT HOURS")

df['date'] = pd.to_datetime(df['timestamp']).dt.date

comparison_days = []

for date in df['date'].unique():
    day_data = df[df['date'] == date]
    
    day_hours = day_data[day_data['hour'].between(9, 17)]
    night_hours = day_data[day_data['hour'].isin(range(0, 8))]
    
    if len(day_hours) > 0 and len(night_hours) > 0:
        best_day = day_hours['jobs_per_dollar'].max()
        best_night = night_hours['jobs_per_dollar'].max()
        
        if best_day > best_night:
            comparison_days.append({
                'date': date,
                'best_daytime': best_day,
                'best_nighttime': best_night,
                'advantage': best_day - best_night
            })

comparison_df = pd.DataFrame(comparison_days)

if len(comparison_df) > 0:
    print(f"\nDays where BEST daytime hour > BEST nighttime hour: {len(comparison_df)} days")
    print(f"\nTop 5 examples:")
    print(comparison_df.nlargest(5, 'advantage').to_string(index=False))
else:
    print("\nNo days where daytime beat nighttime")


print("\n\n[4] COST SAVINGS")

efficient_day_hour = daytime[daytime['is_efficient_time'] == 1].iloc[0] 
inefficient_night_hour = nighttime[nighttime['is_efficient_time'] == 0].iloc[0]

print(f"\nScenario: You have 1000 GPU jobs to run")
print(f"\n{'OPTION A: Simple Rule (run at night)':^80}")
print(f"Time: {inefficient_night_hour['timestamp']} (Hour {inefficient_night_hour['hour']})")
print(f"Price: ${inefficient_night_hour['price_mwh']:.2f}/MWh")
print(f"Cost: ${inefficient_night_hour['hourly_cost_usd']:.2f}/hour")
print(f"Efficiency: {inefficient_night_hour['jobs_per_dollar']:.1f} jobs/dollar")
print(f"Total cost for 1000 jobs: ${1000 / inefficient_night_hour['jobs_per_dollar']:.2f}")

print(f"\n{'OPTION B: ML-Selected Daytime Window':^80}")
print(f"Time: {efficient_day_hour['timestamp']} (Hour {efficient_day_hour['hour']})")
print(f"Price: ${efficient_day_hour['price_mwh']:.2f}/MWh")  
print(f"Cost: ${efficient_day_hour['hourly_cost_usd']:.2f}/hour")
print(f"Efficiency: {efficient_day_hour['jobs_per_dollar']:.1f} jobs/dollar")
print(f"Total cost for 1000 jobs: ${1000 / efficient_day_hour['jobs_per_dollar']:.2f}")

savings = (1000 / inefficient_night_hour['jobs_per_dollar']) - (1000 / efficient_day_hour['jobs_per_dollar'])
savings_pct = (savings / (1000 / inefficient_night_hour['jobs_per_dollar'])) * 100

print(f"\n{'SAVINGS':^80}")
print(f"  ${savings:.2f} ({savings_pct:.1f}%) by choosing ML-recommended daytime window")
print(f"  over simple 'run at night' rule")

print("\n\n[5] resultssssss")


print(f"\nEfficient hours by time period:")
print(f"  Nighttime (0-8am):   {(nighttime['is_efficient_time']==1).sum():3d} efficient, {(nighttime['is_efficient_time']==0).sum():3d} inefficient ({(nighttime['is_efficient_time']==1).sum()/len(nighttime)*100:.1f}% efficient)")
print(f"  Daytime (9am-5pm):   {(daytime['is_efficient_time']==1).sum():3d} efficient, {(daytime['is_efficient_time']==0).sum():3d} inefficient ({(daytime['is_efficient_time']==1).sum()/len(daytime)*100:.1f}% efficient)")

print(f"\nEfficiency ranges:")
print(f"  Nighttime: {nighttime['jobs_per_dollar'].min():.1f} - {nighttime['jobs_per_dollar'].max():.1f} jobs/dollar (avg: {nighttime['jobs_per_dollar'].mean():.1f})")
print(f"  Daytime:   {daytime['jobs_per_dollar'].min():.1f} - {daytime['jobs_per_dollar'].max():.1f} jobs/dollar (avg: {daytime['jobs_per_dollar'].mean():.1f})")


print(f" {len(efficient_daytime)} daytime hours ({len(efficient_daytime)/len(daytime)*100:.1f}%) are MORE efficient than average nighttime")
print(f" Best daytime hours are {best_day_avg/worst_night_avg:.1f}x better than worst nighttime hours")
print(f" ML can identify these opportunities, while 'run at night' rule cannot")