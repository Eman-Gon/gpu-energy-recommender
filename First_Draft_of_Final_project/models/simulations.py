"""
Real-Time GPU Scheduling Simulation
Demonstrates ML-based scheduling vs Rule-based scheduling on 10,000 GPU cluster
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import time
import os

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

print(f"{BOLD}GPU CLUSTER SCHEDULING SIMULATOR{RESET}")
print("10,000 GPU Data Center - Real-Time Scheduling Decisions")

print("\n[1] Loading trained ML model ")

model = joblib.load('../results/logistic_regression_model.pkl')
scaler = joblib.load('../results/scaler.pkl')

with open('../results/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f" Loaded Logistic Regression model (97.6% accuracy)")
print(f" Features: {', '.join(feature_names[:5])}...")

print("\n[2] Simulation setup")

TOTAL_GPUS = 10000
GPU_POWER_WATTS = 300 
SIMULATION_HOURS = 24 

print(f" Cluster size: {TOTAL_GPUS:,} GPUs")
print(f" Power per GPU: {GPU_POWER_WATTS}W")
print(f" Simulation duration: {SIMULATION_HOURS} hours")

print("\n[3] Loading historical patterns...")

df = pd.read_csv('../../eda/merged_data_enhanced.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])

hourly_patterns = df.groupby('hour').agg({
    'price_mwh': ['mean', 'std'],
    'gpu_utilization_pct': ['mean', 'std'],
    'active_jobs': ['mean', 'std']
}).reset_index()

hourly_patterns.columns = ['hour', 'price_mean', 'price_std', 
                            'util_mean', 'util_std', 
                            'jobs_mean', 'jobs_std']

print(f" Loaded {len(df)} historical observations")
print(f" Price range: ${df['price_mwh'].min():.2f} - ${df['price_mwh'].max():.2f}/MWh")

class GPUClusterSimulator:
    def __init__(self, total_gpus, model, scaler, feature_names, hourly_patterns):
        self.total_gpus = total_gpus
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.hourly_patterns = hourly_patterns
        
        self.ml_total_cost = 0
        self.ml_jobs_completed = 0
        self.ml_jobs_delayed = 0
        
        self.rule_total_cost = 0
        self.rule_jobs_completed = 0
        self.rule_jobs_delayed = 0
        
        self.current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
    def generate_current_conditions(self, hour):
        """Generate realistic current conditions based on historical patterns"""
        pattern = self.hourly_patterns[self.hourly_patterns['hour'] == hour].iloc[0]
        
        price = np.random.normal(pattern['price_mean'], pattern['price_std'])
        price = max(price, 15)  
        
        util = np.random.normal(pattern['util_mean'], pattern['util_std'])
        util = np.clip(util, 5, 95)
        
        jobs = int(np.random.normal(pattern['jobs_mean'], pattern['jobs_std']))
        jobs = max(jobs, 10)
        
        return price, util, jobs
    
    def create_features(self, price, util, jobs, hour):
        """Create feature vector for ML model"""
        active_gpus = int(self.total_gpus * (util / 100))
        power_kw = (active_gpus * 300) / 1000  # Convert to kW
        hourly_cost = power_kw * (price / 1000)  # Cost in USD
        
        day_of_week = self.current_time.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_business_hours = 1 if (8 <= hour <= 18 and day_of_week < 5) else 0
        is_peak_hours = 1 if (14 <= hour <= 18) else 0
        
        features = {
            'price_mwh': price,
            'gpu_utilization_pct': util,
            'active_jobs': jobs,
            'power_consumption_kw': power_kw,
            'hourly_cost_usd': hourly_cost,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_business_hours': is_business_hours,
            'is_peak_hours': is_peak_hours
        }
        
        return features, hourly_cost
    
    def ml_decision(self, features):
        """ML-based scheduling decision"""
        feature_vector = [features[name] for name in self.feature_names]
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        prediction = self.model.predict(feature_vector_scaled)[0]
        probability = self.model.predict_proba(feature_vector_scaled)[0][1]
        
        return prediction == 1, probability
    
    def rule_decision(self, hour, price):
        """Simple rule-based scheduling: run at night OR when cheap"""
        is_night = hour < 8
        is_cheap = price < 45
        
        return is_night or is_cheap
    
    def simulate_hour(self, hour_num):
        """Simulate one hour of operations"""
        hour = self.current_time.hour

        price, util, pending_jobs = self.generate_current_conditions(hour)
        features, hourly_cost = self.create_features(price, util, pending_jobs, hour)
        
        ml_schedule, ml_confidence = self.ml_decision(features)
        
        rule_schedule = self.rule_decision(hour, price)
        
        jobs_per_dollar = pending_jobs / (hourly_cost + 0.01)
        
        if ml_schedule:
            self.ml_jobs_completed += pending_jobs
            self.ml_total_cost += hourly_cost
        else:
            self.ml_jobs_delayed += pending_jobs
        
        if rule_schedule:
            self.rule_jobs_completed += pending_jobs
            self.rule_total_cost += hourly_cost
        else:
            self.rule_jobs_delayed += pending_jobs
        
        self.display_hour(hour_num, hour, price, util, pending_jobs, hourly_cost,
                         ml_schedule, ml_confidence, rule_schedule, jobs_per_dollar)
        
        self.current_time += timedelta(hours=1)
    
    def display_hour(self, hour_num, hour, price, util, jobs, cost, 
                     ml_decision, ml_conf, rule_decision, jpd):
        
        time_str = f"{hour:02d}:00"
        
        ml_color = GREEN if ml_decision else RED
        rule_color = GREEN if rule_decision else RED
        
        is_actually_efficient = jpd > 124 
        actual_color = GREEN if is_actually_efficient else RED
        
        print(f"\n{BOLD}Hour {hour_num}/24 - {time_str}{RESET}")
        
        print(f"Current Conditions:")
        print(f"Electricity Price: ${price:6.2f}/MWh")
        print(f"GPU Utilization:   {util:5.1f}%")
        print(f"Pending Jobs:      {jobs:4d}")
        print(f"Hourly Cost:       ${cost:6.2f}")
        print(f"Jobs/Dollar:       {jpd:6.1f} {actual_color}({'EFFICIENT' if is_actually_efficient else 'INEFFICIENT'}){RESET}")
        
        print(f"\n  Scheduling Decisions:")
        print(f"    {ml_color}ML Model:    {'SCHEDULE NOW' if ml_decision else ' WAIT'} (confidence: {ml_conf:.1%}){RESET}")
        print(f"    {rule_color}Rule-Based: {'SCHEDULE NOW' if rule_decision else ' WAIT'}{RESET}")
        
        if ml_decision != rule_decision:
            print(f"\n    {YELLOW}  DECISIONS DIFFER!{RESET}")
            if ml_decision and not rule_decision:
                print(f" ML found opportunity that rule missed")
            else:
                print(f" ML avoided inefficient window that rule would schedule")

    def display_summary(self):
        """Display final simulation summary"""
        print(f"{BOLD}SIMULATION COMPLETE - 24 HOUR SUMMARY{RESET}")
        
        print(f"\n{BOLD}ML-BASED SCHEDULING:{RESET}")
        print(f"Jobs Completed:  {self.ml_jobs_completed:,}")
        print(f"Jobs Delayed:    {self.ml_jobs_delayed:,}")
        print(f"Total Cost:      ${self.ml_total_cost:,.2f}")
        print(f"Cost per Job:    ${self.ml_total_cost/max(self.ml_jobs_completed,1):.4f}")
        print(f"Efficiency:      {self.ml_jobs_completed/max(self.ml_total_cost,0.01):.2f} jobs/dollar")
        
        print(f"\n{BOLD}RULE-BASED SCHEDULING:{RESET}")
        print(f"Jobs Completed:  {self.rule_jobs_completed:,}")
        print(f"Jobs Delayed:    {self.rule_jobs_delayed:,}")
        print(f"Total Cost:      ${self.rule_total_cost:,.2f}")
        print(f"Cost per Job:    ${self.rule_total_cost/max(self.rule_jobs_completed,1):.4f}")
        print(f"Efficiency:      {self.rule_jobs_completed/max(self.rule_total_cost,0.01):.2f} jobs/dollar")
        
        print(f"\n{BOLD}COMPARISON:{RESET}")
        cost_savings = self.rule_total_cost - self.ml_total_cost
        cost_savings_pct = (cost_savings / self.rule_total_cost * 100) if self.rule_total_cost > 0 else 0
        
        jobs_diff = self.ml_jobs_completed - self.rule_jobs_completed
        
        if cost_savings > 0:
            print(f"  {GREEN} ML saved ${cost_savings:,.2f} ({cost_savings_pct:.1f}%){RESET}")
        else:
            print(f"  {RED} ML cost ${-cost_savings:,.2f} more ({-cost_savings_pct:.1f}%){RESET}")
        
        if jobs_diff > 0:
            print(f"  {GREEN} ML completed {jobs_diff:,} more jobs{RESET}")
        elif jobs_diff < 0:
            print(f"  {RED} ML completed {-jobs_diff:,} fewer jobs{RESET}")
        else:
            print(f"  {BLUE}= Same number of jobs completed{RESET}")
        
        yearly_savings = cost_savings * 365
        print(f"\n{BOLD}YEARLY PROJECTION:{RESET}")
        print(f" Estimated annual savings: {GREEN}${yearly_savings:,.2f}{RESET}")
        print(f" Estimated annual job difference: {jobs_diff * 365:,}")
        

print("\n[4] Starting simulation")

time.sleep(2)

simulator = GPUClusterSimulator(
    total_gpus=TOTAL_GPUS,
    model=model,
    scaler=scaler,
    feature_names=feature_names,
    hourly_patterns=hourly_patterns
)

try:
    for hour in range(SIMULATION_HOURS):
        simulator.simulate_hour(hour + 1)
        time.sleep(0.5)
        
except KeyboardInterrupt:
    print(f"\n\n{YELLOW}Simulation stopped by user{RESET}")

simulator.display_summary()

print(f"\n{BOLD}Simulation demonstrates real-time ML-based scheduling decisions{RESET}")
print(f"Model: Logistic Regression (97.6% accuracy on test set)")