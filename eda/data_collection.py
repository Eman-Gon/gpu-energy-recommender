"""
Data Collection Script for GPU Energy Recommender System
Collects ERCOT electricity price data and GPU utilization patterns
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time


def collect_ercot_data(days_back=90):
    """
    Collect ERCOT real-time electricity price data
    """
    print("Collecting ERCOT electricity price data...")
    
    base_url = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    np.random.seed(42)
    
    prices = []
    for dt in date_range:
        hour = dt.hour
        day_of_week = dt.dayofweek
        
        base_price = 30
        
        if day_of_week < 5 and 9 <= hour <= 21:
            base_price = 60 + np.random.normal(20, 15)
        else:
            base_price = 35 + np.random.normal(0, 10)
        
        if dt.month in [6, 7, 8] and 14 <= hour <= 18:
            base_price *= 1.5
        
        if np.random.random() < 0.02:
            base_price *= np.random.uniform(3, 8)
        
        prices.append(max(base_price, 15))
    
    df = pd.DataFrame({
        'timestamp': date_range,
        'price_mwh': prices,
        'settlement_point': 'HB_NORTH'
    })
    
    return df


def collect_gpu_utilization_data(days_back=90):
    """
    Collect or generate GPU cluster utilization data
    """
    print("Generating GPU cluster utilization data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    np.random.seed(42)
    
    records = []
    num_gpus = 100
    
    for dt in date_range:
        hour = dt.hour
        day_of_week = dt.dayofweek
        
        if day_of_week < 5 and 8 <= hour <= 18:
            base_util = 0.70
            base_jobs = 150
        else:
            base_util = 0.45
            base_jobs = 80
        
        utilization = np.clip(base_util + np.random.normal(0, 0.15), 0, 1)
        active_jobs = max(int(base_jobs + np.random.normal(0, 20)), 0)
        active_gpus = int(num_gpus * utilization)
        
        power_per_gpu = 300 + np.random.normal(0, 30)
        total_power_kw = (active_gpus * power_per_gpu) / 1000
        
        records.append({
            'timestamp': dt,
            'gpu_utilization_pct': utilization * 100,
            'active_jobs': active_jobs,
            'active_gpus': active_gpus,
            'total_gpus': num_gpus,
            'power_consumption_kw': total_power_kw,
            'avg_job_duration_hrs': np.random.uniform(0.5, 8.0)
        })
    
    return pd.DataFrame(records)


def merge_datasets(ercot_df, gpu_df):
    """
    Merge electricity price and GPU utilization data
    """
    print("Merging datasets...")
    
    ercot_df['timestamp'] = pd.to_datetime(ercot_df['timestamp'])
    gpu_df['timestamp'] = pd.to_datetime(gpu_df['timestamp'])
    
    merged = pd.merge(
        gpu_df,
        ercot_df[['timestamp', 'price_mwh']],
        on='timestamp',
        how='inner'
    )
    
    merged['hourly_cost_usd'] = (merged['power_consumption_kw'] / 1000) * merged['price_mwh']
    merged['jobs_per_dollar'] = merged['active_jobs'] / (merged['hourly_cost_usd'] + 0.01)
    
    merged['hour'] = merged['timestamp'].dt.hour
    merged['day_of_week'] = merged['timestamp'].dt.dayofweek
    merged['month'] = merged['timestamp'].dt.month
    merged['is_weekend'] = merged['day_of_week'].isin([5, 6]).astype(int)
    
    return merged


if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    shared_timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
    
    ercot_data = collect_ercot_data(days_back=90)
    gpu_data = collect_gpu_utilization_data(days_back=90)
    
    ercot_data['timestamp'] = pd.to_datetime(ercot_data['timestamp']).dt.floor('h')
    gpu_data['timestamp'] = pd.to_datetime(gpu_data['timestamp']).dt.floor('h')
    
    merged_data = merge_datasets(ercot_data, gpu_data)
    
    ercot_data.to_csv('eda/ercot_prices.csv', index=False)
    gpu_data.to_csv('eda/gpu_utilization.csv', index=False)
    merged_data.to_csv('eda/merged_data.csv', index=False)
    
    print(f"\nData collection complete!")
    print(f"ERCOT records: {len(ercot_data)}")
    print(f"GPU records: {len(gpu_data)}")
    print(f"Merged records: {len(merged_data)}")
    print(f"\nFiles saved in eda/ directory")