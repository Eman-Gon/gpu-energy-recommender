# Classification Models for GPU Energy Efficiency Prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("GPU ENERGY EFFICIENCY CLASSIFIER")
print("CSC-466 Final Project. The Training Script")


print("\n[1/6] Loading dataset")

df = pd.read_csv('../../eda/merged_data_enhanced.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Loaded {len(df)} records")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

print("\n[2/6] Preparing features and target")

feature_cols = [
    'price_mwh',
    'gpu_utilization_pct',
    'active_jobs',
    'power_consumption_kw',
    'hourly_cost_usd',
    'hour',
    'day_of_week',
    'is_weekend',
    'is_business_hours',
    'is_peak_hours'
]

X = df[feature_cols].copy()
y = df['is_efficient_time'].copy()

print(f"Features: {len(feature_cols)} columns")
print(f"Target distribution:")
print(f"Efficient (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"Inefficient (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")

print("\n[3/6] Creating time-series train/test split")

split_idx = int(len(df) * 0.75)  

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f" Training set: {len(X_train)} samples")
print(f" Test set: {len(X_test)} samples")
print(f" Split date: {df.iloc[split_idx]['timestamp']}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Features scaled")

print("\n[4/6] Training classification models")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=15,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=6,
        eval_metric='logloss',
        use_label_encoder=False
    )
}
trained_models = {}
results = []

for name, model in models.items():
    print(f"\n   Training {name}")
    

    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc
    })
    
    trained_models[name] = model
    
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    print(f"AUC-ROC:   {auc:.3f}")

print("\n[5/6] Saving results")

results_df = pd.DataFrame(results)
results_df.to_csv('../results/metrics/model_comparison.csv', index=False)
print(f" Saved metrics to: results/metrics/model_comparison.csv")

for name, model in trained_models.items():
    filename = name.lower().replace(' ', '_')
    joblib.dump(model, f'../results/{filename}_model.pkl')
print(f" Saved trained models to: results/")

joblib.dump(scaler, '../results/scaler.pkl')
print(f"Saved scaler val")

with open('../results/feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_cols))
print(f" Saved feature names")

print("\n[6/6] Training Summary:")

print("MODEL COMPARISON RESULTS")
print(results_df.to_string(index=False))


best_model = results_df.loc[results_df['F1-Score'].idxmax()]
print(f"\nBest Model (by F1-Score): {best_model['Model']}")
print(f" F1-Score: {best_model['F1-Score']:.3f}")
print(f" Accuracy: {best_model['Accuracy']:.3f}")
print(f" AUC-ROC:  {best_model['AUC-ROC']:.3f}")

print("\nDONE!!!")