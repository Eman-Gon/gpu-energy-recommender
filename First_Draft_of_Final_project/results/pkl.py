"""
Model Analysis - Inspect Trained Models
"""

import joblib
import pandas as pd

print("ANALYZING TRAINED MODELS\n")

models = {
    'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
    'Decision Tree': joblib.load('decision_tree_model.pkl'),
    'Random Forest': joblib.load('random_forest_model.pkl'),
    'XGBoost': joblib.load('xgboost_model.pkl')
}

scaler = joblib.load('scaler.pkl')

feature_names = [
    'price_mwh', 'gpu_utilization_pct', 'active_jobs', 'power_consumption_kw',
    'hourly_cost_usd', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'is_peak_hours'
]

print("MODEL DETAILS\n")

for name, model in models.items():
    print(f"{name}:")
    print(f"  Type: {type(model).__name__}")
    
    if name == 'Logistic Regression':
        print(f"  Max iterations: {model.max_iter}")
        print(f"  Solver: {model.solver}")
    elif name == 'Decision Tree':
        print(f"  Max depth: {model.max_depth}")
        print(f"  Number of leaves: {model.get_n_leaves()}")
    elif name == 'Random Forest':
        print(f"  Number of trees: {model.n_estimators}")
        print(f"  Max depth: {model.max_depth}")
    elif name == 'XGBoost':
        print(f"  Number of boosting rounds: {model.n_estimators}")
        print(f"  Max depth: {model.max_depth}")
        print(f"  Learning rate: {model.learning_rate}")
    print()

print("FEATURE IMPORTANCE COMPARISON\n")

importance_df = pd.DataFrame({'Feature': feature_names})

for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importance_df[name] = model.feature_importances_

print(importance_df.to_string(index=False))

print("\nTOP 3 FEATURES PER MODEL\n")

for name in ['Decision Tree', 'Random Forest', 'XGBoost']:
    if name in importance_df.columns:
        top3 = importance_df.nlargest(3, name)[['Feature', name]]
        print(f"{name}:")
        for idx, row in top3.iterrows():
            print(f"  {row['Feature']:25s} {row[name]:.3f}")
        print()

print("SCALER PARAMETERS\n")

scaler_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean': scaler.mean_,
    'Std Dev': scaler.scale_
})
print(scaler_df.to_string(index=False))

print("\nSUMMARY\n")
print("Loaded:")
print("  4 trained classification models")
print("  10 features per model")
print("  StandardScaler fitted on training data")

overall_importance = importance_df[['Decision Tree', 'Random Forest', 'XGBoost']].mean(axis=1)
top_feature_idx = overall_importance.idxmax()
top_feature = importance_df.loc[top_feature_idx, 'Feature']
print(f"\nMost important feature (average): {top_feature}")