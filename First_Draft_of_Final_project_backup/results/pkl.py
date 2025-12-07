"""
Model Analysis Script - Inspect All Your Trained Models
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("🔍 ANALYZING YOUR TRAINED MODELS")
print("="*70)

# Load all models
models = {
    'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
    'Decision Tree': joblib.load('decision_tree_model.pkl'),
    'Random Forest': joblib.load('random_forest_model.pkl'),
    'XGBoost': joblib.load('xgboost_model.pkl')
}

scaler = joblib.load('scaler.pkl')

print("\n✅ All models loaded successfully!")

# Feature names (from your training)
feature_names = [
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

print("\n" + "="*70)
print("📊 MODEL DETAILS")
print("="*70)

# Analyze each model
for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"🤖 {name}")
    print(f"{'='*70}")
    
    # Model type
    print(f"Type: {type(model).__name__}")
    
    # Key parameters
    if name == 'Logistic Regression':
        print(f"Max iterations: {model.max_iter}")
        print(f"Solver: {model.solver}")
        print(f"Classes: {model.classes_}")
        
    elif name == 'Decision Tree':
        print(f"Max depth: {model.max_depth}")
        print(f"Number of leaves: {model.get_n_leaves()}")
        print(f"Tree depth: {model.get_depth()}")
        
    elif name == 'Random Forest':
        print(f"Number of trees: {model.n_estimators}")
        print(f"Max depth: {model.max_depth}")
        print(f"Number of features per split: {model.max_features}")
        
    elif name == 'XGBoost':
        print(f"Number of boosting rounds: {model.n_estimators}")
        print(f"Max depth: {model.max_depth}")
        print(f"Learning rate: {model.learning_rate}")

# Feature importance comparison
print("\n" + "="*70)
print("📈 FEATURE IMPORTANCE COMPARISON")
print("="*70)

importance_df = pd.DataFrame({
    'Feature': feature_names
})

# Get feature importances from tree-based models
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importance_df[name] = model.feature_importances_

print("\n" + importance_df.to_string(index=False))

# Top 3 features for each model
print("\n" + "="*70)
print("🏆 TOP 3 MOST IMPORTANT FEATURES PER MODEL")
print("="*70)

for name in ['Decision Tree', 'Random Forest', 'XGBoost']:
    if name in importance_df.columns:
        top3 = importance_df.nlargest(3, name)[['Feature', name]]
        print(f"\n{name}:")
        for idx, row in top3.iterrows():
            print(f"  {row['Feature']:25s} {row[name]:.3f}")

# Scaler information
print("\n" + "="*70)
print("🔧 SCALER INFORMATION (StandardScaler)")
print("="*70)

print(f"\nFeature scaling parameters:")
scaler_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean': scaler.mean_,
    'Std Dev': scaler.scale_
})
print(scaler_df.to_string(index=False))

# Create a visualization of feature importance
print("\n" + "="*70)
print("📊 Creating feature importance comparison chart...")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

tree_models = ['Decision Tree', 'Random Forest', 'XGBoost']
colors = ['#3498db', '#2ecc71', '#e74c3c']

for idx, (name, color) in enumerate(zip(tree_models, colors)):
    if name in importance_df.columns:
        data = importance_df.sort_values(name, ascending=True)
        axes[idx].barh(data['Feature'], data[name], color=color, alpha=0.8)
        axes[idx].set_xlabel('Importance', fontweight='bold')
        axes[idx].set_title(f'{name}\nFeature Importance', fontweight='bold', fontsize=12)
        axes[idx].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_feature_importance_comparison.png")
plt.close()

# Summary
print("\n" + "="*70)
print("📋 SUMMARY")
print("="*70)

print("\n✅ What you have:")
print("  • 4 trained classification models")
print("  • All models use 10 features")
print("  • StandardScaler fitted on training data")
print("  • Models ready for predictions")

print("\n🎯 Key Findings:")
# Find the most important feature overall
overall_importance = importance_df[['Decision Tree', 'Random Forest', 'XGBoost']].mean(axis=1)
top_feature_idx = overall_importance.idxmax()
top_feature = importance_df.loc[top_feature_idx, 'Feature']
print(f"  • Most important feature (average): {top_feature}")

print("\n📊 Visualizations created:")
print("  • model_feature_importance_comparison.png")

print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)