"""
Model Evaluation and Visualization
CSC-466 Fall 2025 - Final Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*60)
print("MODEL EVALUATION & VISUALIZATION")
print("="*60)

# Load data
print("\n[1/4] Loading data and models...")
df = pd.read_csv('../eda/merged_data_enhanced.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Load feature names
with open('../results/feature_names.txt', 'r') as f:
    feature_cols = [line.strip() for line in f.readlines()]

# Prepare data
X = df[feature_cols]
y = df['is_efficient_time']

# Time-series split (same as training)
split_idx = int(len(df) * 0.75)
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# Load scaler and models
scaler = joblib.load('../results/scaler.pkl')
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': joblib.load('../results/logistic_regression_model.pkl'),
    'Decision Tree': joblib.load('../results/decision_tree_model.pkl'),
    'Random Forest': joblib.load('../results/random_forest_model.pkl'),
    'XGBoost': joblib.load('../results/xgboost_model.pkl')
}

print(f"   ✓ Loaded 4 models")
print(f"   ✓ Test set: {len(X_test)} samples")

# ============================================================================
# VISUALIZATION 1: MODEL COMPARISON BAR CHART
# ============================================================================
print("\n[2/4] Creating model comparison visualization...")

results_df = pd.read_csv('../results/metrics/model_comparison.csv')

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.2

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, results_df[metric], width, 
           label=metric, color=colors[i], alpha=0.8)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Classification Model Performance Comparison', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('../results/plots/model_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: results/plots/model_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: ROC CURVES
# ============================================================================
print("\n[3/4] Creating ROC curves...")

fig, ax = plt.subplots(figsize=(10, 8))

for name, model in models.items():
    if name == 'Logistic Regression':
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/plots/roc_curves.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: results/plots/roc_curves.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: CONFUSION MATRICES
# ============================================================================
print("\n[4/4] Creating confusion matrices...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    if name == 'Logistic Regression':
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Inefficient', 'Efficient'],
                yticklabels=['Inefficient', 'Efficient'],
                cbar_kws={'label': 'Count'})
    
    axes[idx].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'Confusion Matrix - {name}', 
                        fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: results/plots/confusion_matrices.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: FEATURE IMPORTANCE
# ============================================================================
print("\nCreating feature importance plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest
rf_model = models['Random Forest']
rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], 
             color='steelblue', alpha=0.8)
axes[0].set_xlabel('Importance', fontsize=11, fontweight='bold')
axes[0].set_title('Feature Importance - Random Forest', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].invert_yaxis()

# XGBoost
xgb_model = models['XGBoost']
xgb_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'], 
             color='coral', alpha=0.8)
axes[1].set_xlabel('Importance', fontsize=11, fontweight='bold')
axes[1].set_title('Feature Importance - XGBoost', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('../results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: results/plots/feature_importance.png")
plt.close()

print("\n" + "="*60)
print("✅ EVALUATION COMPLETE!")
print("="*60)
print("\nGenerated visualizations:")
print("  1. results/plots/model_comparison.png")
print("  2. results/plots/roc_curves.png")
print("  3. results/plots/confusion_matrices.png")
print("  4. results/plots/feature_importance.png")
print("\nYou're ready for your report!")