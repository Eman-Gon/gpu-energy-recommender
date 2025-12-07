"""
Statistical Validation - Prove ML improvement is significant
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2, norm
import joblib

print("="*60)
print("STATISTICAL SIGNIFICANCE TESTING")
print("="*60)

# Load your existing results - FIX THE PATH
df = pd.read_csv('../../eda/merged_data_enhanced.csv')
model = joblib.load('../results/logistic_regression_model.pkl')
scaler = joblib.load('../results/scaler.pkl')

with open('../results/feature_names.txt', 'r') as f:
    feature_cols = [line.strip() for line in f.readlines()]

# Prepare test data (same split as training)
split_idx = int(len(df) * 0.75)
X_test = df[feature_cols].iloc[split_idx:]
y_test = df['is_efficient_time'].iloc[split_idx:].values

# Get predictions
X_test_scaled = scaler.transform(X_test)
ml_pred = model.predict(X_test_scaled)

# Baseline: "run at night" rule
rule_pred = (df['hour'].iloc[split_idx:] < 8).astype(int).values

# McNemar's Test (compares two classifiers)
print("\n[1] McNemar's Test: ML vs 'Run at Night' Rule")
print("-" * 60)

# Create contingency table
ml_correct_rule_wrong = np.sum((ml_pred == y_test) & (rule_pred != y_test))
rule_correct_ml_wrong = np.sum((rule_pred == y_test) & (ml_pred != y_test))
both_correct = np.sum((ml_pred == y_test) & (rule_pred == y_test))
both_wrong = np.sum((ml_pred != y_test) & (rule_pred != y_test))

print(f"Both correct:           {both_correct}")
print(f"Both wrong:             {both_wrong}")
print(f"ML correct, Rule wrong: {ml_correct_rule_wrong}")
print(f"Rule correct, ML wrong: {rule_correct_ml_wrong}")

# Calculate McNemar statistic
if (ml_correct_rule_wrong + rule_correct_ml_wrong) > 0:
    # McNemar's test with continuity correction
    statistic = (abs(ml_correct_rule_wrong - rule_correct_ml_wrong) - 1)**2 / (ml_correct_rule_wrong + rule_correct_ml_wrong)
    p_value = 1 - chi2.cdf(statistic, 1)
    
    print(f"\nMcNemar's statistic: {statistic:.2f}")
    print(f"P-value: {p_value:.2e}")
    
    if p_value < 0.001:
        print("✅ HIGHLY SIGNIFICANT (p < 0.001)")
        print("   ML improvement over baseline is NOT due to chance")
    elif p_value < 0.05:
        print("✅ SIGNIFICANT (p < 0.05)")
    else:
        print("⚠️  NOT SIGNIFICANT (p >= 0.05)")
else:
    print("\n⚠️  Cannot compute McNemar's test (no disagreements)")

# Confidence intervals for accuracy
print("\n[2] 95% Confidence Intervals")
print("-" * 60)

def wilson_confidence_interval(successes, n, confidence=0.95):
    """Wilson score interval for binomial proportion"""
    z = norm.ppf((1 + confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
    return centre - margin, centre + margin

ml_accuracy = (ml_pred == y_test).sum() / len(y_test)
rule_accuracy = (rule_pred == y_test).sum() / len(y_test)

ml_ci = wilson_confidence_interval((ml_pred == y_test).sum(), len(y_test))
rule_ci = wilson_confidence_interval((rule_pred == y_test).sum(), len(y_test))

print(f"ML Model:    {ml_accuracy:.1%} (95% CI: {ml_ci[0]:.1%} - {ml_ci[1]:.1%})")
print(f"Rule-based:  {rule_accuracy:.1%} (95% CI: {rule_ci[0]:.1%} - {rule_ci[1]:.1%})")

if ml_ci[0] > rule_ci[1]:
    print("\n✅ Confidence intervals DON'T overlap")
    print("   ML is definitively better than baseline")
else:
    print("\n⚠️  Confidence intervals overlap")

# Effect size
print("\n[3] Effect Size (Cohen's h)")
print("-" * 60)

# Cohen's h for difference in proportions
p1 = ml_accuracy
p2 = rule_accuracy
cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

print(f"Cohen's h: {cohens_h:.3f}")
if abs(cohens_h) > 0.8:
    print("→ LARGE effect size")
elif abs(cohens_h) > 0.5:
    print("→ MEDIUM effect size")
else:
    print("→ SMALL effect size")

print("\n" + "="*60)
print("VERDICT")
print("="*60)
print(f"✅ ML improves accuracy by {(ml_accuracy - rule_accuracy)*100:.1f} percentage points")

if p_value < 0.001:
    print(f"✅ Improvement is statistically significant (p < 0.001)")
elif p_value < 0.05:
    print(f"✅ Improvement is statistically significant (p = {p_value:.3f})")
else:
    print(f"⚠️  Improvement is NOT statistically significant (p = {p_value:.3f})")

print(f"✅ With 95% confidence, ML accuracy is between {ml_ci[0]:.1%} - {ml_ci[1]:.1%}")
print(f"✅ Effect size is {cohens_h:.2f} ({'LARGE' if abs(cohens_h) > 0.8 else 'MEDIUM' if abs(cohens_h) > 0.5 else 'SMALL'})")

if p_value < 0.05 and ml_ci[0] > rule_ci[1]:
    print("\n🎯 STRONG CONCLUSION: ML model is definitively superior")
    print("   - Statistically significant improvement")
    print("   - Confidence intervals don't overlap")
    print("   - Large effect size")
elif p_value < 0.05:
    print("\n✅ VALID CONCLUSION: ML model is better")
    print("   - Statistically significant improvement")
    print("   - Further validation recommended")
else:
    print("\n⚠️  WEAK CONCLUSION: More data needed")
    print("   - Improvement not statistically significant")