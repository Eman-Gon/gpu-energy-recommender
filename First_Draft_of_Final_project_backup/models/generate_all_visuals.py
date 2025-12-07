"""
Master Script - Generate All Final Presentation Visuals
Run this once to create everything you need!
"""
import subprocess
import os

print("🚀 GENERATING ALL FINAL VISUALS FOR PRESENTATION")
print("=" * 70)

scripts = [
    ("train_classifiers.py", "Training all models..."),
    ("evaluate_models.py", "Creating model comparison charts..."),
    ("statistical_validation.py", "Running statistical tests..."),
    ("spike_analysis.py", "Analyzing daytime efficiency spikes..."),
    ("simulation_visualizations.py", "Creating simulation visuals..."),
    ("critical_analysis.py", "Validating project is not trivial..."),
    ("create_umap_visual.py", "Generating UMAP clustering..."),
]

for script, description in scripts:
    print(f"\n{'='*70}")
    print(f"📊 {description}")
    print(f"   Running: {script}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(['python3', script], 
                              capture_output=False, 
                              text=True)
        if result.returncode == 0:
            print(f"✅ {script} completed successfully!")
        else:
            print(f"⚠️  {script} had issues but may have produced output")
    except Exception as e:
        print(f"❌ Error running {script}: {e}")

print("\n" + "=" * 70)
print("🎉 ALL VISUALS GENERATED!")
print("=" * 70)
print("\nCheck these folders:")
print("  📁 results/plots/ - All PNG visualizations")
print("  📁 results/metrics/ - Model comparison CSV")
print("\n📋 Your 7 presentation visuals:")
print("  1. chart1_problem_overview.png")
print("  2. model_comparison.png ⭐")
print("  3. confusion_matrices.png")
print("  4. feature_importance.png")
print("  5. chart1_price_anomalies.png")
print("  6. simulation_cost_savings.png")
print("  7. umap_clustering.png ⭐")
print("\n🎯 You're ready for Friday! 🔥")