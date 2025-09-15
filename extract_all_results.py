import json
import numpy as np
import os

results_dir = 'phase2_results_20250912_195334'

print("="*80)
print("COMPLETE PHASE 2 RESULTS EXTRACTION")
print("="*80)

# 1. Load all data
with open(f'{results_dir}/complete_results.json') as f:
    complete = json.load(f)
    
with open(f'{results_dir}/ground_truth.json') as f:
    ground_truth = json.load(f)
    
with open(f'{results_dir}/predictions.json') as f:
    predictions = json.load(f)
    
with open(f'{results_dir}/metrics.json') as f:
    metrics = json.load(f)

# 2. Basic Statistics
print("\n1. OVERALL STATISTICS")
print("-"*40)
print(f"Total parameters tested: {len(ground_truth)}")
print(f"Baseline model accuracy: {complete['baseline_accuracy']:.4f}")
print(f"Baseline model loss: {complete['baseline_loss']:.6f}")

# 3. Ground Truth Distribution
detrimental = sum(1 for v in ground_truth.values() if v['label'] == 'detrimental')
beneficial = len(ground_truth) - detrimental
print(f"\n2. GROUND TRUTH DISTRIBUTION")
print("-"*40)
print(f"Detrimental parameters (ΔL < 0): {detrimental} ({100*detrimental/len(ground_truth):.2f}%)")
print(f"Beneficial parameters (ΔL > 0): {beneficial} ({100*beneficial/len(ground_truth):.2f}%)")

# 4. Loss Change Analysis
delta_losses = [v['delta_loss'] for v in ground_truth.values()]
delta_accs = [v['delta_accuracy'] for v in ground_truth.values()]

print(f"\n3. LOSS CHANGE STATISTICS")
print("-"*40)
print(f"Mean ΔLoss: {np.mean(delta_losses):.6f}")
print(f"Std ΔLoss: {np.std(delta_losses):.6f}")
print(f"Min ΔLoss: {np.min(delta_losses):.6f}")
print(f"Max ΔLoss: {np.max(delta_losses):.6f}")
print(f"Median ΔLoss: {np.median(delta_losses):.6f}")

print(f"\n4. ACCURACY CHANGE STATISTICS")
print("-"*40)
print(f"Mean ΔAccuracy: {np.mean(delta_accs):.6f}")
print(f"Std ΔAccuracy: {np.std(delta_accs):.6f}")
print(f"Min ΔAccuracy: {np.min(delta_accs):.6f}")
print(f"Max ΔAccuracy: {np.max(delta_accs):.6f}")
print(f"Median ΔAccuracy: {np.median(delta_accs):.6f}")

# 5. Influence Score Analysis
influences = [v['influence'] for v in predictions.values()]
print(f"\n5. INFLUENCE SCORE STATISTICS")
print("-"*40)
print(f"Mean influence: {np.mean(influences):.6f}")
print(f"Std influence: {np.std(influences):.6f}")
print(f"Min influence: {np.min(influences):.6f}")
print(f"Max influence: {np.max(influences):.6f}")
print(f"Median influence: {np.median(influences):.6f}")
print(f"% positive (predict detrimental): {100*sum(1 for i in influences if i > 0)/len(influences):.2f}%")
print(f"% negative (predict beneficial): {100*sum(1 for i in influences if i < 0)/len(influences):.2f}%")

# 6. Confusion Matrix Details
cm = metrics['confusion_matrix']
print(f"\n6. CONFUSION MATRIX ANALYSIS")
print("-"*40)
print(f"True Positives (correctly predicted detrimental): {cm['true_positives']}")
print(f"False Positives (predicted detrimental, actually beneficial): {cm['false_positives']}")
print(f"True Negatives (correctly predicted beneficial): {cm['true_negatives']}")
print(f"False Negatives (predicted beneficial, actually detrimental): {cm['false_negatives']}")

# 7. Performance by Magnitude
influences_arr = np.array(influences)
delta_losses_arr = np.array(delta_losses)

# Quartile analysis
q1_idx = np.where(influences_arr < np.percentile(influences_arr, 25))[0]
q2_idx = np.where((influences_arr >= np.percentile(influences_arr, 25)) & 
                  (influences_arr < np.percentile(influences_arr, 50)))[0]
q3_idx = np.where((influences_arr >= np.percentile(influences_arr, 50)) & 
                  (influences_arr < np.percentile(influences_arr, 75)))[0]
q4_idx = np.where(influences_arr >= np.percentile(influences_arr, 75))[0]

print(f"\n7. ACCURACY BY INFLUENCE QUARTILES")
print("-"*40)
print(f"Q1 (most negative influence): {np.mean(delta_losses_arr[q1_idx] < 0)*100:.1f}% actually detrimental")
print(f"Q2: {np.mean(delta_losses_arr[q2_idx] < 0)*100:.1f}% actually detrimental")
print(f"Q3: {np.mean(delta_losses_arr[q3_idx] < 0)*100:.1f}% actually detrimental")
print(f"Q4 (most positive influence): {np.mean(delta_losses_arr[q4_idx] < 0)*100:.1f}% actually detrimental")

# 8. Extreme cases
top_100_neg = np.argsort(influences_arr)[:100]
top_100_pos = np.argsort(influences_arr)[-100:]

print(f"\n8. EXTREME CASES PERFORMANCE")
print("-"*40)
print(f"Top 100 most negative influences:")
print(f"  Predicted beneficial, actually detrimental: {np.mean(delta_losses_arr[top_100_neg] < 0)*100:.1f}%")
print(f"Top 100 most positive influences:")
print(f"  Predicted detrimental, actually beneficial: {np.mean(delta_losses_arr[top_100_pos] > 0)*100:.1f}%")

# 9. Parameter names analysis
param_names = [v['param_name'] for v in ground_truth.values()]
unique_layers = list(set([name.split('.')[0] for name in param_names]))
print(f"\n9. PARAMETER DISTRIBUTION BY LAYER")
print("-"*40)
for layer in sorted(unique_layers)[:10]:  # Show first 10 layer types
    layer_params = [i for i, name in enumerate(param_names) if name.startswith(layer)]
    if layer_params:
        layer_detrimental = np.mean([delta_losses[i] < 0 for i in layer_params])
        print(f"{layer}: {len(layer_params)} params, {layer_detrimental*100:.1f}% detrimental")

# 10. Find the worst predictions
all_errors = []
for idx in ground_truth.keys():
    true_delta = ground_truth[idx]['delta_loss']
    influence = predictions[idx]['influence']
    # Error: predicted detrimental but actually beneficial, or vice versa
    if (influence > 0 and true_delta > 0) or (influence < 0 and true_delta < 0):
        error_magnitude = abs(true_delta)
        all_errors.append({
            'idx': idx,
            'param_name': ground_truth[idx]['param_name'],
            'influence': influence,
            'actual_delta': true_delta,
            'error_magnitude': error_magnitude
        })

all_errors.sort(key=lambda x: x['error_magnitude'], reverse=True)

print(f"\n10. WORST PREDICTIONS (Highest Impact Errors)")
print("-"*40)
for i, err in enumerate(all_errors[:5]):
    print(f"{i+1}. {err['param_name']}")
    print(f"   Influence: {err['influence']:.6f} (predicted {'detrimental' if err['influence'] > 0 else 'beneficial'})")
    print(f"   Actual ΔL: {err['actual_delta']:.6f} (actually {'detrimental' if err['actual_delta'] < 0 else 'beneficial'})")
    print(f"   Impact: {err['error_magnitude']:.6f}")

print("\n" + "="*80)
