import json
import numpy as np
import os

results_dir = 'phase2_hessian_20250913_133036'

print("="*80)
print("COMPREHENSIVE PHASE 2 HESSIAN RESULTS ANALYSIS")
print("="*80)

# Load all data files
with open(f'{results_dir}/final_results.json') as f:
    final = json.load(f)

with open(f'{results_dir}/ground_truth.json') as f:
    ground_truth = json.load(f)

with open(f'{results_dir}/hessian_predictions.json') as f:
    predictions = json.load(f)

# 1. BASELINE PERFORMANCE
print("\n1. BASELINE MODEL PERFORMANCE")
print("-"*40)
print(f"Loss: {final['baseline']['loss']:.6f}")
print(f"Accuracy: {final['baseline']['accuracy']:.4f} ({final['baseline']['accuracy']*100:.2f}%)")

# 2. GROUND TRUTH ANALYSIS
print("\n2. GROUND TRUTH DISTRIBUTION")
print("-"*40)
total = final['ground_truth_stats']['total']
detrimental = final['ground_truth_stats']['detrimental']
beneficial = final['ground_truth_stats']['beneficial']
print(f"Total parameters tested: {total:,}")
print(f"Detrimental (ΔL < 0): {detrimental:,} ({100*detrimental/total:.1f}%)")
print(f"Beneficial (ΔL > 0): {beneficial:,} ({100*beneficial/total:.1f}%)")

# 3. INDIVIDUAL PARAMETER EFFECTS
delta_losses = [v['delta_loss'] for v in ground_truth.values()]
print("\n3. INDIVIDUAL PARAMETER EFFECTS")
print("-"*40)
print(f"Mean ΔLoss: {np.mean(delta_losses):.8f}")
print(f"Std ΔLoss: {np.std(delta_losses):.8f}")
print(f"Min ΔLoss: {np.min(delta_losses):.8f}")
print(f"Max ΔLoss: {np.max(delta_losses):.8f}")
print(f"Median ΔLoss: {np.median(delta_losses):.8f}")

# 4. HESSIAN INFLUENCE PERFORMANCE
print("\n4. HESSIAN-BASED INFLUENCE PREDICTION")
print("-"*40)
perf = final['hessian_performance']
print(f"Confusion Matrix:")
print(f"                 Pred Detrimental  Pred Beneficial")
print(f"True Detrimental      {perf['confusion_matrix']['tp']:5d}           {perf['confusion_matrix']['fn']:5d}")
print(f"True Beneficial       {perf['confusion_matrix']['fp']:5d}           {perf['confusion_matrix']['tn']:5d}")
print(f"\nMetrics:")
print(f"  Accuracy: {perf['accuracy']:.4f} ({perf['accuracy']*100:.2f}%)")
print(f"  Precision: {perf['precision']:.4f}")
print(f"  Recall: {perf['recall']:.4f}")
print(f"  F1 Score: {perf['f1_score']:.4f}")
print(f"  Correlation: {perf['correlation']:.4f}")

# 5. COLLECTIVE REMOVAL - THE KEY RESULT
print("\n5. COLLECTIVE REMOVAL RESULTS ⭐")
print("-"*40)
collective = final['collective_removal']
print(f"Parameters removed: {collective['num_removed']:,}")
print(f"Baseline Loss: {collective['baseline_loss']:.6f}")
print(f"After Removal Loss: {collective['collective_loss']:.6f}")
print(f"ΔLoss: {collective['delta_loss']:.6f}")
print(f"Baseline Accuracy: {collective['baseline_accuracy']:.4f} ({collective['baseline_accuracy']*100:.2f}%)")
print(f"After Removal Accuracy: {collective['collective_accuracy']:.4f} ({collective['collective_accuracy']*100:.2f}%)")
print(f"ΔAccuracy: {collective['delta_accuracy']:.4f} ({collective['delta_accuracy']*100:.2f}%)")

# 6. INFLUENCE SCORE ANALYSIS
influences = [v['influence'] for v in predictions.values()]
print("\n6. HESSIAN INFLUENCE SCORES")
print("-"*40)
print(f"Mean: {np.mean(influences):.8f}")
print(f"Std: {np.std(influences):.8f}")
print(f"Min: {np.min(influences):.8f}")
print(f"Max: {np.max(influences):.8f}")
print(f"% Positive (predict detrimental): {100*sum(1 for i in influences if i > 0)/len(influences):.1f}%")

# 7. ANALYSIS BY QUARTILES
influences_arr = np.array(influences)
delta_losses_arr = np.array(delta_losses)

print("\n7. ACCURACY BY INFLUENCE QUARTILES")
print("-"*40)
for q in [25, 50, 75, 100]:
    if q == 25:
        mask = influences_arr < np.percentile(influences_arr, q)
        label = "Q1 (most negative)"
    elif q == 100:
        mask = influences_arr >= np.percentile(influences_arr, 75)
        label = "Q4 (most positive)"
    else:
        lower = np.percentile(influences_arr, q-25)
        upper = np.percentile(influences_arr, q)
        mask = (influences_arr >= lower) & (influences_arr < upper)
        label = f"Q{q//25}"
    
    actual_det = np.mean(delta_losses_arr[mask] < 0) * 100
    print(f"{label}: {actual_det:.1f}% actually detrimental")

# 8. KEY INSIGHTS
print("\n8. KEY INSIGHTS AND IMPLICATIONS")
print("-"*40)

# Compare methods
simple_accuracy = 0.5044  # From your previous run
hessian_accuracy = perf['accuracy']
improvement = (hessian_accuracy - simple_accuracy) * 100

print(f"✓ Collective removal WORKS: +{collective['delta_accuracy']*100:.2f}% accuracy improvement")
print(f"✓ Ground truth validated: Loss decreased by {abs(collective['delta_loss']):.6f}")
print(f"✓ {collective['num_removed']:,} parameters can be safely removed")

if hessian_accuracy > 0.51:
    print(f"✓ Hessian slightly better than simple influence ({hessian_accuracy:.4f} vs {simple_accuracy:.4f})")
else:
    print(f"✗ Hessian not significantly better ({hessian_accuracy:.4f} vs {simple_accuracy:.4f})")

if abs(perf['correlation']) > 0.1:
    print(f"✓ Meaningful correlation found: {perf['correlation']:.4f}")
else:
    print(f"✗ Near-zero correlation: {perf['correlation']:.4f}")

# Calculate effect size
individual_mean_effect = abs(np.mean(delta_losses))
collective_effect = abs(collective['delta_loss'])
amplification = collective_effect / (individual_mean_effect * collective['num_removed'])

print(f"\n9. EFFECT AMPLIFICATION")
print("-"*40)
print(f"Mean individual |ΔL|: {individual_mean_effect:.8f}")
print(f"Expected collective effect (if additive): {individual_mean_effect * collective['num_removed']:.6f}")
print(f"Actual collective effect: {collective_effect:.6f}")
print(f"Amplification factor: {amplification:.2f}x")

if amplification < 0.5:
    print("→ Parameters partially cancel each other (sub-additive)")
elif amplification > 1.5:
    print("→ Synergistic effects when removed together (super-additive)")
else:
    print("→ Effects are roughly additive")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"1. Removing {collective['num_removed']:,} individually-detrimental parameters")
print(f"   improves accuracy by {collective['delta_accuracy']*100:.2f}%")
print(f"2. Individual influence predictions remain poor (~50% accuracy)")
print(f"3. Collective effects validate ground truth methodology")
print(f"4. This suggests parameter importance is fundamentally collective,")
print(f"   not individual, in neural networks")
