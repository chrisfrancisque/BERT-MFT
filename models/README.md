# Model Checkpoints for MFT Analysis

## Baseline Model (48.86% accuracy)
- Location: `baseline/checkpoint_50pct_baseline/`
- Description: Warmed BERT model with classifier trained for 1 epoch
- Used as starting point for both FFT and LoRA

## Full Fine-Tuning (FFT) Checkpoints
- Location: `fft_checkpoints/`
- Checkpoints saved at: 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 88%, 90% accuracy
- All 109.5M parameters updated during training
- Training samples: 30,000 from SST-2
- Each checkpoint ~418MB

## LoRA Checkpoints  
- Location: `lora_checkpoints/`
- 3 epochs of training from same baseline
- Only ~40K parameters updated (0.037%)
- Target modules: query and value attention matrices
- Rank: 8, Alpha: 16

## MFT Analysis Plan
Run MFT on each checkpoint to measure:
1. Percentage of parameters that can be masked at different accuracy levels
2. Test hypothesis: Higher accuracy → More maskable parameters
3. Compare sparsity patterns between FFT and LoRA models
4. Determine if LoRA models have different redundancy patterns than FFT

## Expected Results
- Lower accuracy models (50-60%): ~5-15% maskable
- Medium accuracy models (70-80%): ~20-35% maskable  
- High accuracy models (85-90%): ~40-60% maskable
- LoRA models: Potentially different pattern due to adapter architecture
