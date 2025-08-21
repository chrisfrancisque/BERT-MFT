# MFT (Mask Fine-Tuning) Implementation Results

## Summary
Implementation of "Boosting Large Language Models with Mask Fine-Tuning" paper on BERT with SST-2 dataset.

## Key Findings

### Original Hypothesis
Higher accuracy models would have more maskable/redundant parameters.

### Actual Results

#### Gradient-Based Approach (Failed)
- Identified parameters "moving toward zero" as detrimental
- All models collapsed to ~49% accuracy
- Issue: Parameters moving toward zero ≠ unimportant parameters

#### Learnable Mask Approach (Paper's Method)
Using learnable scores to identify which parameters to mask:

| Checkpoint | Original Acc | After MFT | Change |
|------------|-------------|-----------|---------|
| Baseline (50%) | 49.54% | 53.78% | **+4.24%** |
| FFT 75% | 77.98% | 65.71% | -12.27% |
| FFT 90% | 90.37% | 87.04% | -3.33% |

### Key Insights

1. **MFT works better on less-optimized models** - The baseline improved while highly-trained models degraded
2. **Masking ratio is critical** - 90% masking caused catastrophic failure; 10% masking worked
3. **Layer selection matters** - Only masked layers 4-7 (middle layers)
4. **Critical components must be preserved**:
   - All embeddings (token, position, token_type)
   - LayerNorm parameters
   - Classifier head
   - Pooler layer

### Technical Details

- **Parameters masked**: 10% of maskable parameters (~2.8M out of 28.3M)
- **Target layers**: 4, 5, 6, 7 (middle transformer layers)
- **Training**: 2 epochs with learnable mask scores
- **Learning rate**: 5e-4 for mask scores
- **Regularization**: L1 penalty on scores (0.0001)

## Implementation Notes

The successful implementation (`mft_fixed.py`) uses:
1. Learnable scores for each maskable parameter
2. Temporary masking during forward pass
3. Gradient flow through mask scores only (model weights frozen)
4. Top-k selection for masking (keep 90%, mask 10%)

## Conclusion

MFT shows promise for improving less-optimized models but can degrade highly-optimized ones. The technique requires careful tuning of:
- Masking ratio
- Layer selection
- Which parameters to exclude from masking

The hypothesis was partially supported: higher accuracy models do have different parameter importance distributions, but these parameters are not safely maskable - they're critical for performance.
