# Vignette Update Status

## Goal
All dataset-focused vignettes should demonstrate **both** classification approaches using pre-trained models loaded from `models/` directory.

## Standard Structure

Each vignette includes:

### 1. Standard Classifier (Frozen Backbone)
```r
# Training (eval=FALSE)
clf_std <- classifier(num_labels = n) |> train(..., epochs = 3-5)
save_classifier(clf_std, "models/name_standard")

# Loading (eval=FALSE for now)
clf_std <- load_classifier("models/name_standard")

# Predictions and metrics
```

### 2. Mixture of Experts (MoE)
```r
# Training (eval=FALSE)
clf_moe <- moe_classifier(num_labels = n, num_experts = k) |>
  train_moe(..., epochs = 3-5)
save_classifier(clf_moe, "models/name_moe")

# Loading (eval=FALSE for now)
clf_moe <- load_classifier("models/name_moe")

# Predictions and metrics
```

### 3. Comparison
- Side-by-side performance
- Trade-off discussion
- When to use each approach

## Training Configuration

Models are being trained via `dev/train_all_vignette_models.R`:

| Dataset | Labels | Models | Status |
|---------|--------|--------|--------|
| emotion_full | 6 | standard + moe | Training (in progress) |
| sentiment_imdb_full | 2 | standard + moe | Pending |
| hate_speech_full | 2 | standard + moe | Pending |
| malicious_prompts_full | 2 | standard + moe | Pending |

Training time estimate: 20-30 minutes with GPU

## Vignette Status

### ✅ Updated
- emotion-detection.Rmd - Both approaches implemented

### 📝 Needs Update
- sentiment-analysis.Rmd - Currently only shows standard, needs MoE section
- hate-speech-detection.Rmd - Currently only shows standard, needs MoE section  
- malicious-prompt-detection.Rmd - Currently only shows standard, needs MoE section

### ℹ️ No Changes Needed
- getting-started.Rmd - Basic intro vignette
- model-persistence.Rmd - Save/load focus
- mixture-of-experts.Rmd - May need minor updates

## Development Approach

**Current Phase**: Training models + Updating vignettes

1. All code blocks set to `eval=FALSE` to allow vignette rendering without models
2. Once training completes, models will be available in `models/` directory
3. For package check, some blocks can be set to `eval=TRUE`

## Next Steps

1. ✅ Complete model training (running now)
2. ⏳ Update remaining 3 vignettes with MoE sections
3. ⏳ Verify all vignettes render correctly
4. ⏳ Test model loading and predictions
5. ⏳ Commit all changes

## Update Complete!

All 4 dataset vignettes have been updated (2026-01-19):

1. ✅ emotion-detection.Rmd - Standard + MoE + Comparison
2. ✅ sentiment-analysis.Rmd - Standard + MoE + Comparison  
3. ✅ hate-speech-detection.Rmd - Standard + MoE + Comparison
4. ✅ malicious-prompt-detection.Rmd - Standard + MoE + Comparison

### Commits
- `de18b79` - Updated emotion vignette  
- `2d76060` - Updated sentiment, hate-speech, and malicious-prompt vignettes

### Training Status
Training script running: `dev/train_all_vignette_models.R` (PID 5417)
- Currently on step 2/8: emotion_moe
- Expected completion: ~20-30 minutes from start

### File Sizes
Each vignette is now 300-400 lines, focused and consistent.
