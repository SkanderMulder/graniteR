# Publishing Models to GitHub Releases

This guide explains how to publish the pre-trained vignette models to GitHub releases so users can download them.

## Prerequisites

1. Install GitHub CLI: `gh` (if not already installed)
2. Authenticate: `gh auth login`
3. Trained models in `models/` directory

## Steps

### 1. Create a Release

```bash
gh release create v0.1.1 \
  --title "graniteR v0.1.1 - Pre-trained Models" \
  --notes "Pre-trained models for package vignettes (600MB total)"
```

### 2. Upload Model Files

```bash
# Navigate to package directory
cd /home/skandeer/dev/graniteR

# Upload all model files to the release
gh release upload v0.1.1 \
  models/emotion_standard_config.rds \
  models/emotion_standard_weights.pt \
  models/emotion_moe_config.rds \
  models/emotion_moe_weights.pt \
  models/sentiment_standard_config.rds \
  models/sentiment_standard_weights.pt \
  models/sentiment_moe_config.rds \
  models/sentiment_moe_weights.pt \
  models/hate_speech_standard_config.rds \
  models/hate_speech_standard_weights.pt
```

### 3. Verify Upload

```bash
gh release view v0.1.1
```

## Users Can Now Download

```r
library(graniteR)

# Download a single model
download_model("emotion_standard")

# Or download all vignette models
download_vignette_models()
```

## Model URLs

After upload, models will be available at:
```
https://github.com/SkanderMulder/graniteR/releases/download/v0.1.1/emotion_standard_config.rds
https://github.com/SkanderMulder/graniteR/releases/download/v0.1.1/emotion_standard_weights.pt
...
```

## Alternative: Single Command

```bash
cd /home/skandeer/dev/graniteR/models
gh release create v0.1.1 \
  --title "graniteR v0.1.1 - Pre-trained Models" \
  --notes "Pre-trained models for package vignettes" \
  *.rds *.pt
```

## Updating Models

To update models in an existing release:

```bash
# Delete old files
gh release delete-asset v0.1.1 emotion_standard_weights.pt

# Upload new version
gh release upload v0.1.1 models/emotion_standard_weights.pt
```

## Model Sizes

```bash
ls -lh models/
```

Total: ~600MB
- emotion_moe: ~612MB
- emotion_standard: ~1.2MB
- hate_speech_standard: ~1.2MB
- sentiment_moe: ~1.3KB
- sentiment_standard: ~1.2MB
