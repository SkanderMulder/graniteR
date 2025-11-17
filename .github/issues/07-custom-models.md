# Custom Model Support and Documentation

## Overview
Expand documentation and tooling for using custom/fine-tuned models beyond the default Granite models.

## Current State
- Package works with "any Hugging Face transformer encoder model" (README)
- Users can specify `model_name` parameter
- Limited documentation on model selection and compatibility

## Gaps to Address

### 1. Model Compatibility Guide
Document which model types work and which don't:

**Compatible** (Encoder-only):
- BERT variants (bert-base, distilbert, roberta)
- Granite Embedding models
- Sentence-BERT models
- DeBERTa
- ELECTRA
- ALBERT

**Incompatible**:
- Decoder-only (GPT-2, GPT-3)
- Encoder-decoder (T5, BART)
- Vision models (CLIP, ViT)

### 2. Model Selection Vignette
Create `vignettes/model-selection.Rmd`:

**Topics**:
- How to choose a model for your task
- Performance vs size tradeoffs
- Domain-specific models (legal, medical, code)
- Multilingual models
- Speed benchmarks for common models

**Example Content**:
```r
# General purpose (English)
classifier(model_name = "ibm-granite/granite-embedding-r2")

# Multilingual
classifier(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Domain-specific: Legal
classifier(model_name = "nlpaueb/legal-bert-base-uncased")

# Small and fast
classifier(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Large and accurate
classifier(model_name = "microsoft/deberta-v3-large")
```

### 3. Custom Model Tutorial
Vignette: `vignettes/custom-models.Rmd`

**Scenario 1: Using Pre-trained Models from Hub**
```r
# Browse Hugging Face Hub
# https://huggingface.co/models?pipeline_tag=feature-extraction

# Load specific model
clf <- classifier(
  num_labels = 3,
  model_name = "sentence-transformers/all-mpnet-base-v2"
)
```

**Scenario 2: Using Your Own Fine-tuned Model**
```r
# After fine-tuning a model with transformers in Python
# Upload to HF Hub or use local path

clf <- classifier(
  num_labels = 2,
  model_name = "/path/to/my-fine-tuned-model"  # Local path
)

# Or from private HF repo
clf <- classifier(
  num_labels = 2,
  model_name = "myusername/my-private-model",
  token = Sys.getenv("HF_TOKEN")  # Hugging Face token
)
```

**Scenario 3: Downloading and Caching Models**
```r
# Pre-download models to avoid runtime delays
download_model("ibm-granite/granite-embedding-r2", 
               cache_dir = "~/.cache/huggingface")

# Use offline mode
clf <- classifier(
  model_name = "ibm-granite/granite-embedding-r2",
  offline = TRUE  # Use only cached models
)
```

### 4. Model Utilities
Add helper functions to `R/model.R`:

```r
#' List available models from Hugging Face Hub
#' @export
list_available_models <- function(
  task = "feature-extraction",
  language = NULL,
  sort = "downloads"
) {
  # Query HF API
  # Return tibble of models with metadata
}

#' Get model info
#' @export
model_info <- function(model_name) {
  # Returns: size, architecture, languages, license
}

#' Validate model compatibility
#' @export
check_model_compatibility <- function(model_name) {
  # Tests if model is encoder-only
  # Returns helpful error if incompatible
}

#' Download model to cache
#' @export
download_model <- function(model_name, cache_dir = NULL) {
  # Pre-downloads model and tokenizer
}
```

### 5. Model Comparison Tool
```r
# Compare multiple models on your data
compare_models <- function(
  data,
  text_col,
  label_col,
  models = c(
    "ibm-granite/granite-embedding-r2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "bert-base-uncased"
  ),
  metrics = c("accuracy", "f1", "speed")
) {
  # Train each model
  # Return comparison tibble + plots
}
```

### 6. Documentation Improvements

**README Updates**:
- Add "Supported Models" section
- Link to model selection guide
- Examples with 3-4 different models

**Function Documentation**:
- Expand `model_name` parameter docs
- Add examples with different models
- Link to model selection vignette

**pkgdown Site**:
- Add "Model Gallery" page
- Showcase 10-15 recommended models with use cases
- Performance comparison table

### 7. Troubleshooting Guide
Common issues:

**Issue**: Model download fails
- Solution: Check internet connection, HF Hub status
- Alternative: Download manually and use local path

**Issue**: Model incompatible (decoder-only)
- Solution: Clear error message with suggestions
- Provide list of compatible alternatives

**Issue**: Model too large for memory
- Solution: Recommend smaller alternative
- Document memory requirements for common models

**Issue**: Slow inference
- Solution: Benchmark table in docs
- Suggest quantization or distilled models

## Implementation Plan

### Phase 1: Core Documentation
- [ ] Create `vignettes/model-selection.Rmd`
- [ ] Create `vignettes/custom-models.Rmd`
- [ ] Update README with model examples
- [ ] Add model comparison table

### Phase 2: Helper Functions
- [ ] Implement `list_available_models()`
- [ ] Implement `model_info()`
- [ ] Implement `check_model_compatibility()`
- [ ] Implement `download_model()`

### Phase 3: Advanced Features
- [ ] Model comparison tool
- [ ] Benchmarking utilities
- [ ] pkgdown model gallery

### Phase 4: Testing
- [ ] Test with 10+ different models
- [ ] Document edge cases
- [ ] Add integration tests

## Examples to Include

### Domain-Specific Models
```r
# Legal
legal_clf <- classifier(
  model_name = "nlpaueb/legal-bert-base-uncased",
  num_labels = 5
)

# Scientific/Biomedical
bio_clf <- classifier(
  model_name = "dmis-lab/biobert-base-cased-v1.1",
  num_labels = 3
)

# Code/Programming
code_clf <- classifier(
  model_name = "microsoft/codebert-base",
  num_labels = 4
)
```

### Multilingual
```r
# 50+ languages
multilingual_clf <- classifier(
  model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
  num_labels = 2
)
```

### Size Variants
```r
# Tiny (fast, mobile)
tiny_clf <- classifier(
  model_name = "sentence-transformers/all-MiniLM-L6-v2",  # 22M params
  num_labels = 2
)

# Base (balanced)
base_clf <- classifier(
  model_name = "ibm-granite/granite-embedding-r2",  # 149M params
  num_labels = 2
)

# Large (accurate)
large_clf <- classifier(
  model_name = "microsoft/deberta-v3-large",  # 304M params
  num_labels = 2
)
```

## Success Criteria
- Users can easily find and use appropriate models
- Clear guidance on model selection
- Documentation covers 90% of use cases
- Helpful errors for incompatible models
- Examples with 10+ different model types

## Benefits
- Unlock full potential of Hugging Face ecosystem
- Better model-task matching
- Easier experimentation
- Lower barrier to entry

## Related Issues
- Edge case handling (#[TBD])
- Benchmarking suite (#[TBD])
- Documentation improvements

## Priority
Medium-High - Important for adoption and differentiation

## Timeline
- Phase 1 (Docs): 1 week
- Phase 2 (Utilities): 1 week
- Phase 3 (Advanced): 1 week
