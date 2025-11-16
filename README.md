# graniteR <img src="man/figures/logo.png" align="right" height="139" />

<!-- badges: start -->
[![R-CMD-check](https://github.com/skandermulder/graniteR/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/skandermulder/graniteR/actions/workflows/R-CMD-check.yaml)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

R interface for text embeddings and classification using transformer encoder models. Designed as a homage to [IBM's Granite Embedding R2](https://arxiv.org/html/2508.21085v1) (149M parameters, ModernBERT with Flash Attention), but compatible with any Hugging Face transformer encoder model.

**Privacy**: All models execute locally. No data transmission to external servers.

> **Note**: While optimized for Granite models, this package works with other encoder models (BERT, RoBERTa, DistilBERT, etc.) by specifying `model_name` in function calls.

## Installation

```r
# Install package
devtools::install_github("skandermulder/graniteR")

# Install Python dependencies (UV - completes in 1-2 minutes)
library(graniteR)
install_granite_uv()
```

## Quick Start

**Embeddings:**
```r
library(graniteR)
library(dplyr)

tibble(text = c("positive", "negative")) |>
  embed(text)  # 768-dimensional dense vectors
```

**Binary Classification:**
```r
train <- tibble(
  text = c("I love this", "terrible", "great", "poor"),
  label = c(1, 0, 1, 0)
)

clf <- classifier(num_labels = 2) |>
  train(train, text_col = text, label_col = label, epochs = 3)

predict(clf, tibble(text = c("excellent", "bad")), text_col = text)
```

**Multi-Class Classification:**
```r
train <- tibble(
  text = c("urgent issue", "routine request", "critical failure", "minor bug"),
  priority = c("high", "low", "critical", "medium")
)

clf <- classifier(num_labels = 4) |>
  train(train, text_col = text, label_col = priority, epochs = 5)

# Returns class predictions or probability distributions
predict(clf, new_data, text_col = text, type = "class")
predict(clf, new_data, text_col = text, type = "prob")
```

## Features

- **Local execution**: All inference runs on-device, ensuring data privacy
- **Multi-class support**: Binary and n-class classification with softmax output
- **Fast dependency installation**: UV package manager (1-2 min vs 10-20 min with pip)
- **GPU acceleration**: Automatic CUDA detection with CPU fallback
- **Training monitoring**: Real-time loss and validation accuracy tracking
- **Flexible prediction**: Class labels or probability distributions

## Documentation

- `vignette("getting-started")` - Installation and basic usage
- `vignette("malicious-prompt-detection")` - Binary classification example
- `vignette("technical-approaches")` - Embeddings vs fine-tuning comparison
- `examples/multiclass_classification.R` - Multi-class classification workflow
- `granite_check_system()` - System diagnostics and setup verification

## License

MIT © 2024
