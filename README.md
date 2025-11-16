# graniteR <img src="man/figures/logo.png" align="right" height="139" />

<!-- badges: start -->
[![R-CMD-check](https://github.com/skandermulder/graniteR/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/skandermulder/graniteR/actions/workflows/R-CMD-check.yaml)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

R interface to [IBM's Granite Embedding R2 model](https://arxiv.org/html/2508.21085v1) (149M parameters) for text embeddings and multi-class classification. Based on ModernBERT with Flash Attention, achieving 19-44% inference speed improvements over comparable models.

**Privacy**: All models execute locally. No data transmission to external servers.

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
  granite_embed(text)  # 768-dimensional dense vectors
```

**Binary Classification:**
```r
train <- tibble(
  text = c("I love this", "terrible", "great", "poor"),
  label = c(1, 0, 1, 0)
)

classifier <- granite_classifier(num_labels = 2) |>
  granite_train(train, text_col = text, label_col = label, epochs = 3)

granite_predict(classifier, tibble(text = c("excellent", "bad")), text_col = text)
```

**Multi-Class Classification:**
```r
train <- tibble(
  text = c("urgent issue", "routine request", "critical failure", "minor bug"),
  priority = c("high", "low", "critical", "medium")
)

classifier <- granite_classifier(num_labels = 4) |>
  granite_train(train, text_col = text, label_col = priority, epochs = 5)

# Returns class predictions or probability distributions
granite_predict(classifier, new_data, text_col = text, type = "class")
granite_predict(classifier, new_data, text_col = text, type = "prob")
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
