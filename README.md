# graniteR <img src="man/figures/logo.png" align="right" height="139" />

R interface to IBM Granite embedding models via Python's transformers library.

## Inspiration

This package is inspired by the [Granite Embedding R2 Models](https://arxiv.org/html/2508.21085v1) paper from IBM Research AI. The paper introduces a family of high-performance encoder-based embedding models designed for enterprise-scale dense retrieval, featuring:

- Extended 8,192-token context length (16x expansion)
- State-of-the-art performance across diverse retrieval domains
- 19-44% speed advantages over leading competitors
- Models ranging from 47M to 149M parameters
- Apache 2.0 licensing for enterprise use

The Granite models use ModernBERT architecture with Flash Attention optimizations and are trained using a multi-stage pipeline including retrieval-oriented pretraining, contrastive finetuning, and knowledge distillation.

## Installation

### Step 1: Install the R Package

```r
# Install from GitHub
devtools::install_github("skandermulder/graniteR")
```

### Step 2: Install Python Dependencies

graniteR requires Python dependencies (transformers, torch, datasets, numpy) to function. Choose one of the two methods below:

#### Option A: Fast Setup with UV (Recommended)

UV is a modern Python package manager that is 10-100x faster than pip. This is the recommended approach for faster installation.

**1. Run the automated setup script:**

```bash
cd path/to/graniteR
./setup_python.sh
```

The script will:
- Install UV automatically if not present
- Create a virtual environment at `.venv`
- Install all Python dependencies rapidly

**2. Configure R to use the virtual environment:**

Add this to your `.Rprofile` or at the start of your R script:

```r
Sys.setenv(RETICULATE_PYTHON = ".venv/bin/python")
```

**3. Load the package:**

```r
library(graniteR)
```

#### Option B: Traditional Installation

If you prefer using pip or conda:

```r
library(graniteR)
install_granite()  # Uses pip (slower)
```

Or use the UV function from within R:

```r
library(graniteR)
install_granite_uv()  # Uses UV (faster)
```

### Verifying Installation

```r
library(graniteR)

# Test with a simple embedding
tibble(text = "Hello world") |>
  granite_embed(text)
```

### Why UV?

Traditional pip installation of PyTorch and transformers can take **10-20 minutes** due to large package sizes and dependency resolution. UV solves this by:

- **Parallel downloads**: Downloads packages concurrently
- **Fast dependency resolution**: Uses Rust-based resolver
- **Efficient caching**: Reuses downloaded packages
- **Result**: Same installation in **1-2 minutes**

## Usage

### Generate Embeddings

```r
library(graniteR)
library(dplyr)

data <- tibble(
  id = 1:3,
  text = c(
    "This is a positive sentence",
    "This is a negative sentence",
    "This is a neutral sentence"
  )
)

embeddings <- data |>
  granite_embed(text)

head(embeddings)
```

### Text Classification

```r
train_data <- tibble(
  text = c(
    "I love this product",
    "This is terrible",
    "Great experience",
    "Very disappointing"
  ),
  label = c(1, 0, 1, 0)
)

classifier <- granite_classifier(num_labels = 2) |>
  granite_train(
    train_data,
    text,
    label,
    epochs = 3,
    batch_size = 2
  )

new_data <- tibble(
  text = c("Amazing product", "Not good")
)

predictions <- granite_predict(classifier, new_data, text)
```

## Features

- Pipe-friendly interface following tidyverse conventions
- Support for sentence embeddings with Granite-R2
- Text classification with fine-tuning
- GPU acceleration support
- Minimal dependencies

## Model

The package uses IBM's Granite Embedding English R2 model by default:
- Model: `ibm-granite/granite-embedding-english-r2`
- Size: 149M parameters
- Embedding dimension: 768
- Max sequence length: 512 tokens

## Troubleshooting

### Python environment not recognized

Verify your Python path is set correctly:

```r
Sys.getenv("RETICULATE_PYTHON")
```

Should show the path to your virtual environment's Python. If not set:

```r
Sys.setenv(RETICULATE_PYTHON = "/absolute/path/to/graniteR/.venv/bin/python")
```

### UV installation issues

If the setup script can't find UV after installation, add UV to your PATH:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Add this line to your `~/.bashrc` or `~/.zshrc` for permanent effect.

### More Help

See [SETUP.md](SETUP.md) for detailed troubleshooting and configuration options.

## License

MIT
