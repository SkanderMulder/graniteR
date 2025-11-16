# Quick Setup Guide for graniteR

This guide provides step-by-step instructions for setting up graniteR with fast Python dependency installation.

## Quick Start (Recommended)

### 1. Install R Package

```r
devtools::install_github("skandermulder/graniteR")
```

### 2. Install Python Dependencies with UV

From your terminal:

```bash
cd path/to/graniteR
./setup_python.sh
```

The script automatically:
- Installs UV if not present
- Creates a virtual environment at `.venv`
- Installs all Python dependencies in 1-2 minutes

### 3. Configure R

Add to your `.Rprofile` or at the start of your R scripts:

```r
Sys.setenv(RETICULATE_PYTHON = ".venv/bin/python")
```

### 4. Test Installation

```r
library(graniteR)
library(tibble)

tibble(text = "Hello world") |> granite_embed(text)
```

## Alternative Installation Methods

### Method 1: UV from R

```r
library(graniteR)
install_granite_uv()
```

### Method 2: Traditional pip (slower)

```r
library(graniteR)
install_granite()
```

## Why UV?

| Method | Time | Notes |
|--------|------|-------|
| pip | 10-20 min | Traditional, sequential downloads |
| UV | 1-2 min | Parallel downloads, fast resolver |

UV advantages:
- Parallel package downloads
- Rust-based dependency resolver
- Efficient caching
- 10-100x faster than pip

## Troubleshooting

### UV not found after installation

The setup script installs UV to `~/.cargo/bin`. Add to your shell profile:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Then restart your terminal or run `source ~/.bashrc` (or `~/.zshrc`).

### Python environment not recognized in R

Verify the path:

```r
Sys.getenv("RETICULATE_PYTHON")
```

Should show: `/path/to/graniteR/.venv/bin/python`

If not, set it explicitly:

```r
Sys.setenv(RETICULATE_PYTHON = "/absolute/path/to/graniteR/.venv/bin/python")
```

### Import errors for transformers or torch

Verify dependencies are installed:

```bash
source .venv/bin/activate
python -c "import transformers; import torch; print('OK')"
```

If errors occur, reinstall:

```bash
uv pip install transformers torch datasets numpy
```

### GPU not detected

Check CUDA availability:

```r
library(graniteR)
granite_model(device = "cuda")  # Will error if CUDA not available
```

For CPU-only usage:

```r
granite_model(device = "cpu")  # Default
```

## Environment Variables

Useful environment variables for configuration:

```r
# Python path
Sys.setenv(RETICULATE_PYTHON = ".venv/bin/python")

# Force CPU (even if GPU available)
Sys.setenv(CUDA_VISIBLE_DEVICES = "")

# Specify GPU device
Sys.setenv(CUDA_VISIBLE_DEVICES = "0")
```

## Development Setup

For package development:

```bash
# Clone repository
git clone https://github.com/skandermulder/graniteR.git
cd graniteR

# Setup Python environment
./setup_python.sh

# Install R package in development mode
R -e "devtools::install()"
```

## Additional Resources

- [Getting Started Vignette](vignettes/getting-started.Rmd)
- [Granite R2 Paper](https://arxiv.org/html/2508.21085v1)
- [Hugging Face Models](https://huggingface.co/ibm-granite)
