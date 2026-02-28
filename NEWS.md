# graniteR 0.1.2

## New Features

- Added `trust_remote_code` parameter to `moe_classifier()` function
  - Enables MoE classifiers to use models with custom code like perplexity-ai/pplx-embed-v1-0.6b
  - Consistent with `classifier()` function interface
  - Updated emotion-detection vignette to demonstrate usage with perplexity model

# graniteR 0.1.1

## Breaking Changes

- Moved AutoML functionality (`auto_classify()`, ensemble methods, meta-feature extraction) to `dev/` directory
  - These features are experimental and will be part of a future release
  - Users should use `classifier()` or `moe_classifier()` for current classification tasks

## New Features

- Added support for custom Hugging Face models with `trust_remote_code` parameter
  - New parameter in `classifier()`, `moe_classifier()`, `granite_model()`, and `granite_tokenizer()`
  - Enables use of models like perplexity-ai/pplx-embed-v1-0.6b that contain custom code
  - Automatically falls back to custom classification head for embedding-only models
  - Custom wrapper (`EmbeddingModelForSequenceClassification`) adds classification capability to embedding models

- Added `find_model()` function to locate pre-trained models
  - Automatically searches in package installation directory (`inst/extdata/models/`)
  - Falls back to development directory and backward-compatible locations
  - Provides clear error messages with download instructions

- Added `get_models_dir()` function to get the appropriate models directory
  - Returns installed package location or development location
  - Automatically creates directory if it doesn't exist

## Improvements

- Improved error messages for models requiring `trust_remote_code`
  - Clear, actionable error message when attempting to use models with custom code
  - Shows exact command needed to fix the issue
  - Prevents confusing Python error messages from reaching users

- Improved model storage organization
  - Pre-trained models now stored in `inst/extdata/models/` following R package conventions
  - Makes vignettes portable and ensures models work in installed packages
  - Updated `download_model()` to use package-appropriate locations by default

- Updated all vignettes to use `find_model()` for model loading
  - Ensures examples work correctly after package installation
  - Better error messages when models are not found

- Updated `.Rbuildignore` to exclude legacy `models/` directory

## Bug Fixes

- Fixed Hugging Face connection errors after saving models
  - Models now load from local cache first to avoid unnecessary network calls
  - Added retry logic with exponential backoff for connection errors (max 5 retries)
  - Added resource cleanup after save operations to prevent stale connections
  - Fixes "Connection aborted" / "RemoteDisconnected" errors when creating models after saving

- Fixed critical MoE classifier training issues
  - Fixed undefined variable `moe_clf` references in `train_moe()` function (should be `classifier`)
  - Fixed missing `classification_loss` and `load_balance_loss` keys in `MoETextClassifier` return dictionary
  - Fixed missing `learning_rate` parameter pass-through in training helper functions
  - MoE training now works correctly for both binary and multi-class classification

- Fixed interrupt handling to properly stop Python/PyTorch training processes when user interrupts execution (Ctrl+C)
  - Added periodic interrupt checks during training loops (every 5 batches)
  - Added GPU memory cleanup on interrupt to prevent resource leaks
  - Training interrupts now properly propagate and stop all underlying Python processes

# graniteR 0.1.0

## Initial Release

- Added `granite_model()` for creating Granite models
- Added `granite_tokenizer()` for creating tokenizers
- Added `granite_embed()` for generating sentence embeddings
- Added `granite_classifier()` for classification tasks
- Added `granite_train()` for fine-tuning classifiers
- Added `granite_predict()` for making predictions
- Added `install_pyenv()` for Python dependency installation
- Support for CPU and GPU (CUDA) devices
- Pipe-friendly interface using native R pipe |>
- Follows tidyverse conventions
