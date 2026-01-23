# graniteR 0.1.1 (Development)

## Breaking Changes

- Moved AutoML functionality (`auto_classify()`, ensemble methods, meta-feature extraction) to `dev/` directory
  - These features are experimental and will be part of a future release
  - Users should use `classifier()` or `moe_classifier()` for current classification tasks

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
