# graniteR 0.1.1 (Development)

## Bug Fixes

- Fixed interrupt handling to properly stop Python/PyTorch training processes when user interrupts execution (Ctrl+C)
  - Added periodic interrupt checks during training loops (every 5 batches)
  - Added GPU memory cleanup on interrupt to prevent resource leaks
  - Training interrupts now properly propagate and stop all underlying Python processes
  - Fixed issue in `train()`, `train_moe()`, and `auto_classify()` where interrupts would be caught but Python processes continued running

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
