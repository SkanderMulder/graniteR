# graniteR (development)

## Documentation

- Enhanced package-level documentation with comprehensive description, examples, and function references

## CI/CD

- Added test coverage reporting workflow with Codecov integration
- Added coverage badge to README

## Package Quality

- Added minimum version requirement for processx dependency (>= 3.5.0)
- Fixed LICENSE.md placeholder text with proper copyright information
- Added cross-reference links (@seealso) to save/load functions for better documentation navigation

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
