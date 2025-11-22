#' @keywords internal
#' @importFrom stats median predict sd
#' @importFrom utils tail
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end
NULL

#' graniteR: Interface to IBM Granite Embedding Models
#'
#' @description
#' graniteR provides a pipe-friendly interface to IBM's Granite embedding models
#' and other Hugging Face transformer encoder models via Python's transformers
#' library. The package enables text embeddings, classification, and fine-tuning
#' workflows following tidyverse conventions.
#'
#' @details
#' ## Key Features
#'
#' - **Transfer learning**: Frozen pretrained models with trainable classification
#'   heads for efficient training
#' - **Local execution**: All inference runs on-device, ensuring data privacy
#' - **Multi-class support**: Binary and n-class classification with softmax output
#' - **GPU acceleration**: Automatic CUDA detection with CPU fallback
#' - **Fast setup**: UV package manager for Python dependencies (1-2 min vs 10-20 min)
#'
#' ## Main Functions
#'
#' - [embed()]: Generate sentence embeddings from text
#' - [classifier()]: Create a classification model
#' - [train()]: Train a classifier on labeled data
#' - [predict()]: Make predictions with a trained classifier
#' - [install_pyenv()]: Install Python dependencies
#' - [granite_check_system()]: Check system configuration
#'
#' ## Python Integration
#'
#' graniteR uses [reticulate] to interface with Python's transformers and torch
#' libraries. Python dependencies can be installed using [install_pyenv()], which
#' uses the UV package manager for fast installation.
#'
#' @seealso
#' Useful links:
#' - \url{https://github.com/skandermulder/graniteR}
#' - Report bugs at \url{https://github.com/skandermulder/graniteR/issues}
#'
#' @examples
#' \dontrun{
#' library(graniteR)
#'
#' # Install Python dependencies
#' install_pyenv()
#'
#' # Generate embeddings
#' data <- tibble::tibble(text = c("Hello world", "Goodbye world"))
#' embeddings <- embed(data, text)
#'
#' # Train a binary classifier
#' train_data <- tibble::tibble(
#'   text = c("I love this", "terrible", "great", "poor"),
#'   label = c(1, 0, 1, 0)
#' )
#' clf <- classifier(num_labels = 2) |>
#'   train(train_data, text_col = text, label_col = label, epochs = 3)
#'
#' # Make predictions
#' new_data <- tibble::tibble(text = c("excellent", "bad"))
#' predictions <- predict(clf, new_data, text_col = text)
#' }
#'
#' @name graniteR-package
#' @aliases graniteR
#' @docType package
NULL
