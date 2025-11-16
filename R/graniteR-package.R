#' graniteR: Interface to IBM Granite Embedding Models
#'
#' Provides a pipe-friendly interface to IBM's Granite embedding models
#' via Python's transformers library. Key features include:
#' \itemize{
#'   \item Generate sentence embeddings with \code{granite_embed()}
#'   \item Train text classifiers with \code{granite_classifier()}
#'   \item Fine-tune models with \code{granite_train()}
#'   \item Make predictions with \code{granite_predict()}
#' }
#'
#' @section Installation:
#' The package requires Python dependencies. Use \code{install_granite()}
#' or \code{install_granite_uv()} for faster setup.
#'
#' @section Getting Started:
#' See the "Getting Started" vignette: \code{vignette("getting-started", package = "graniteR")}
#'
#' @importFrom tibble tibble
#' @importFrom rlang .data
#' @docType package
#' @name graniteR-package
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end
NULL