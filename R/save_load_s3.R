#' Save granite classifiers with proper PyTorch weight handling
#'
#' Saves granite classifiers by separately storing PyTorch weights and R configuration.
#' This ensures models can be properly loaded in new R sessions despite reticulate limitations.
#'
#' @param object Trained classifier object
#' @param file File path (without extension)
#' @param ... Additional arguments (unused)
#'
#' @return Invisibly returns file path
#'
#' @details
#' **Usage**:
#' \preformatted{
#' # Save - creates two files
#' save_classifier(clf, "models/my_model")
#' # Creates: my_model_weights.pt + my_model_config.rds
#'
#' # Load in new session
#' clf <- load_classifier("models/my_model")
#' }
#'
#' **What gets saved**:
#' \itemize{
#'   \item Weights file: PyTorch state_dict with all trained parameters
#'   \item Config file: Model architecture, labels, device, freeze_backbone
#' }
#'
#' @examples
#' \dontrun{
#' # Train and save
#' clf <- classifier(6) |> train(data, text, label, epochs = 3)
#' save_classifier(clf, "models/emotion_v1")
#'
#' # Load and use
#' clf <- load_classifier("models/emotion_v1")
#' predict(clf, test_data, text)
#' }
#'
#' @name save-load-classifiers
#' @export
save_classifier <- function(object, file = "", ...) {
  save_classifier_impl(object, file)
}

#' Internal: Save classifier implementation
#' @keywords internal
save_classifier_impl <- function(classifier, file) {
  # Remove .rds extension if present
  file <- sub("\\.rds$", "", file, ignore.case = TRUE)

  # Get path components
  path <- dirname(file)
  if (path != "." && !dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }

  # Define file paths
  weights_file <- paste0(file, "_weights.pt")
  config_file <- paste0(file, "_config.rds")

  # Extract config
  config <- list(
    model_type = if ("moe_classifier" %in% class(classifier)) "moe" else "standard",
    num_labels = classifier$num_labels,
    model_name = classifier$tokenizer$model_name,
    freeze_backbone = classifier$freeze_backbone %||% TRUE,
    device = classifier$device,
    is_trained = classifier$is_trained
  )

  if (config$model_type == "moe") {
    config$num_experts <- classifier$num_experts
  }

  # Save PyTorch weights
  torch <- reticulate::import("torch")
  torch$save(classifier$model$model$state_dict(), weights_file)

  # Save config with base R saveRDS
  base::saveRDS(config, config_file)

  cli::cli_alert_success("Saved classifier")
  cli::cli_alert_info("Weights: {basename(weights_file)}")
  cli::cli_alert_info("Config: {basename(config_file)}")
  cli::cli_alert_info("To load: clf <- load_classifier('{file}')")

  invisible(file)
}

#' Load a saved granite classifier
#'
#' Custom readRDS behavior for granite classifiers. Point it at the *_config.rds file
#' and it will automatically load the weights and reconstruct the model.
#'
#' @param file Path to the *_config.rds file
#' @param device Device to load model on (NULL = use saved device)
#'
#' @return A trained classifier ready for predictions
#'
#' @details
#' **Usage**:
#' \preformatted{
#' # Save
#' saveRDS(clf, "models/my_model")
#'
#' # Load - point to the config file
#' clf <- readRDS("models/my_model_config.rds")
#'
#' # Or use the helper
#' clf <- load_classifier("models/my_model")
#' }
#'
#' @export
load_classifier <- function(file, device = NULL) {
  # Handle both "model" and "model_config.rds" inputs
  file <- sub("_config\\.rds$", "", file, ignore.case = TRUE)
  file <- sub("\\.rds$", "", file, ignore.case = TRUE)

  config_file <- paste0(file, "_config.rds")
  weights_file <- paste0(file, "_weights.pt")

  if (!file.exists(config_file)) {
    stop("Config file not found: ", config_file)
  }
  if (!file.exists(weights_file)) {
    stop("Weights file not found: ", weights_file)
  }

  # Load config
  config <- base::readRDS(config_file)

  if (is.null(device)) {
    device <- config$device %||% "cpu"
  }

  cli::cli_alert_info("Loading {config$model_type} classifier")
  cli::cli_alert_info("Labels: {config$num_labels}, Device: {device}")

  # Reconstruct model
  if (config$model_type == "moe") {
    clf <- moe_classifier(
      num_labels = config$num_labels,
      num_experts = config$num_experts,
      model_name = config$model_name,
      device = device,
      freeze_backbone = config$freeze_backbone
    )
  } else {
    clf <- classifier(
      num_labels = config$num_labels,
      model_name = config$model_name,
      device = device,
      freeze_backbone = config$freeze_backbone
    )
  }

  # Load weights
  torch <- reticulate::import("torch")
  state_dict <- torch$load(weights_file, map_location = device)
  clf$model$model$load_state_dict(state_dict)

  clf$is_trained <- TRUE

  cli::cli_alert_success("Model loaded and ready for predictions")

  clf
}

# Helper for NULL coalescing
`%||%` <- function(x, y) if (is.null(x)) y else x
