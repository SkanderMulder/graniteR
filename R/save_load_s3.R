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

  # Determine model type and extract PyTorch model
  is_moe <- "moe_classifier" %in% class(classifier)
  pytorch_model <- if (is_moe) {
    classifier$model
  } else {
    classifier$model$model
  }

  # Extract device
  device <- if (is_moe) {
    classifier$device
  } else {
    classifier$model$device
  }

  # Extract config
  config <- list(
    model_type = if (is_moe) "moe" else "standard",
    num_labels = classifier$num_labels,
    model_name = classifier$tokenizer$model_name,
    freeze_backbone = classifier$freeze_backbone %||% TRUE,
    trust_remote_code = classifier$trust_remote_code %||% FALSE,
    device = device,
    is_trained = classifier$is_trained
  )

  if (config$model_type == "moe") {
    config$num_experts <- classifier$num_experts
  }

  # Save PyTorch weights with smart compression
  torch <- reticulate::import("torch")
  state_dict <- pytorch_model$state_dict()

  # For frozen models, only save trainable parameters (huge size reduction)
  # For unfrozen models, save everything in FP16
  if (config$freeze_backbone) {
    # Save only classifier head/experts (typically <5MB vs 570MB)
    state_dict_to_save <- reticulate::dict()
    saved_keys <- 0
    for (key in names(state_dict)) {
      # Save only head/classifier/expert layers, skip backbone
      # Pattern matches: head., classifier., classification_head., experts., gating.
      if (grepl("(head\\.|classifier\\.|classification_head\\.|experts\\.|gating\\.)", key)) {
        tensor <- state_dict[[key]]
        # Still use FP16 for additional compression
        if (grepl("float32", as.character(tensor$dtype))) {
          state_dict_to_save[[key]] <- tensor$half()
        } else {
          state_dict_to_save[[key]] <- tensor
        }
        saved_keys <- saved_keys + 1
      }
    }
    torch$save(state_dict_to_save, weights_file)
    if (is_moe) {
      cli::cli_alert_info("Saved {saved_keys} expert/gating parameters (backbone excluded)")
    } else {
      cli::cli_alert_info("Saved {saved_keys} classifier head parameters (backbone excluded)")
    }
  } else {
    # For unfrozen models, save everything in FP16
    state_dict_fp16 <- reticulate::dict()
    fp32_converted <- 0
    for (key in names(state_dict)) {
      tensor <- state_dict[[key]]
      if (grepl("float32", as.character(tensor$dtype))) {
        state_dict_fp16[[key]] <- tensor$half()
        fp32_converted <- fp32_converted + 1
      } else {
        state_dict_fp16[[key]] <- tensor
      }
    }
    torch$save(state_dict_fp16, weights_file)
    cli::cli_alert_info("Saved full model in FP16 ({fp32_converted} tensors, 50% reduction)")
  }

  # Save config with base R saveRDS
  base::saveRDS(config, config_file)

  # Clean up resources to avoid connection issues with subsequent operations
  gc(verbose = FALSE, full = TRUE)

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
      freeze_backbone = config$freeze_backbone,
      trust_remote_code = config$trust_remote_code %||% FALSE
    )
  } else {
    clf <- classifier(
      num_labels = config$num_labels,
      model_name = config$model_name,
      device = device,
      freeze_backbone = config$freeze_backbone,
      trust_remote_code = config$trust_remote_code %||% FALSE
    )
  }

  # Load weights
  torch <- reticulate::import("torch")
  saved_state_dict <- torch$load(weights_file, map_location = device)

  # Get current model state
  pytorch_model <- if (config$model_type == "moe") clf$model else clf$model$model
  current_state_dict <- pytorch_model$state_dict()

  # Convert FP16 back to FP32 and merge with current state
  # This allows partial loading (e.g., only classifier head)
  fp16_converted <- 0
  for (key in names(saved_state_dict)) {
    if (key %in% names(current_state_dict)) {
      tensor <- saved_state_dict[[key]]
      # Convert FP16 tensors back to FP32
      if (grepl("float16", as.character(tensor$dtype))) {
        current_state_dict[[key]] <- tensor$float()
        fp16_converted <- fp16_converted + 1
      } else {
        current_state_dict[[key]] <- tensor
      }
    }
  }

  # Load the merged state dict with strict=False to allow partial loading
  # This is crucial when only classifier head was saved (freeze_backbone=True)
  pytorch_model$load_state_dict(current_state_dict, strict = FALSE)

  num_loaded <- length(names(saved_state_dict))
  num_total <- length(names(current_state_dict))

  if (num_loaded < num_total) {
    cli::cli_alert_info("Loaded {num_loaded}/{num_total} parameters (classifier head only)")
  } else if (fp16_converted > 0) {
    cli::cli_alert_info("Loaded {num_loaded} FP16 tensors and converted to FP32")
  }

  clf$is_trained <- TRUE

  cli::cli_alert_success("Model loaded and ready for predictions")

  clf
}

# Helper for NULL coalescing
`%||%` <- function(x, y) if (is.null(x)) y else x
