#' Get the default models directory
#'
#' Returns the appropriate directory for storing pre-trained models.
#' For installed packages, uses inst/extdata/models/. For development, uses extdata/models/.
#'
#' @return Character; path to models directory
#' @export
#' @examples
#' get_models_dir()
get_models_dir <- function() {
  # Try installed package location first
  pkg_path <- system.file("extdata", "models", package = "graniteR")

  if (pkg_path != "" && dir.exists(pkg_path)) {
    return(pkg_path)
  }

  # For development: use extdata/models in package root
  dev_path <- file.path("inst", "extdata", "models")
  if (dir.exists(dev_path)) {
    return(dev_path)
  }

  # Create if doesn't exist
  dir.create(dev_path, recursive = TRUE, showWarnings = FALSE)
  return(dev_path)
}

#' Find a model file path
#'
#' Searches for model files in multiple locations:
#' 1. inst/extdata/models/ (installed package)
#' 2. extdata/models/ (development)
#' 3. models/ (backward compatibility)
#'
#' @param model_name Character; base model name (e.g., "emotion_standard")
#' @return Character; full path to model (without _config.rds or _weights.pt suffix)
#' @export
#' @examples
#' \dontrun{
#' model_path <- find_model("emotion_standard")
#' clf <- load_classifier(model_path)
#' }
find_model <- function(model_name) {
  # Search locations in priority order
  search_paths <- c(
    system.file("extdata", "models", package = "graniteR"),
    file.path("inst", "extdata", "models"),
    "models"
  )

  for (path in search_paths) {
    if (path == "") next
    if (!dir.exists(path)) next

    model_path <- file.path(path, model_name)
    config_file <- paste0(model_path, "_config.rds")
    weights_file <- paste0(model_path, "_weights.pt")

    if (file.exists(config_file) && file.exists(weights_file)) {
      return(model_path)
    }
  }

  cli::cli_abort(c(
    "Model {.val {model_name}} not found.",
    "i" = "Download it with: {.code download_model('{model_name}')}"
  ))
}

#' Download pre-trained vignette models
#'
#' Downloads models used in package vignettes from GitHub releases.
#' These models are too large to include in the CRAN package.
#'
#' @param model_name Character; name of model to download. One of:
#'   "emotion_standard", "emotion_moe", "sentiment_standard",
#'   "sentiment_moe", "hate_speech_standard", "hate_speech_moe",
#'   "malicious_prompts_standard", "malicious_prompts_moe"
#' @param destination Character; directory to save models (default: auto-detect)
#' @param version Character; package version/release tag (default: "v0.1.1")
#' @return Invisibly returns TRUE if successful
#' @importFrom utils download.file
#' @export
#' @examples
#' \dontrun{
#' # Download a single model
#' download_model("emotion_standard")
#'
#' # Download all vignette models
#' download_vignette_models()
#' }
download_model <- function(model_name,
                          destination = NULL,
                          version = "v0.1.1") {

  if (is.null(destination)) {
    destination <- get_models_dir()
  }

  available_models <- c(
    "emotion_standard", "emotion_moe",
    "sentiment_standard", "sentiment_moe",
    "hate_speech_standard", "hate_speech_moe",
    "malicious_prompts_standard", "malicious_prompts_moe"
  )

  if (!model_name %in% available_models) {
    cli::cli_abort("Model {.val {model_name}} not available. Choose from: {.val {available_models}}")
  }

  if (!dir.exists(destination)) {
    dir.create(destination, recursive = TRUE)
  }

  base_url <- paste0(
    "https://github.com/SkanderMulder/graniteR/releases/download/",
    version, "/"
  )

  config_file <- paste0(model_name, "_config.rds")
  weights_file <- paste0(model_name, "_weights.pt")

  config_path <- file.path(destination, config_file)
  weights_path <- file.path(destination, weights_file)

  # Check if already downloaded
  if (file.exists(config_path) && file.exists(weights_path)) {
    cli::cli_alert_info("Model {.val {model_name}} already exists in {.path {destination}}")
    return(invisible(TRUE))
  }

  cli::cli_progress_step("Downloading {model_name} configuration...")
  tryCatch({
    download.file(
      paste0(base_url, config_file),
      config_path,
      mode = "wb",
      quiet = TRUE
    )
  }, error = function(e) {
    cli::cli_abort("Failed to download config: {conditionMessage(e)}")
  })

  cli::cli_progress_step("Downloading {model_name} weights (may take a few minutes)...")
  tryCatch({
    download.file(
      paste0(base_url, weights_file),
      weights_path,
      mode = "wb",
      quiet = TRUE
    )
  }, error = function(e) {
    cli::cli_abort("Failed to download weights: {conditionMessage(e)}")
  })

  cli::cli_alert_success("Downloaded {model_name} to {.path {destination}}")

  invisible(TRUE)
}

#' Download all vignette models
#'
#' Convenience function to download all pre-trained models used in vignettes.
#'
#' @param destination Character; directory to save models (default: auto-detect)
#' @param version Character; package version/release tag (default: "v0.1.1")
#' @return Invisibly returns TRUE if successful
#' @export
#' @examples
#' \dontrun{
#' download_vignette_models()
#' }
download_vignette_models <- function(destination = NULL,
                                    version = "v0.1.1") {

  if (is.null(destination)) {
    destination <- get_models_dir()
  }

  models <- c(
    "emotion_standard", "emotion_moe",
    "sentiment_standard", "sentiment_moe",
    "hate_speech_standard", "hate_speech_moe",
    "malicious_prompts_standard", "malicious_prompts_moe"
  )

  cli::cli_h2("Downloading Vignette Models")
  cli::cli_alert_info("This will download ~590MB of model files (8 models)")

  for (model in models) {
    download_model(model, destination, version)
  }

  cli::cli_alert_success("All vignette models downloaded!")

  invisible(TRUE)
}

#' List available pre-trained models
#'
#' @return Character vector of available model names
#' @export
#' @examples
#' list_available_models()
list_available_models <- function() {
  c(
    "emotion_standard", "emotion_moe",
    "sentiment_standard", "sentiment_moe",
    "hate_speech_standard", "hate_speech_moe",
    "malicious_prompts_standard", "malicious_prompts_moe"
  )
}
