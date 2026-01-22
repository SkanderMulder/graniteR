#' Download pre-trained vignette models
#'
#' Downloads models used in package vignettes from GitHub releases.
#' These models are too large to include in the CRAN package.
#'
#' @param model_name Character; name of model to download. One of:
#'   "emotion_standard", "emotion_moe", "sentiment_standard",
#'   "sentiment_moe", "hate_speech_standard"
#' @param destination Character; directory to save models (default: "models/")
#' @param version Character; package version/release tag (default: "v0.1.1")
#' @return Invisibly returns TRUE if successful
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
                          destination = "models/",
                          version = "v0.1.1") {

  available_models <- c(
    "emotion_standard", "emotion_moe",
    "sentiment_standard", "sentiment_moe",
    "hate_speech_standard"
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
#' @param destination Character; directory to save models (default: "models/")
#' @param version Character; package version/release tag (default: "v0.1.1")
#' @return Invisibly returns TRUE if successful
#' @export
#' @examples
#' \dontrun{
#' download_vignette_models()
#' }
download_vignette_models <- function(destination = "models/",
                                    version = "v0.1.1") {

  models <- c(
    "emotion_standard", "emotion_moe",
    "sentiment_standard", "sentiment_moe",
    "hate_speech_standard"
  )

  cli::cli_h2("Downloading Vignette Models")
  cli::cli_alert_info("This will download ~600MB of model files")

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
    "hate_speech_standard"
  )
}
