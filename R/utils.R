# Internal helpers
check_model <- function(model) {
  if (is.null(model)) {
    stop("Model is NULL. Create a model first using granite_model().")
  }
  invisible(model)
}

check_tokenizer <- function(tokenizer) {
  if (is.null(tokenizer)) {
    stop("Tokenizer is NULL. Create a tokenizer first using granite_tokenizer().")
  }
  invisible(tokenizer)
}

# Move encodings to device if CUDA
to_device <- function(encodings, labels = NULL, device = "cpu") {
  if (device == "cuda") {
    cuda_device <- torch$device("cuda")
    encodings$input_ids <- encodings$input_ids$to(cuda_device)
    encodings$attention_mask <- encodings$attention_mask$to(cuda_device)
    if (!is.null(labels)) {
      labels <- labels$to(cuda_device)
    }
  }
  list(encodings = encodings, labels = labels)
}

#' Check system capabilities for graniteR
#'
#' Checks Python environment, CUDA availability, and provides system information.
#'
#' @return Invisibly returns a list with system information
#' @export
#' @examplesIf requireNamespace("reticulate")
#' granite_check_system()
granite_check_system <- function() {
  cli::cli_h1("graniteR System Check")

  py_available <- reticulate::py_available(initialize = TRUE)
  transformers_ok <- py_available && reticulate::py_module_available("transformers")
  torch_ok <- py_available && reticulate::py_module_available("torch")

  # Python configuration
  cli::cli_h2("Python Configuration")
  if (py_available) {
    py_config <- reticulate::py_config()
    py_version <- if (is.list(py_config$version)) py_config$version[[1]] else py_config$version
    cli::cli_alert_success("Python: {.path {py_config$python}}")
    cli::cli_alert_success("Version: {as.character(py_version)}")
  } else {
    cli::cli_alert_danger("Python not available")
    cli::cli_alert_info("Run {.run install_pyenv()}")
  }

  # Python packages
  cli::cli_h2("Python Packages")
  if (transformers_ok) {
    cli::cli_alert_success("transformers")
  } else {
    cli::cli_alert_danger("transformers")
  }
  if (torch_ok) {
    cli::cli_alert_success("torch")
  } else {
    cli::cli_alert_danger("torch")
  }
  if (py_available && (!transformers_ok || !torch_ok)) {
    cli::cli_alert_info("Run {.run install_pyenv()}")
  }

  # CUDA availability
  cuda_available <- torch_ok && suppressWarnings(tryCatch(
    reticulate::import("torch")$cuda$is_available(),
    error = function(e) FALSE
  ))

  cli::cli_h2("CUDA Support")
  if (torch_ok && cuda_available) {
    torch <- reticulate::import("torch")
    cuda_version <- tryCatch(torch$version$cuda, error = function(e) "unknown")
    device_count <- tryCatch(torch$cuda$device_count(), error = function(e) 0L)
    cli::cli_alert_success("CUDA available (version: {cuda_version})")
    cli::cli_alert_success("CUDA devices: {device_count}")
  } else if (torch_ok) {
    cli::cli_alert_warning("CUDA not available (CPU only)")
    cli::cli_alert_info("This is normal if you don't have an NVIDIA GPU")
  } else {
    cli::cli_alert_warning("Cannot check (torch not available)")
  }

  # Recommendations
  cli::cli_h2("Recommendations")
  if (!py_available || !transformers_ok || !torch_ok) {
    cli::cli_ul(c(
      "Run {.run install_pyenv()} for fast setup (uses UV)",
      "Or run {.file ./setup_python.sh} from package directory"
    ))
  } else {
    cli::cli_alert_success("System ready for graniteR!")
  }

  invisible(list(
    python_available = py_available,
    transformers = transformers_ok,
    torch = torch_ok,
    cuda = cuda_available
  ))
}
