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
  cat("graniteR System Check\n")
  cat("====================\n\n")

  py_available <- reticulate::py_available(initialize = TRUE)
  transformers_ok <- py_available && reticulate::py_module_available("transformers")
  torch_ok <- py_available && reticulate::py_module_available("torch")

  # Python configuration
  cat("Python Configuration\n")
  cat("--------------------\n")
  if (py_available) {
    py_config <- reticulate::py_config()
    py_version <- if (is.list(py_config$version)) py_config$version[[1]] else py_config$version
    cat("Python:", py_config$python, "\n")
    cat("Version:", as.character(py_version), "\n")
  } else {
    cat("Python not available\n")
    cat("Run install_pyenv()\n")
  }
  cat("\n")

  # Python packages
  cat("Python Packages\n")
  cat("---------------\n")
  if (transformers_ok) {
    cat("transformers: OK\n")
  } else {
    cat("transformers: NOT FOUND\n")
  }
  if (torch_ok) {
    cat("torch: OK\n")
  } else {
    cat("torch: NOT FOUND\n")
  }
  if (py_available && (!transformers_ok || !torch_ok)) {
    cat("Run install_pyenv()\n")
  }
  cat("\n")

  # CUDA availability
  cuda_available <- torch_ok && suppressWarnings(tryCatch(
    reticulate::import("torch")$cuda$is_available(),
    error = function(e) FALSE
  ))

  cat("CUDA Support\n")
  cat("------------\n")
  if (torch_ok && cuda_available) {
    torch <- reticulate::import("torch")
    cuda_version <- tryCatch(torch$version$cuda, error = function(e) "unknown")
    device_count <- tryCatch(torch$cuda$device_count(), error = function(e) 0L)
    cat("CUDA available (version:", cuda_version, ")\n")
    cat("CUDA devices:", device_count, "\n")
  } else if (torch_ok) {
    cat("CUDA not available (CPU only)\n")
    cat("This is normal if you don't have an NVIDIA GPU\n")
  } else {
    cat("Cannot check (torch not available)\n")
  }
  cat("\n")

  # Recommendations
  cat("Recommendations\n")
  cat("---------------\n")
  if (!py_available || !transformers_ok || !torch_ok) {
    cat("- Run install_pyenv() for fast setup (uses UV)\n")
    cat("- Or run ./setup_python.sh from package directory\n")
  } else {
    cat("System ready for graniteR!\n")
  }

  invisible(list(
    python_available = py_available,
    transformers = transformers_ok,
    torch = torch_ok,
    cuda = cuda_available
  ))
}
