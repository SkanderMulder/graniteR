# Using native pipe from R >= 4.1.0

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

#' Check system capabilities for graniteR
#'
#' Checks Python environment, CUDA availability, and provides system information.
#'
#' @return Invisibly returns a list with system information
#' @export
#' @examples
#' \dontrun{
#' granite_check_system()
#' }
granite_check_system <- function() {
  cat("=== graniteR System Check ===\n\n")
  
  # Python availability
  cat("Python Configuration:\n")
  py_available <- reticulate::py_available(initialize = TRUE)
  if (py_available) {
    py_config <- reticulate::py_config()
    cat("  ✓ Python:", py_config$python, "\n")
    py_version <- if (is.list(py_config$version)) py_config$version[[1]] else py_config$version
    cat("  ✓ Version:", as.character(py_version), "\n")
  } else {
    cat("  ✗ Python not available\n")
    cat("    Run install_granite() or install_granite_uv()\n")
  }
  
  # Python packages
  cat("\nPython Packages:\n")
  if (py_available) {
    transformers_ok <- reticulate::py_module_available("transformers")
    torch_ok <- reticulate::py_module_available("torch")
    
    cat("  ", if (transformers_ok) "✓" else "✗", " transformers\n")
    cat("  ", if (torch_ok) "✓" else "✗", " torch\n")
    
    if (!transformers_ok || !torch_ok) {
      cat("    Run install_granite() or install_granite_uv()\n")
    }
  } else {
    cat("  ✗ Cannot check (Python not available)\n")
  }
  
  # CUDA availability
  cat("\nCUDA Support:\n")
  if (py_available && reticulate::py_module_available("torch")) {
    cuda_available <- suppressWarnings(tryCatch({
      torch <- reticulate::import("torch")
      torch$cuda$is_available()
    }, error = function(e) FALSE))
    
    if (cuda_available) {
      cuda_version <- tryCatch({
        torch$version$cuda
      }, error = function(e) "unknown")
      cat("  ✓ CUDA available (version:", cuda_version, ")\n")
      
      device_count <- tryCatch({
        torch$cuda$device_count()
      }, error = function(e) 0L)
      cat("  ✓ CUDA devices:", device_count, "\n")
    } else {
      cat("  ✗ CUDA not available (CPU only)\n")
      cat("    This is normal if you don't have an NVIDIA GPU\n")
      cat("    Training and inference will use CPU\n")
    }
  } else {
    cat("  ? Cannot check (torch not available)\n")
  }
  
  # Recommendations
  cat("\nRecommendations:\n")
  if (!py_available || 
      !reticulate::py_module_available("transformers") || 
      !reticulate::py_module_available("torch")) {
    cat("  • Run install_granite_uv() for fast setup (recommended)\n")
    cat("  • Or run install_granite() for standard pip installation\n")
    cat("  • Or run ./setup_python.sh from package directory\n")
  } else {
    cat("  ✓ System ready for graniteR!\n")
  }
  
  cat("\n")
  
  invisible(list(
    python_available = py_available,
    transformers = if (py_available) reticulate::py_module_available("transformers") else FALSE,
    torch = if (py_available) reticulate::py_module_available("torch") else FALSE,
    cuda = if (py_available && reticulate::py_module_available("torch")) {
      tryCatch({
        torch <- reticulate::import("torch")
        torch$cuda$is_available()
      }, error = function(e) FALSE)
    } else FALSE
  ))
}
