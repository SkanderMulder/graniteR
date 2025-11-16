# Setup Python environment for tests
# This file runs before any tests

# Priority 1: Check if RETICULATE_PYTHON is already set
if (nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
  message("Using RETICULATE_PYTHON: ", Sys.getenv("RETICULATE_PYTHON"))
} else {
  # Priority 2: Look for .venv directory
  # Start from tests/testthat and go up to find package root
  venv_search_paths <- c(
    file.path("..", ".."),  # From tests/testthat -> package root
    ".",                    # From package root
    "../.."                 # Alternative
  )
  
  venv_found <- FALSE
  for (base_path in venv_search_paths) {
    venv_dir <- file.path(base_path, ".venv")
    if (dir.exists(venv_dir)) {
      # Use use_virtualenv with absolute path for proper environment detection
      venv_abs <- normalizePath(venv_dir, mustWork = FALSE)
      tryCatch({
        reticulate::use_virtualenv(venv_abs, required = TRUE)
        message("✓ Using .venv: ", venv_abs)
        venv_found <- TRUE
        break
      }, error = function(e) {
        # Continue searching
      })
    }
  }
  
  if (!venv_found) {
    # Priority 3: Try system Python installations
    python_paths <- c(
      Sys.which("python3"),
      Sys.which("python")
    )
    
    for (py_path in python_paths) {
      if (nzchar(py_path) && file.exists(py_path)) {
        reticulate::use_python(py_path, required = FALSE)
        message("Using system Python: ", py_path)
        break
      }
    }
  }
}

# Check if transformers is available
if (reticulate::py_available(initialize = TRUE)) {
  if (!reticulate::py_module_available("transformers")) {
    message("\n⚠️  transformers module not available. Tests requiring Python will be skipped.")
    message("   To enable all tests, run: install_granite() or install_granite_uv()")
  } else {
    message("✓ Python modules ready: transformers, torch")
  }
} else {
  message("\n⚠️  Python not configured. Tests requiring Python will be skipped.")
  message("   Set RETICULATE_PYTHON or run: ./setup_python.sh")
}
