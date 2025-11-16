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

# Helper function to skip tests if Python or modules are not available
skip_if_no_python_or_modules <- function() {
  if (!reticulate::py_available(initialize = TRUE)) {
    testthat::skip("Python not available for testing")
  }
  if (!reticulate::py_module_available("transformers")) {
    testthat::skip("transformers Python module not available for testing")
  }
}

# Announce whether tests will be skipped or not
if (reticulate::py_available(initialize = TRUE) && reticulate::py_module_available("transformers")) {
  message("✓ Python environment ready for tests (transformers found).")
  # Force import of transformers and torch for tests
  tryCatch({
    assignInNamespace("transformers", reticulate::import("transformers", delay_load = FALSE), "graniteR")
    assignInNamespace("torch", reticulate::import("torch", delay_load = FALSE), "graniteR")
  }, error = function(e) {
    message("Failed to force import transformers/torch in tests: ", e$message)
  })
} else {
  message("⚠️ Python dependencies not found. Tests requiring them will be skipped.")
  message("   Run install_granite() to set up the Python environment.")
}
