# Package initialization
transformers <- NULL
torch <- NULL

.onLoad <- function(libname, pkgname) {
  reticulate::py_require("transformers")
  reticulate::py_require("torch")

  transformers <<- reticulate::import("transformers", delay_load = TRUE)
  torch <<- reticulate::import("torch", delay_load = TRUE)
}

#' Install Python dependencies for graniteR using UV
#'
#' @param venv_path Path to virtual environment (default: .venv)
#'
#' @details
#' This function uses UV for fast Python dependency installation.
#' UV is 10-100x faster than pip. If UV is not installed, this
#' function will provide instructions to install it automatically.
#'
#' Alternatively, run the setup script from the package directory:
#' \code{./setup_python.sh}
#'
#' @export
#' @examples
#' \dontrun{
#' install_granite()
#' # Then set the Python path
#' Sys.setenv(RETICULATE_PYTHON = ".venv/bin/python")
#' }
install_granite <- function(venv_path = ".venv") {
  if (!requireNamespace("processx", quietly = TRUE)) {
    stop("Package 'processx' is required. Install it with: install.packages('processx')")
  }

  uv_available <- tryCatch(
    {
      processx::run("uv", "--version")
      TRUE
    },
    error = function(e) FALSE
  )

  if (!uv_available) {
    message("UV not found. Install it with:")
    message("  curl -LsSf https://astral.sh/uv/install.sh | sh")
    message("\nOr run the setup script: ./setup_python.sh")
    message("\nAfter installing UV, restart R and run install_granite() again.")
    return(invisible(FALSE))
  }

  pkg_root <- system.file(package = "graniteR")
  if (pkg_root == "") pkg_root <- getwd()

  venv_full_path <- file.path(pkg_root, venv_path)

  if (!dir.exists(venv_full_path)) {
    message("Creating virtual environment with UV...")
    processx::run("uv", c("venv", venv_full_path))
  }

  message("Installing dependencies with UV...")
  req_file <- file.path(pkg_root, "inst", "python", "requirements.txt")

  if (file.exists(req_file)) {
    processx::run("uv", c("pip", "install", "-r", req_file),
                 env = c(VIRTUAL_ENV = venv_full_path))
  } else {
    processx::run("uv", c("pip", "install", "transformers", "torch", "datasets", "numpy"),
                 env = c(VIRTUAL_ENV = venv_full_path))
  }

  python_path <- file.path(venv_full_path, "bin", "python")
  message("\nSetup complete! Add this to your .Rprofile or script:")
  message('  Sys.setenv(RETICULATE_PYTHON = "', python_path, '")')

  invisible(TRUE)
}
