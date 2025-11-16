# Package initialization
transformers <- NULL
torch <- NULL

.onLoad <- function(libname, pkgname) {
  reticulate::py_require("transformers")
  reticulate::py_require("torch")

  transformers <<- reticulate::import("transformers", delay_load = TRUE)
  torch <<- reticulate::import("torch", delay_load = TRUE)
}

#' Install Python dependencies for graniteR
#'
#' @param method Installation method (virtualenv, conda, or auto)
#' @param conda Path to conda executable (if using conda)
#'
#' @details
#' For faster installation, consider using UV via the setup script:
#' Run \code{./setup_python.sh} in the package root, then set
#' \code{Sys.setenv(RETICULATE_PYTHON = ".venv/bin/python")}
#'
#' @export
install_granite <- function(method = "auto", conda = "auto") {
  reticulate::py_install(
    packages = c("transformers", "torch", "datasets", "numpy"),
    method = method,
    conda = conda
  )
}

#' Install Python dependencies using UV (faster)
#'
#' @param venv_path Path to virtual environment (default: .venv)
#'
#' @details
#' This function uses UV for faster Python dependency installation.
#' UV is significantly faster than pip. If UV is not installed, this
#' function will provide instructions to install it.
#'
#' @export
install_granite_uv <- function(venv_path = ".venv") {
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
