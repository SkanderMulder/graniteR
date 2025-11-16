# Package initialization
transformers <- NULL
torch <- NULL

.onLoad <- function(libname, pkgname) {
  # Initialize Python modules - use .onLoad instead of .onAttach
  # so the bindings are unlocked when we assign
  if (reticulate::py_available(initialize = TRUE)) {
    tryCatch({
      transformers <<- reticulate::import("transformers", delay_load = FALSE)
      torch <<- reticulate::import("torch", delay_load = FALSE)
    }, error = function(e) {
      # Silently fail - we'll show message in .onAttach
    })
  }
}

.onAttach <- function(libname, pkgname) {
  # Show startup messages
  if (is.null(transformers) || is.null(torch)) {
    packageStartupMessage(
      "Python dependencies not found. ",
      "Run install_pyenv() to set up."
    )
  }
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
#' The Python path is automatically configured for the current R session.
#' To make it permanent, add the suggested line to your .Rprofile.
#'
#' Alternatively, run the setup script from the package directory:
#' \code{./setup_python.sh}
#'
#' @export
#' @importFrom processx run
#' @examplesIf requireNamespace("processx")
#' \dontrun{
#' install_pyenv()
#' # Python path is configured automatically
#' # To make permanent, add to .Rprofile as shown in output
#' }
install_pyenv <- function(venv_path = ".venv") {
  if (!requireNamespace("processx", quietly = TRUE)) {
    stop("Package 'processx' is required. Install it with: install.packages('processx')")
  }

  uv_path <- Sys.which("uv")
  if (uv_path == "") {
    cli::cli_alert_danger("UV not found in PATH")
    cli::cli_alert_info("Install it with: {.code curl -LsSf https://astral.sh/uv/install.sh | sh}")
    cli::cli_alert_info("Or run: {.file ./setup_python.sh}")
    cli::cli_alert_info("After installation, you may need to add UV to your PATH")
    cli::cli_alert_info("Then restart R and run {.run install_pyenv()} again")
    return(invisible(FALSE))
  }

  pkg_root <- if (system.file(package = "graniteR") == "") getwd() else system.file(package = "graniteR")
  venv_full_path <- file.path(pkg_root, venv_path)

  if (!dir.exists(venv_full_path)) {
    cli::cli_progress_step("Creating virtual environment with UV")
    result <- tryCatch({
      processx::run(uv_path, c("venv", venv_full_path))
      TRUE
    }, error = function(e) {
      cli::cli_progress_done()
      cli::cli_alert_danger("Failed to create virtual environment")
      cli::cli_alert_danger(conditionMessage(e))
      FALSE
    })
    if (!result) return(invisible(FALSE))
    cli::cli_progress_done()
  }

  cli::cli_progress_step("Installing dependencies with UV")
  req_file <- file.path(pkg_root, "inst", "python", "requirements.txt")

  deps <- if (file.exists(req_file)) {
    c("pip", "install", "-r", req_file)
  } else {
    c("pip", "install", "transformers", "torch", "datasets", "numpy")
  }

  result <- tryCatch({
    processx::run(uv_path, deps, env = c(VIRTUAL_ENV = venv_full_path))
    TRUE
  }, error = function(e) {
    cli::cli_progress_done()
    cli::cli_alert_danger("Failed to install dependencies")
    cli::cli_alert_danger(conditionMessage(e))
    FALSE
  })

  if (!result) return(invisible(FALSE))
  cli::cli_progress_done()

  python_path <- file.path(venv_full_path, "bin", "python")

  # Automatically configure Python path for current session
  Sys.setenv(RETICULATE_PYTHON = python_path)

  cli::cli_alert_success("Setup complete!")
  cli::cli_alert_success("Python path configured for this session")
  cli::cli_alert_info("To make permanent, add to your .Rprofile:")
  cli::cli_code('Sys.setenv(RETICULATE_PYTHON = "{python_path}")')

  invisible(TRUE)
}
