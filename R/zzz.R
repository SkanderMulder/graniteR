# Package initialization
transformers <- NULL
torch <- NULL

.onAttach <- function(libname, pkgname) {
  # Try to load Python dependencies with informative messages
  tryCatch({
    reticulate::py_require("transformers")
    reticulate::py_require("torch")

    transformers <<- reticulate::import("transformers", delay_load = TRUE)
    torch <<- reticulate::import("torch", delay_load = TRUE)
  }, error = function(e) {
    packageStartupMessage(
      "Python dependencies not found. ",
      "Run install_granite() or install_granite_uv() to set up."
    )
  })
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
#' @importFrom processx run
#' @examplesIf requireNamespace("processx")
#' \dontrun{
#' install_granite()
#' # Then set the Python path
#' Sys.setenv(RETICULATE_PYTHON = ".venv/bin/python")
#' }
install_granite <- function(venv_path = ".venv") {
  if (!requireNamespace("processx", quietly = TRUE)) {
    stop("Package 'processx' is required. Install it with: install.packages('processx')")
  }

  uv_path <- Sys.which("uv")
  if (uv_path == "") {
    cli::cli_alert_danger("UV not found in PATH")
    cli::cli_alert_info("Install it with: {.code curl -LsSf https://astral.sh/uv/install.sh | sh}")
    cli::cli_alert_info("Or run: {.file ./setup_python.sh}")
    cli::cli_alert_info("After installation, you may need to add UV to your PATH")
    cli::cli_alert_info("Then restart R and run {.run install_granite()} again")
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
  cli::cli_alert_success("Setup complete!")
  cli::cli_alert_info("Add this to your .Rprofile or script:")
  cli::cli_code('Sys.setenv(RETICULATE_PYTHON = "{python_path}")')

  invisible(TRUE)
}
