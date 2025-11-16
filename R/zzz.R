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
#' @export
install_granite <- function(method = "auto", conda = "auto") {
  reticulate::py_install(
    packages = c("transformers", "torch", "datasets", "numpy"),
    method = method,
    conda = conda
  )
}
