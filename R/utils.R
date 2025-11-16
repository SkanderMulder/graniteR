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
