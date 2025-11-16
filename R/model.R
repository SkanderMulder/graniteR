#' Create a Granite embedding model
#'
#' @param model_name Model identifier from Hugging Face Hub
#' @param task Type of model (embedding, classification, or regression)
#' @param num_labels Number of output labels for classification
#' @param device Device to use ("cpu" or "cuda")
#' @return A Granite model object
#' @export
#' @seealso \code{\link{granite_tokenizer}}
#' @examplesIf requireNamespace("transformers")
#' # Create an embedding model
#' model <- granite_model()
#'
#' # Create a classification model
#' model <- granite_model(task = "classification", num_labels = 3)
granite_model <- function(
  model_name = "ibm-granite/granite-embedding-english-r2",
  task = c("embedding", "classification", "regression"),
  num_labels = NULL,
  device = "cpu"
) {
  task <- match.arg(task)
  
  # Check CUDA availability if device is cuda
  if (device == "cuda") {
    cuda_available <- suppressWarnings(tryCatch({
      torch$cuda$is_available()
    }, error = function(e) FALSE))
    
    if (!cuda_available) {
      warning(
        "CUDA device requested but not available. Falling back to CPU. ",
        "This may be due to incompatible CUDA/driver versions.",
        call. = FALSE
      )
      device <- "cpu"
    }
  }

  # Suppress transformers warnings during model loading
  model <- suppressWarnings({
    switch(
      task,
      embedding = transformers$AutoModel$from_pretrained(model_name),
      classification = {
        if (is.null(num_labels)) {
          stop("num_labels must be specified for classification tasks")
        }
        transformers$AutoModelForSequenceClassification$from_pretrained(
          model_name,
          num_labels = as.integer(num_labels)
        )
      },
      regression = {
        transformers$AutoModelForSequenceClassification$from_pretrained(
          model_name,
          num_labels = 1L
        )
      }
    )
  })

  if (device == "cuda") {
    model$to(torch$device("cuda"))
  }

  structure(
    list(
      model = model,
      model_name = model_name,
      task = task,
      num_labels = num_labels,
      device = device
    ),
    class = "granite_model"
  )
}

#' Create a Granite tokenizer
#'
#' @param model_name Model identifier from Hugging Face Hub
#' @return A Granite tokenizer object
#' @export
#' @seealso \code{\link{granite_model}}
#' @examplesIf requireNamespace("transformers")
#' tokenizer <- granite_tokenizer()
granite_tokenizer <- function(
  model_name = "ibm-granite/granite-embedding-english-r2"
) {
  tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)

  structure(
    list(
      tokenizer = tokenizer,
      model_name = model_name
    ),
    class = "granite_tokenizer"
  )
}

#' Print method for granite_model
#' @param x granite_model object to print
#' @param ... Additional arguments passed to print
#' @export
print.granite_model <- function(x, ...) {
  cat("Granite Model\n")
  cat("Model:", x$model_name, "\n")
  cat("Task:", x$task, "\n")
  if (!is.null(x$num_labels)) {
    cat("Labels:", x$num_labels, "\n")
  }
  cat("Device:", x$device, "\n")
  invisible(x)
}

#' Print method for granite_tokenizer
#' @param x granite_tokenizer object to print
#' @param ... Additional arguments passed to print
#' @export
print.granite_tokenizer <- function(x, ...) {
  cat("Granite Tokenizer\n")
  cat("Model:", x$model_name, "\n")
  invisible(x)
}
