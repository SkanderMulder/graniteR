# Internal helpers
check_model <- function(model) {
  if (is.null(model)) {
    stop("Model is NULL. Create a model first using granite_model().")
  }
  invisible(model)
}

# Load model from Hugging Face with retry logic to handle connection issues
load_model_with_retry <- function(model_name, task, num_labels = NULL, max_retries = 5) {
  for (attempt in seq_len(max_retries)) {
    tryCatch({
      # Clear any stale HTTP connections
      if (attempt > 1) {
        gc(verbose = FALSE, full = TRUE)
        Sys.sleep(min(2^(attempt - 1), 10))  # Exponential backoff, max 10s
      }

      model <- switch(
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

      return(model)
    }, error = function(e) {
      error_msg <- conditionMessage(e)

      # Check if it's a connection error that might be retryable
      is_connection_error <- any(sapply(
        c("Connection aborted", "RemoteDisconnected", "Connection reset", "Timeout"),
        function(pattern) grepl(pattern, error_msg, ignore.case = TRUE)
      ))

      if (is_connection_error && attempt < max_retries) {
        cli::cli_alert_warning(
          "Connection error (attempt {attempt}/{max_retries}): Retrying in {min(2^(attempt - 1), 10)}s..."
        )
        # Don't stop, let the loop continue
        NULL
      } else if (attempt >= max_retries) {
        cli::cli_alert_danger("Failed to load model after {max_retries} attempts")
        stop(error_msg, call. = FALSE)
      } else {
        # Non-connection error, fail immediately
        stop(error_msg, call. = FALSE)
      }
    })
  }
}

# Load tokenizer from Hugging Face with retry logic
load_tokenizer_with_retry <- function(model_name, max_retries = 5) {
  for (attempt in seq_len(max_retries)) {
    tryCatch({
      # Clear any stale HTTP connections
      if (attempt > 1) {
        gc(verbose = FALSE, full = TRUE)
        Sys.sleep(min(2^(attempt - 1), 10))  # Exponential backoff, max 10s
      }

      tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)
      return(tokenizer)
    }, error = function(e) {
      error_msg <- conditionMessage(e)

      # Check if it's a connection error that might be retryable
      is_connection_error <- any(sapply(
        c("Connection aborted", "RemoteDisconnected", "Connection reset", "Timeout"),
        function(pattern) grepl(pattern, error_msg, ignore.case = TRUE)
      ))

      if (is_connection_error && attempt < max_retries) {
        cli::cli_alert_warning(
          "Connection error loading tokenizer (attempt {attempt}/{max_retries}): Retrying in {min(2^(attempt - 1), 10)}s..."
        )
        # Don't stop, let the loop continue
        NULL
      } else if (attempt >= max_retries) {
        cli::cli_alert_danger("Failed to load tokenizer after {max_retries} attempts")
        stop(error_msg, call. = FALSE)
      } else {
        # Non-connection error, fail immediately
        stop(error_msg, call. = FALSE)
      }
    })
  }
}

check_tokenizer <- function(tokenizer) {
  if (is.null(tokenizer)) {
    stop("Tokenizer is NULL. Create a tokenizer first using granite_tokenizer().")
  }
  invisible(tokenizer)
}

# Move encodings to device if CUDA
to_device <- function(encodings, labels = NULL, device = "cpu") {
  if (device == "cuda") {
    cuda_device <- torch$device("cuda")
    encodings$input_ids <- encodings$input_ids$to(cuda_device)
    encodings$attention_mask <- encodings$attention_mask$to(cuda_device)
    if (!is.null(labels)) {
      labels <- labels$to(cuda_device)
    }
  }
  list(encodings = encodings, labels = labels)
}

# Check for pending R interrupts and clean up Python resources if interrupted
check_interrupt <- function(model = NULL, device = "cpu") {
  tryCatch({
    Sys.sleep(0)
  }, interrupt = function(e) {
    if (!is.null(model)) {
      tryCatch({
        if (!is.null(model$zero_grad)) {
          model$zero_grad()
        }
      }, error = function(e) {})
    }
    if (device == "cuda") {
      tryCatch({
        torch$cuda$empty_cache()
      }, error = function(e) {})
    }
    stop("Training interrupted by user", call. = FALSE)
  })
  invisible(NULL)
}

#' Check system capabilities for graniteR
#'
#' Checks Python environment, CUDA availability, and provides system information.
#'
#' @return Invisibly returns a list with system information
#' @export
#' @examplesIf requireNamespace("reticulate")
#' granite_check_system()
granite_check_system <- function() {
  cli::cli_h1("graniteR System Check")


  py_available <- reticulate::py_available(initialize = TRUE)
  transformers_ok <- py_available && reticulate::py_module_available("transformers")
  torch_ok <- py_available && reticulate::py_module_available("torch")

  # Python configuration
  cli::cli_h2("Python Configuration")

  if (py_available) {
    py_config <- reticulate::py_config()
    py_version <- if (is.list(py_config$version)) py_config$version[[1]] else py_config$version
    cli::cli_alert_success("Python: {.path {py_config$python}}")
    cli::cli_alert_success("Version: {as.character(py_version)}")
  } else {
    cli::cli_alert_danger("Python not available")
    cli::cli_alert_info("Run {.run install_pyenv()}")
  }

  cli::cli_h2("Python Packages")
  if (transformers_ok) {
    cli::cli_alert_success("transformers")
  } else {
    cli::cli_alert_danger("transformers")
  }
  if (torch_ok) {
    cli::cli_alert_success("torch")
  } else {
    cli::cli_alert_danger("torch")
  }
  if (py_available && (!transformers_ok || !torch_ok)) {
    cli::cli_alert_info("Run {.run install_pyenv()}")
  }

  # CUDA availability
  cuda_available <- torch_ok && suppressWarnings(tryCatch(
    reticulate::import("torch")$cuda$is_available(),
    error = function(e) FALSE
  ))

  cli::cli_h2("CUDA Support")

  if (torch_ok && cuda_available) {
    torch <- reticulate::import("torch")
    cuda_version <- tryCatch(torch$version$cuda, error = function(e) "unknown")
    device_count <- tryCatch(torch$cuda$device_count(), error = function(e) 0L)
    cli::cli_alert_success("CUDA available (version: {cuda_version})")
    cli::cli_alert_success("CUDA devices: {device_count}")
  } else if (torch_ok) {
    cli::cli_alert_warning("CUDA not available (CPU only)")
    cli::cli_alert_info("This is normal if you don't have an NVIDIA GPU")
  } else {
    cli::cli_alert_warning("Cannot check (torch not available)")
  }


  # Recommendations
  cli::cli_h2("Recommendations")

  if (!py_available || !transformers_ok || !torch_ok) {
    cli::cli_ul(c(
      "Run {.run install_pyenv()} for fast setup (uses UV)",
      "Or run {.file ./setup_python.sh} from package directory"
    ))
  } else {
    cli::cli_alert_success("System ready for graniteR!")
  }

  invisible(list(
    python_available = py_available,
    transformers = transformers_ok,
    torch = torch_ok,
    cuda = cuda_available
  ))
}
