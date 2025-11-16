#' Create a Granite classifier
#'
#' Creates a text classifier using IBM's Granite embedding model. Supports both
#' binary classification (2 classes) and multi-class classification (3+ classes).
#' The number of classes can be specified explicitly or inferred from training data.
#'
#' @param num_labels Number of output classes (e.g., 2 for binary, 4 for multi-class).
#'   If NULL and data is provided, will be inferred from unique labels.
#' @param data Optional training data frame to infer num_labels from
#' @param label_col Optional label column name (unquoted) to infer num_labels from
#' @param model_name Model identifier from Hugging Face Hub
#' @param device Device to use ("cpu" or "cuda")
#' @return A Granite classifier object with model and tokenizer
#' @export
#' @seealso \code{\link{granite_train}}, \code{\link{granite_predict}}
#' @examplesIf requireNamespace("transformers")
#' # Explicit num_labels
#' classifier <- granite_classifier(num_labels = 2)
#'
#' # Infer from data
#' data <- tibble::tibble(text = c("a", "b"), label = c("high", "low"))
#' classifier <- granite_classifier(data = data, label_col = label)
granite_classifier <- function(
  num_labels = NULL,
  data = NULL,
  label_col = NULL,
  model_name = "ibm-granite/granite-embedding-english-r2",
  device = "cpu"
) {
  # Infer num_labels from data if not provided
  if (is.null(num_labels) && !is.null(data) && !is.null(rlang::enquo(label_col))) {
    label_col_quo <- rlang::enquo(label_col)
    labels <- dplyr::pull(data, !!label_col_quo)
    num_labels <- length(unique(labels))
    message("Inferred num_labels = ", num_labels, " from data")
  }
  
  if (is.null(num_labels)) {
    stop("num_labels must be specified or inferred from data. ",
         "Use: granite_classifier(num_labels = 2) or ",
         "granite_classifier(data = data, label_col = label)")
  }
  model <- granite_model(
    model_name = model_name,
    task = "classification",
    num_labels = num_labels,
    device = device
  )

  tokenizer <- granite_tokenizer(model_name)

  structure(
    list(
      model = model,
      tokenizer = tokenizer,
      num_labels = num_labels,
      is_trained = FALSE
    ),
    class = c("granite_classifier", "granite_model")
  )
}

#' Train a Granite classifier
#'
#' Trains a text classifier on labeled data. Supports integer, character, and
#' factor labels. Character and factor labels are automatically converted to
#' integers (alphabetically for characters, by factor levels for factors).
#'
#' @param classifier Granite classifier object
#' @param data Training data frame
#' @param text_col Column name containing text (unquoted or string)
#' @param label_col Column name containing labels (unquoted or string).
#'   Can be integer, character, or factor.
#' @param epochs Number of training epochs
#' @param batch_size Batch size for training
#' @param learning_rate Learning rate for optimizer
#' @param validation_split Fraction of data to use for validation
#' @param verbose Whether to print training progress (default: TRUE)
#' @return Updated classifier object with trained model
#' @export
#' @seealso \code{\link{granite_classifier}}, \code{\link{granite_predict}}
#' @examplesIf requireNamespace("transformers")
#' library(dplyr)
#'
#' # Binary classification with integer labels
#' data <- tibble::tibble(
#'   text = c("positive example", "negative example"),
#'   label = c(0, 1)
#' )
#' classifier <- granite_classifier(num_labels = 2) |>
#'   granite_train(data, text, label, epochs = 3)
#'
#' # Multi-class with character labels (auto-converted alphabetically)
#' data_multi <- tibble::tibble(
#'   text = c("urgent issue", "minor bug", "question", "critical"),
#'   priority = c("high", "low", "medium", "critical")
#' )
#' classifier_multi <- granite_classifier(num_labels = 4) |>
#'   granite_train(data_multi, text, priority, epochs = 5)
granite_train <- function(
  classifier,
  data,
  text_col,
  label_col,
  epochs = 3,
  batch_size = 8,
  learning_rate = 5e-5,
  validation_split = 0.2,
  verbose = TRUE
) {
  text_col <- rlang::enquo(text_col)
  label_col <- rlang::enquo(label_col)

  text_col_name <- rlang::as_name(text_col)
  label_col_name <- rlang::as_name(label_col)

  texts <- dplyr::pull(data, !!text_col)
  labels <- dplyr::pull(data, !!label_col)

  if (is.factor(labels) || is.character(labels)) {
    labels <- as.integer(as.factor(labels)) - 1L
  }

  n_train <- floor(length(texts) * (1 - validation_split))
  train_indices <- seq_len(n_train)

  train_texts <- texts[train_indices]
  train_labels <- labels[train_indices]
  val_texts <- texts[-train_indices]
  val_labels <- labels[-train_indices]

  model <- classifier$model$model
  tokenizer <- classifier$tokenizer$tokenizer

  optimizer <- torch$optim$AdamW(
    model$parameters(),
    lr = learning_rate
  )

  model$train()
  
  if (verbose) {
    message(sprintf("Training on %d samples, validating on %d samples",
                    length(train_texts), length(val_texts)))
  }

  for (epoch in seq_len(epochs)) {
    total_loss <- 0
    n_batches <- ceiling(length(train_texts) / batch_size)

    for (i in seq_len(n_batches)) {
      start_idx <- (i - 1) * batch_size + 1
      end_idx <- min(i * batch_size, length(train_texts))

      encodings <- tokenizer(
        train_texts[start_idx:end_idx],
        padding = TRUE,
        truncation = TRUE,
        return_tensors = "pt"
      )

      labels_tensor <- torch$tensor(as.integer(train_labels[start_idx:end_idx]))
      moved <- to_device(encodings, labels_tensor, classifier$model$device)

      optimizer$zero_grad()
      outputs <- model(
        input_ids = moved$encodings$input_ids,
        attention_mask = moved$encodings$attention_mask,
        labels = moved$labels
      )

      outputs$loss$backward()
      optimizer$step()
      total_loss <- total_loss + outputs$loss$item()
    }

    if (verbose) {
      val_msg <- if (length(val_texts) > 0) {
        val_accuracy <- evaluate_classifier(
          model, tokenizer, val_texts, val_labels,
          batch_size, classifier$model$device
        )
        sprintf("Epoch %d/%d - Loss: %.4f - Val Accuracy: %.4f",
                epoch, epochs, total_loss / n_batches, val_accuracy)
      } else {
        sprintf("Epoch %d/%d - Loss: %.4f", epoch, epochs, total_loss / n_batches)
      }
      message(val_msg)
    }
  }

  classifier$is_trained <- TRUE
  classifier
}

#' Internal evaluation function
#' @keywords internal
evaluate_classifier <- function(model, tokenizer, texts, labels, batch_size, device) {
  model$eval()
  correct <- 0

  for (i in seq_len(ceiling(length(texts) / batch_size))) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(texts))

    encodings <- tokenizer(
      texts[start_idx:end_idx],
      padding = TRUE,
      truncation = TRUE,
      return_tensors = "pt"
    )

    moved <- to_device(encodings, device = device)

    with(torch$no_grad(), {
      outputs <- model(
        input_ids = moved$encodings$input_ids,
        attention_mask = moved$encodings$attention_mask
      )
      predictions <- torch$argmax(outputs$logits, dim = -1L)$cpu()$numpy()
      correct <- correct + sum(predictions == labels[start_idx:end_idx])
    })
  }

  model$train()
  correct / length(labels)
}

#' Make predictions with a Granite classifier
#'
#' @param classifier Trained Granite classifier object
#' @param data Data frame containing text to classify
#' @param text_col Column name containing text (unquoted or string)
#' @param type Type of prediction ("class" or "prob")
#' @param batch_size Batch size for prediction
#' @return Data frame with predictions
#' @export
#' @seealso \code{\link{granite_classifier}}, \code{\link{granite_train}}
#' @examplesIf requireNamespace("transformers")
#' # Assuming 'classifier' is a trained model and 'new_data' is a tibble
#' # predictions <- granite_predict(classifier, new_data, text_col = text)
granite_predict <- function(
  classifier,
  data,
  text_col,
  type = c("class", "prob"),
  batch_size = 32
) {
  type <- match.arg(type)

  if (!classifier$is_trained) {
    warning("Classifier has not been trained yet. Predictions may be random.")
  }

  text_col <- rlang::enquo(text_col)
  text_col_name <- rlang::as_name(text_col)

  texts <- dplyr::pull(data, .data[[text_col_name]])

  model <- classifier$model$model
  tokenizer <- classifier$tokenizer$tokenizer

  model$eval()
  predictions_list <- list()

  for (i in seq_len(ceiling(length(texts) / batch_size))) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(texts))

    encodings <- tokenizer(
      texts[start_idx:end_idx],
      padding = TRUE,
      truncation = TRUE,
      return_tensors = "pt"
    )

    moved <- to_device(encodings, device = classifier$model$device)

    with(torch$no_grad(), {
      outputs <- model(
        input_ids = moved$encodings$input_ids,
        attention_mask = moved$encodings$attention_mask
      )

      predictions_list[[i]] <- if (type == "class") {
        torch$argmax(outputs$logits, dim = -1L)$cpu()$numpy()
      } else {
        torch$nn$functional$softmax(outputs$logits, dim = -1L)$cpu()$numpy()
      }
    })
  }

  if (type == "class") {
    dplyr::mutate(data, prediction = unlist(predictions_list))
  } else {
    prob_df <- as.data.frame(do.call(rbind, predictions_list))
    names(prob_df) <- paste0("prob_", seq_len(ncol(prob_df)))
    dplyr::bind_cols(data, prob_df)
  }
}
