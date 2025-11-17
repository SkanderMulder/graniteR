#' Create a text classifier
#'
#' Creates a text classifier using transformer encoder models. Supports both
#' binary classification (2 classes) and multi-class classification (3+ classes).
#' The number of classes can be specified explicitly or inferred from training data.
#'
#' The pretrained base model is frozen and only the classification head is trained.
#' This approach is more efficient and prevents overfitting on smaller datasets
#' while leveraging the pretrained representations.
#'
#' @param num_labels Number of output classes (e.g., 2 for binary, 4 for multi-class).
#'   If NULL and data is provided, will be inferred from unique labels.
#' @param data Optional training data frame to infer num_labels from
#' @param label_col Optional label column name (unquoted) to infer num_labels from
#' @param model_name Model identifier from Hugging Face Hub
#' @param device Device to use ("cpu" or "cuda"). If NULL, automatically detects GPU availability.
#' @return A classifier object with model and tokenizer
#' @details
#' The classifier uses a frozen pretrained model with only the classification head
#' being trainable. This provides several advantages:
#' \itemize{
#'   \item Faster training (fewer parameters to update)
#'   \item Lower memory requirements
#'   \item Better generalization on small datasets
#'   \item Preserves pretrained knowledge
#' }
#' @export
#' @seealso \code{\link{train}}, \code{\link{predict}}
#' @examplesIf requireNamespace("transformers")
#' # Explicit num_labels
#' clf <- classifier(num_labels = 2)
#'
#' # Infer from data
#' data <- tibble::tibble(text = c("a", "b"), label = c("high", "low"))
#' clf <- classifier(data = data, label_col = label)
classifier <- function(
  num_labels = NULL,
  data = NULL,
  label_col = NULL,
  model_name = "ibm-granite/granite-embedding-english-r2",
  device = NULL
) {
  # Auto-detect device if not specified
  if (is.null(device)) {
    torch <- reticulate::import("torch", delay_load = TRUE)
    device <- if (torch$cuda$is_available()) "cuda" else "cpu"
    cli::cli_alert_info("Using device: {device}")
  }
  # Infer num_labels from data if not provided
  if (is.null(num_labels) && !is.null(data) && !is.null(rlang::enquo(label_col))) {
    label_col_quo <- rlang::enquo(label_col)
    labels <- dplyr::pull(data, !!label_col_quo)
    num_labels <- length(unique(labels))
    cli::cli_alert_info("Inferred num_labels = {num_labels} from data")
  }
  
  if (is.null(num_labels)) {
    stop("num_labels must be specified or inferred from data. ",
         "Use: classifier(num_labels = 2) or ",
         "classifier(data = data, label_col = label)")
  }
  model <- granite_model(
    model_name = model_name,
    task = "classification",
    num_labels = num_labels,
    device = device
  )

  tokenizer <- granite_tokenizer(model_name)

  # Verify parameter freezing
  trainable_params <- sum(sapply(model$model$parameters(), function(p) p$requires_grad))
  total_params <- length(model$model$parameters())

  cli::cli_alert_info(
    "Created classifier with {trainable_params} trainable parameters (head only) out of {total_params} total"
  )

  structure(
    list(
      model = model,
      tokenizer = tokenizer,
      num_labels = num_labels,
      is_trained = FALSE
    ),
    class = c("classifier", "granite_classifier", "granite_model")
  )
}

#' Train a text classifier
#'
#' Trains a text classifier on labeled data. Supports integer, character, and
#' factor labels. Character and factor labels are automatically converted to
#' integers (alphabetically for characters, by factor levels for factors).
#'
#' Only the classification head is trained while the base model remains frozen.
#' This is efficient and effective for most classification tasks, requiring less
#' data and training time compared to full fine-tuning.
#'
#' @param classifier Classifier object
#' @param data Training data frame
#' @param text_col Column name containing text (unquoted or string)
#' @param label_col Column name containing labels (unquoted or string).
#'   Can be integer, character, or factor.
#' @param epochs Number of training epochs
#' @param batch_size Batch size for training
#' @param learning_rate Learning rate for optimizer (typically higher than full
#'   fine-tuning since only the head is trained)
#' @param validation_split Fraction of data to use for validation
#' @param verbose Whether to print training progress (default: TRUE)
#' @return Updated classifier object with trained model
#' @details
#' The training process only updates the classification head parameters while
#' keeping the pretrained encoder frozen. This typically requires:
#' \itemize{
#'   \item Higher learning rates (2e-5 to 1e-3) than full fine-tuning
#'   \item Fewer epochs (3-10 typically sufficient)
#'   \item Less training data to avoid overfitting
#' }
#' @export
#' @seealso \code{\link{classifier}}, \code{\link{predict}}
#' @examplesIf requireNamespace("transformers")
#' library(dplyr)
#'
#' # Binary classification with integer labels
#' data <- tibble::tibble(
#'   text = c("positive example", "negative example"),
#'   label = c(0, 1)
#' )
#' clf <- classifier(num_labels = 2) |>
#'   train(data, text, label, epochs = 3)
#'
#' # Multi-class with character labels (auto-converted alphabetically)
#' data_multi <- tibble::tibble(
#'   text = c("urgent issue", "minor bug", "question", "critical"),
#'   priority = c("high", "low", "medium", "critical")
#' )
#' clf_multi <- classifier(num_labels = 4) |>
#'   train(data_multi, text, priority, epochs = 5)
train <- function(
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
    cli::cli_alert_info("Training on {length(train_texts)} samples, validating on {length(val_texts)} samples")
  }

  for (epoch in seq_len(epochs)) {
    total_loss <- 0
    n_batches <- ceiling(length(train_texts) / batch_size)

    if (verbose) {
      cli::cli_progress_bar(
        format = "Epoch {epoch}/{epochs} | Batch {cli::pb_current}/{cli::pb_total} | Loss: {round(total_loss / cli::pb_current, 4)}",
        total = n_batches,
        clear = FALSE
      )
    }

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

      if (verbose) {
        cli::cli_progress_update()
      }
    }

    if (verbose) {
      cli::cli_progress_done()
    }

    if (verbose) {
      if (length(val_texts) > 0) {
        val_accuracy <- evaluate_classifier(
          model, tokenizer, val_texts, val_labels,
          batch_size, classifier$model$device
        )
        cli::cli_alert_info("Epoch {epoch}/{epochs} - Loss: {round(total_loss / n_batches, 4)} - Val Accuracy: {round(val_accuracy, 4)}")
      } else {
        cli::cli_alert_info("Epoch {epoch}/{epochs} - Loss: {round(total_loss / n_batches, 4)}")
      }
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

#' Make predictions with a trained classifier
#'
#' @param classifier Trained classifier object
#' @param data Data frame containing text to classify
#' @param text_col Column name containing text (unquoted or string)
#' @param type Type of prediction ("class" or "prob")
#' @param batch_size Batch size for prediction
#' @param ... Additional arguments (unused, for S3 consistency)
#' @return Data frame with predictions
#' @export
#' @seealso \code{\link{classifier}}, \code{\link{train}}
#' @examplesIf requireNamespace("transformers")
#' # Assuming 'clf' is a trained model and 'new_data' is a tibble
#' # predictions <- predict(clf, new_data, text_col = text)
predict.granite_classifier <- function(
  classifier,
  data,
  text_col,
  type = c("class", "prob"),
  batch_size = 32,
  ...
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
  n_batches <- ceiling(length(texts) / batch_size)

  cli::cli_progress_bar(
    format = "Predicting | Batch {cli::pb_current}/{cli::pb_total}",
    total = n_batches,
    clear = FALSE
  )

  for (i in seq_len(n_batches)) {
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

    cli::cli_progress_update()
  }

  cli::cli_progress_done()

  if (type == "class") {
    dplyr::mutate(data, prediction = unlist(predictions_list))
  } else {
    prob_df <- as.data.frame(do.call(rbind, predictions_list))
    names(prob_df) <- paste0("prob_", seq_len(ncol(prob_df)))
    dplyr::bind_cols(data, prob_df)
  }
}

#' @rdname predict.granite_classifier
#' @export
predict.classifier <- predict.granite_classifier
