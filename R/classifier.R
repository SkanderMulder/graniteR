#' Create a Granite classifier
#'
#' Creates a text classifier using IBM's Granite embedding model. Supports both
#' binary classification (2 classes) and multi-class classification (3+ classes).
#'
#' @param num_labels Number of output classes (e.g., 2 for binary, 4 for multi-class)
#' @param model_name Model identifier from Hugging Face Hub
#' @param device Device to use ("cpu" or "cuda")
#' @return A Granite classifier object with model and tokenizer
#' @export
#' @examples
#' \dontrun{
#' # Binary classification
#' classifier <- granite_classifier(num_labels = 2)
#'
#' # Multi-class classification (e.g., priority levels)
#' classifier <- granite_classifier(num_labels = 4)
#' }
granite_classifier <- function(
  num_labels,
  model_name = "ibm-granite/granite-embedding-english-r2",
  device = "cpu"
) {
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
#' @return Updated classifier object with trained model
#' @export
#' @examples
#' \dontrun{
#' library(dplyr)
#'
#' # Binary classification with integer labels
#' data <- tibble(
#'   text = c("positive example", "negative example"),
#'   label = c(0, 1)
#' )
#' classifier <- granite_classifier(num_labels = 2) |>
#'   granite_train(data, text, label, epochs = 3)
#'
#' # Multi-class with character labels (auto-converted alphabetically)
#' data_multi <- tibble(
#'   text = c("urgent issue", "minor bug", "question", "critical"),
#'   priority = c("high", "low", "medium", "critical")
#' )
#' classifier_multi <- granite_classifier(num_labels = 4) |>
#'   granite_train(data_multi, text, priority, epochs = 5)
#' }
granite_train <- function(
  classifier,
  data,
  text_col,
  label_col,
  epochs = 3,
  batch_size = 8,
  learning_rate = 5e-5,
  validation_split = 0.2
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

  for (epoch in seq_len(epochs)) {
    total_loss <- 0
    n_batches <- ceiling(length(train_texts) / batch_size)

    for (i in seq_len(n_batches)) {
      start_idx <- (i - 1) * batch_size + 1
      end_idx <- min(i * batch_size, length(train_texts))

      batch_texts <- train_texts[start_idx:end_idx]
      batch_labels <- train_labels[start_idx:end_idx]

      encodings <- tokenizer(
        batch_texts,
        padding = TRUE,
        truncation = TRUE,
        return_tensors = "pt"
      )

      labels_tensor <- torch$tensor(as.integer(batch_labels))

      if (classifier$model$device == "cuda") {
        encodings$input_ids <- encodings$input_ids$to(torch$device("cuda"))
        encodings$attention_mask <- encodings$attention_mask$to(torch$device("cuda"))
        labels_tensor <- labels_tensor$to(torch$device("cuda"))
      }

      optimizer$zero_grad()

      outputs <- model(
        input_ids = encodings$input_ids,
        attention_mask = encodings$attention_mask,
        labels = labels_tensor
      )

      loss <- outputs$loss
      loss$backward()
      optimizer$step()

      total_loss <- total_loss + loss$item()
    }

    avg_loss <- total_loss / n_batches

    if (length(val_texts) > 0) {
      val_accuracy <- evaluate_classifier(
        model, tokenizer, val_texts, val_labels,
        batch_size, classifier$model$device
      )
      message(sprintf(
        "Epoch %d/%d - Loss: %.4f - Val Accuracy: %.4f",
        epoch, epochs, avg_loss, val_accuracy
      ))
    } else {
      message(sprintf(
        "Epoch %d/%d - Loss: %.4f",
        epoch, epochs, avg_loss
      ))
    }
  }

  classifier$is_trained <- TRUE
  classifier
}

# Internal evaluation function
evaluate_classifier <- function(model, tokenizer, texts, labels, batch_size, device) {
  model$eval()
  correct <- 0
  total <- 0

  n_batches <- ceiling(length(texts) / batch_size)

  for (i in seq_len(n_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(texts))

    batch_texts <- texts[start_idx:end_idx]
    batch_labels <- labels[start_idx:end_idx]

    encodings <- tokenizer(
      batch_texts,
      padding = TRUE,
      truncation = TRUE,
      return_tensors = "pt"
    )

    if (device == "cuda") {
      encodings$input_ids <- encodings$input_ids$to(torch$device("cuda"))
      encodings$attention_mask <- encodings$attention_mask$to(torch$device("cuda"))
    }

    with(torch$no_grad(), {
      outputs <- model(
        input_ids = encodings$input_ids,
        attention_mask = encodings$attention_mask
      )

      predictions <- torch$argmax(outputs$logits, dim = -1L)$cpu()$numpy()
      correct <- correct + sum(predictions == batch_labels)
      total <- total + length(batch_labels)
    })
  }

  model$train()
  correct / total
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
#' @examples
#' \dontrun{
#' predictions <- granite_predict(classifier, new_data, text)
#' }
granite_predict <- function(
  classifier,
  data,
  text_col = text,
  type = c("class", "prob"),
  batch_size = 32
) {
  type <- match.arg(type)

  if (!classifier$is_trained) {
    warning("Classifier has not been trained yet. Predictions may be random.")
  }

  text_col <- rlang::enquo(text_col)
  text_col_name <- rlang::as_name(text_col)

  texts <- dplyr::pull(data, !!text_col)

  model <- classifier$model$model
  tokenizer <- classifier$tokenizer$tokenizer

  model$eval()

  predictions_list <- list()
  n_batches <- ceiling(length(texts) / batch_size)

  for (i in seq_len(n_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(texts))
    batch_texts <- texts[start_idx:end_idx]

    encodings <- tokenizer(
      batch_texts,
      padding = TRUE,
      truncation = TRUE,
      return_tensors = "pt"
    )

    if (classifier$model$device == "cuda") {
      encodings$input_ids <- encodings$input_ids$to(torch$device("cuda"))
      encodings$attention_mask <- encodings$attention_mask$to(torch$device("cuda"))
    }

    with(torch$no_grad(), {
      outputs <- model(
        input_ids = encodings$input_ids,
        attention_mask = encodings$attention_mask
      )

      if (type == "class") {
        preds <- torch$argmax(outputs$logits, dim = -1L)$cpu()$numpy()
        predictions_list[[i]] <- preds
      } else {
        probs <- torch$nn$functional$softmax(outputs$logits, dim = -1L)$cpu()$numpy()
        predictions_list[[i]] <- probs
      }
    })
  }

  if (type == "class") {
    predictions <- unlist(predictions_list)
    dplyr::mutate(data, prediction = predictions)
  } else {
    all_probs <- do.call(rbind, predictions_list)
    prob_df <- as.data.frame(all_probs)
    names(prob_df) <- paste0("prob_", seq_len(ncol(prob_df)))
    dplyr::bind_cols(data, prob_df)
  }
}
