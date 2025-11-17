#' Create a Mixture of Experts (MoE) text classifier
#'
#' Creates a text classifier using a Mixture of Experts architecture. This advanced
#' approach uses multiple specialized expert networks whose outputs are dynamically
#' weighted by a gating network, potentially improving performance on multi-class tasks.
#'
#' @param num_labels Number of output classes
#' @param num_experts Number of expert networks (default: 4 for multi-class, 3 for binary)
#' @param data Optional training data frame to infer num_labels from
#' @param label_col Optional label column name (unquoted) to infer num_labels from
#' @param model_name Model identifier from Hugging Face Hub
#' @param device Device to use ("cpu" or "cuda"). If NULL, automatically detects GPU availability.
#' @param freeze_backbone Whether to freeze the pretrained backbone (default: FALSE for MoE)
#' @param hidden_dim Hidden dimension for expert networks (default: backbone_size)
#' @param dropout Dropout probability for expert networks (default: 0.2)
#' @param expert_depth Number of layers per expert network (default: 2)
#' @return A MoE classifier object with model and tokenizer
#'
#' @details
#' The Mixture of Experts (MoE) architecture provides several advantages:
#' \itemize{
#'   \item Multiple specialized experts learn different aspects of the task
#'   \item Dynamic gating network weights experts per input
#'   \item Can improve accuracy on complex multi-class problems
#'   \item Load balancing encourages diverse expert usage
#' }
#'
#' The architecture consists of:
#' \itemize{
#'   \item Pretrained backbone (usually unfrozen for MoE to be effective)
#'   \item N expert networks (deeper feed-forward heads)
#'   \item Gating network that learns to weight experts
#'   \item Load balancing loss to encourage expert diversity
#' }
#'
#' **Important**: MoE works best with freeze_backbone=FALSE. With frozen backbone,
#' the standard classifier often performs similarly or better due to simpler optimization.
#' MoE requires more training data and compute but can achieve higher accuracy on
#' complex multi-class tasks when properly tuned.
#'
#' @export
#' @seealso \code{\link{train_moe}}, \code{\link{predict.moe_classifier}}
#' @examplesIf requireNamespace("transformers")
#' # Create MoE classifier for emotion detection (6 classes)
#' clf <- moe_classifier(num_labels = 6, num_experts = 4)
#'
#' # Binary classification with 3 experts
#' clf_binary <- moe_classifier(num_labels = 2, num_experts = 3)
#'
#' # Infer from data
#' data <- tibble::tibble(text = c("a", "b", "c"), label = c("joy", "sad", "angry"))
#' clf <- moe_classifier(data = data, label_col = label)
moe_classifier <- function(
  num_labels = NULL,
  num_experts = NULL,
  data = NULL,
  label_col = NULL,
  model_name = "ibm-granite/granite-embedding-english-r2",
  device = NULL,
  freeze_backbone = FALSE,
  hidden_dim = NULL,
  dropout = 0.2,
  expert_depth = 2
) {
  device_was_null <- is.null(device)
  if (device_was_null) {
    torch <- reticulate::import("torch", delay_load = TRUE)
    device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  }
  if (device_was_null) {
    cli::cli_alert_info("Using device: {device}")
  }

  if (is.null(num_labels) && !is.null(data) && !is.null(rlang::enquo(label_col))) {
    label_col_quo <- rlang::enquo(label_col)
    labels <- dplyr::pull(data, !!label_col_quo)
    num_labels <- length(unique(labels))
    cli::cli_alert_info("Inferred num_labels = {num_labels} from data")
  }

  if (is.null(num_labels)) {
    stop("num_labels must be specified or inferred from data")
  }

  if (is.null(num_experts)) {
    num_experts <- if (num_labels > 2) 4 else 3
    cli::cli_alert_info("Using {num_experts} experts (recommended for {num_labels} classes)")
  }

  if (freeze_backbone) {
    cli::cli_alert_warning("MoE with frozen backbone may not improve over standard classifier")
    cli::cli_alert_info("Consider setting freeze_backbone=FALSE for better MoE performance")
  }

  moe_module <- reticulate::import_from_path(
    "moe_classifier",
    path = system.file("python", package = "graniteR")
  )

  model <- if (num_labels <= 3) {
    moe_module$MoETextClassifier(
      model_name = model_name,
      num_experts = as.integer(num_experts),
      num_classes = as.integer(num_labels),
      freeze_backbone = freeze_backbone,
      hidden_dim = if (is.null(hidden_dim)) NULL else as.integer(hidden_dim),
      dropout = dropout
    )
  } else {
    moe_module$MoEEmotionClassifier(
      model_name = model_name,
      num_experts = as.integer(num_experts),
      num_classes = as.integer(num_labels),
      freeze_backbone = freeze_backbone,
      hidden_dim = if (is.null(hidden_dim)) NULL else as.integer(hidden_dim),
      dropout = dropout,
      expert_depth = as.integer(expert_depth)
    )
  }

  model$to(device)
  tokenizer <- granite_tokenizer(model_name)

  all_params <- reticulate::iterate(model$parameters())
  trainable_params <- sum(sapply(all_params, function(p) as.logical(p$requires_grad)))
  total_params <- length(reticulate::iterate(model$parameters()))

  cli::cli_alert_info(
    "Created MoE classifier with {num_experts} experts and {trainable_params} trainable parameters out of {total_params} total"
  )

  structure(
    list(
      model = model,
      tokenizer = tokenizer,
      num_labels = num_labels,
      num_experts = num_experts,
      device = device,
      is_trained = FALSE,
      model_type = "moe"
    ),
    class = c("moe_classifier", "granite_classifier", "granite_model")
  )
}

#' Train a MoE text classifier
#'
#' Trains a Mixture of Experts classifier on labeled data. The training process
#' includes both classification loss and load balancing loss to encourage diverse
#' expert usage.
#'
#' @param classifier MoE classifier object
#' @param data Training data frame
#' @param text_col Column name containing text (unquoted or string)
#' @param label_col Column name containing labels (unquoted or string)
#' @param epochs Number of training epochs
#' @param batch_size Batch size for training
#' @param learning_rate Learning rate for optimizer
#' @param validation_split Fraction of data to use for validation
#' @param verbose Whether to print training progress (default: TRUE)
#' @return Updated classifier object with trained model
#'
#' @details
#' The MoE training process optimizes:
#' \itemize{
#'   \item Classification loss (cross-entropy)
#'   \item Load balancing loss (encourages diverse expert usage)
#' }
#'
#' During training, the gating network learns to route different inputs to
#' different experts, potentially allowing specialization (e.g., one expert
#' for positive sentiment, another for negative).
#'
#' @export
#' @seealso \code{\link{moe_classifier}}, \code{\link{predict.moe_classifier}}
#' @examplesIf requireNamespace("transformers")
#' library(dplyr)
#'
#' data <- tibble::tibble(
#'   text = c("I feel happy", "I am sad", "I love this"),
#'   emotion = c("joy", "sadness", "love")
#' )
#' clf <- moe_classifier(num_labels = 3, num_experts = 3) |>
#'   train_moe(data, text, emotion, epochs = 3)
train_moe <- function(
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

  model <- classifier$model
  tokenizer <- classifier$tokenizer$tokenizer

  optimizer <- torch$optim$AdamW(
    model$parameters(),
    lr = learning_rate
  )

  model$train()

  if (verbose) {
    cli::cli_alert_info("Training MoE with {classifier$num_experts} experts on {length(train_texts)} samples, validating on {length(val_texts)} samples")
  }

  training_start_time <- Sys.time()
  total_batches <- ceiling(length(train_texts) / batch_size) * epochs
  batch_times <- numeric()
  prev_loss <- NULL
  prev_accuracy <- NULL

  tryCatch({
    for (epoch in seq_len(epochs)) {
      total_loss <- 0
      total_cls_loss <- 0
      total_lb_loss <- 0
      n_batches <- ceiling(length(train_texts) / batch_size)
      epoch_start_time <- Sys.time()
      avg_loss <- 0
      prev_batch_loss <- NULL
      eta_str <- "calculating..."
      loss_delta_str <- ""

      if (verbose) {
        cli::cli_progress_bar(
          format = "Epoch {epoch}/{epochs} | Batch {cli::pb_current}/{cli::pb_total} | Loss: {sprintf('%.4f', avg_loss)}{loss_delta_str} | ETA: {eta_str}",
          total = n_batches,
          clear = FALSE
        )
      }

      for (i in seq_len(n_batches)) {
        batch_start <- Sys.time()
        start_idx <- (i - 1) * batch_size + 1
        end_idx <- min(i * batch_size, length(train_texts))

        encodings <- tokenizer(
          train_texts[start_idx:end_idx],
          padding = TRUE,
          truncation = TRUE,
          return_tensors = "pt"
        )

        labels_tensor <- torch$tensor(as.integer(train_labels[start_idx:end_idx]))
        moved <- to_device(encodings, labels_tensor, classifier$device)

        optimizer$zero_grad()
        outputs <- model(
          input_ids = moved$encodings$input_ids,
          attention_mask = moved$encodings$attention_mask,
          labels = moved$labels
        )

        outputs$loss$backward()
        optimizer$step()
        total_loss <- total_loss + outputs$loss$item()
        total_cls_loss <- total_cls_loss + outputs$classification_loss$item()
        total_lb_loss <- total_lb_loss + outputs$load_balance_loss$item()

        batch_times <- c(batch_times, as.numeric(difftime(Sys.time(), batch_start, units = "secs")))

        if (verbose) {
          avg_loss <- total_loss / i

          if (!is.null(prev_batch_loss) && i > 1) {
            delta_val <- ((prev_batch_loss - avg_loss) / prev_batch_loss) * 100
            loss_delta_str <- sprintf(" (%+.1f%%)", delta_val)
          } else {
            loss_delta_str <- ""
          }
          prev_batch_loss <- avg_loss

          recent_batch_times <- tail(batch_times, min(10, length(batch_times)))
          avg_batch_time <- mean(recent_batch_times)

          current_batch <- (epoch - 1) * n_batches + i
          remaining_batches <- total_batches - current_batch
          eta_secs <- avg_batch_time * remaining_batches

          eta_str <- if (eta_secs < 60) {
            sprintf("%.0fs", eta_secs)
          } else if (eta_secs < 3600) {
            sprintf("%.1fm", eta_secs / 60)
          } else {
            sprintf("%.1fh", eta_secs / 3600)
          }

          cli::cli_progress_update(set = i, .envir = environment())
        }
      }

      if (verbose) {
        cli::cli_progress_done()
      }

      if (verbose) {
        epoch_time <- as.numeric(difftime(Sys.time(), epoch_start_time, units = "secs"))
        current_loss <- total_loss / n_batches
        avg_cls_loss <- total_cls_loss / n_batches
        avg_lb_loss <- total_lb_loss / n_batches

        if (length(val_texts) > 0) {
          val_accuracy <- evaluate_moe_classifier(
            model, tokenizer, val_texts, val_labels,
            batch_size, classifier$device
          )

          loss_delta <- if (!is.null(prev_loss)) {
            delta_val <- ((prev_loss - current_loss) / prev_loss) * 100
            sprintf(" (%+.1f%%)", delta_val)
          } else {
            ""
          }

          acc_delta <- if (!is.null(prev_accuracy)) {
            delta_val <- (val_accuracy - prev_accuracy) * 100
            sprintf(" (%+.1f%%)", delta_val)
          } else {
            ""
          }

          cli::cli_alert_info("Epoch {epoch}/{epochs} - Loss: {round(current_loss, 4)}{loss_delta} (cls: {round(avg_cls_loss, 4)}, lb: {round(avg_lb_loss, 4)}) - Val Acc: {round(val_accuracy, 4)}{acc_delta} - Time: {round(epoch_time, 1)}s")

          prev_loss <- current_loss
          prev_accuracy <- val_accuracy
        } else {
          loss_delta <- if (!is.null(prev_loss)) {
            delta_val <- ((prev_loss - current_loss) / prev_loss) * 100
            sprintf(" (%+.1f%%)", delta_val)
          } else {
            ""
          }

          cli::cli_alert_info("Epoch {epoch}/{epochs} - Loss: {round(current_loss, 4)}{loss_delta} (cls: {round(avg_cls_loss, 4)}, lb: {round(avg_lb_loss, 4)}) - Time: {round(epoch_time, 1)}s")
          prev_loss <- current_loss
        }
      }
    }

    if (verbose) {
      total_time <- as.numeric(difftime(Sys.time(), training_start_time, units = "secs"))
      time_str <- if (total_time < 60) {
        sprintf("%.1fs", total_time)
      } else if (total_time < 3600) {
        sprintf("%.1fm", total_time / 60)
      } else {
        sprintf("%.2fh", total_time / 3600)
      }
      cli::cli_alert_success("MoE training complete in {time_str}")
    }

    classifier$is_trained <- TRUE
  }, interrupt = function(e) {
    if (verbose) {
      cli::cli_alert_warning("Training interrupted by user")
    }
    classifier$is_trained <- TRUE
  }, error = function(e) {
    if (verbose) {
      cli::cli_alert_danger("Training failed: {e$message}")
    }
    classifier$is_trained <- FALSE
  })

  classifier
}

#' Internal evaluation function for MoE classifier
#' @keywords internal
evaluate_moe_classifier <- function(model, tokenizer, texts, labels, batch_size, device) {
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
      result <- model(
        input_ids = moved$encodings$input_ids,
        attention_mask = moved$encodings$attention_mask
      )
      logits <- result[[1]]
      predictions <- torch$argmax(logits, dim = -1L)$cpu()$numpy()
      correct <- correct + sum(predictions == labels[start_idx:end_idx])
    })
  }

  model$train()
  correct / length(labels)
}

#' Make predictions with a trained MoE classifier
#'
#' @param classifier Trained MoE classifier object
#' @param data Data frame containing text to classify
#' @param text_col Column name containing text (unquoted or string)
#' @param type Type of prediction ("class", "prob", or "expert_weights")
#' @param batch_size Batch size for prediction
#' @param ... Additional arguments (unused, for S3 consistency)
#' @return Data frame with predictions
#'
#' @details
#' When type = "expert_weights", returns the gating network weights showing
#' which experts were most active for each input. This can provide insights
#' into expert specialization.
#'
#' @export
#' @seealso \code{\link{moe_classifier}}, \code{\link{train_moe}}
predict.moe_classifier <- function(
  classifier,
  data,
  text_col,
  type = c("class", "prob", "expert_weights"),
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

  model <- classifier$model
  tokenizer <- classifier$tokenizer$tokenizer

  model$eval()
  predictions_list <- list()
  gate_weights_list <- list()
  n_batches <- ceiling(length(texts) / batch_size)

  pred_start_time <- Sys.time()
  samples_processed <- 0
  total_samples <- length(texts)
  eta_str <- "calculating..."
  progress_pct <- "0%"
  speed_str <- ""

  cli::cli_progress_bar(
    format = "Predicting | {progress_pct} | Batch {cli::pb_current}/{cli::pb_total} | {speed_str} | ETA: {eta_str}",
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

    moved <- to_device(encodings, device = classifier$device)

    with(torch$no_grad(), {
      result <- model(
        input_ids = moved$encodings$input_ids,
        attention_mask = moved$encodings$attention_mask
      )

      logits <- result[[1]]
      gate_weights <- result[[2]]

      predictions_list[[i]] <- if (type == "class") {
        torch$argmax(logits, dim = -1L)$cpu()$numpy()
      } else {
        torch$nn$functional$softmax(logits, dim = -1L)$cpu()$numpy()
      }

      gate_weights_list[[i]] <- gate_weights$cpu()$numpy()
    })

    samples_processed <- end_idx
    progress_pct <- sprintf("%.1f%%", (samples_processed / total_samples) * 100)

    elapsed <- as.numeric(difftime(Sys.time(), pred_start_time, units = "secs"))
    samples_per_sec <- samples_processed / elapsed
    speed_str <- sprintf("%.0f samples/s", samples_per_sec)

    remaining_samples <- total_samples - samples_processed
    eta_secs <- remaining_samples / samples_per_sec

    eta_str <- if (eta_secs < 60) {
      sprintf("%.0fs", eta_secs)
    } else if (eta_secs < 3600) {
      sprintf("%.1fm", eta_secs / 60)
    } else {
      sprintf("%.1fh", eta_secs / 3600)
    }

    cli::cli_progress_update(set = i, .envir = environment())
  }

  cli::cli_progress_done()

  total_time <- as.numeric(difftime(Sys.time(), pred_start_time, units = "secs"))
  time_str <- if (total_time < 60) {
    sprintf("%.1fs", total_time)
  } else if (total_time < 3600) {
    sprintf("%.1fm", total_time / 60)
  } else {
    sprintf("%.2fh", total_time / 3600)
  }
  avg_speed <- total_samples / total_time
  cli::cli_alert_success("Predicted {total_samples} samples in {time_str} ({sprintf('%.0f', avg_speed)} samples/s)")

  if (type == "expert_weights") {
    gate_df <- as.data.frame(do.call(rbind, gate_weights_list))
    names(gate_df) <- paste0("expert_", seq_len(ncol(gate_df)))
    dplyr::bind_cols(data, gate_df)
  } else if (type == "class") {
    dplyr::mutate(data, prediction = unlist(predictions_list))
  } else {
    prob_df <- as.data.frame(do.call(rbind, predictions_list))
    names(prob_df) <- paste0("prob_", seq_len(ncol(prob_df)))
    dplyr::bind_cols(data, prob_df)
  }
}
