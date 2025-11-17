#' Automatic text classification with model selection
#'
#' This is the easy entry point for text classification. It automatically:
#' - Determines optimal model architecture based on dataset size and classes
#' - Performs k-fold cross-validation to select best approach
#' - Trains final model on full dataset
#' - Returns ready-to-use classifier
#'
#' @param data Training data frame
#' @param text_col Column containing text (unquoted)
#' @param label_col Column containing labels (unquoted)
#' @param cv_folds Number of cross-validation folds (default: 5)
#' @param max_time_minutes Maximum time budget in minutes (default: 30)
#' @param device Device to use ("cpu" or "cuda"). Auto-detected if NULL.
#' @param verbose Show detailed progress (default: TRUE)
#'
#' @return A trained classifier (standard or MoE) ready for predictions
#'
#' @details
#' The function automatically decides between:
#' - **Frozen head-only**: Small datasets (<5K), binary tasks
#' - **Frozen standard**: Medium datasets (5-20K)
#' - **Full fine-tune standard**: Large datasets (20-50K)
#' - **MoE with fine-tuning**: Very large datasets (50K+) with 4+ classes
#'
#' Model selection via cross-validation:
#' 1. Splits data into k folds
#' 2. Tests multiple approaches (frozen, unfrozen, MoE if applicable)
#' 3. Selects best based on validation accuracy
#' 4. Trains final model on all data
#'
#' Time budget management:
#' - If time is limited, skips expensive approaches (MoE, full fine-tuning)
#' - Provides early stopping if budget is exhausted
#' - Returns best model trained within budget
#'
#' @export
#' @seealso \code{\link{classifier}}, \code{\link{moe_classifier}}
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#'
#' # Simple usage - just data and columns
#' clf <- auto_classify(
#'   data = emotion_sample,
#'   text_col = text,
#'   label_col = label
#' )
#'
#' # Make predictions
#' predictions <- predict(clf, test_data, text)
#'
#' # With time budget
#' clf <- auto_classify(
#'   data = emotion_full,
#'   text_col = text,
#'   label_col = label,
#'   max_time_minutes = 15
#' )
#' }
auto_classify <- function(
  data,
  text_col,
  label_col,
  cv_folds = 5,
  max_time_minutes = 30,
  device = NULL,
  verbose = TRUE
) {
  start_time <- Sys.time()
  text_col <- rlang::enquo(text_col)
  label_col <- rlang::enquo(label_col)

  texts <- dplyr::pull(data, !!text_col)
  labels <- dplyr::pull(data, !!label_col)

  if (is.factor(labels) || is.character(labels)) {
    label_mapping <- sort(unique(labels))
    labels <- as.integer(factor(labels, levels = label_mapping)) - 1L
  }

  num_labels <- length(unique(labels))
  n_samples <- length(texts)

  if (verbose) {
    cli::cli_h1("Auto-Classify: Automatic Model Selection")
    cli::cli_alert_info("Dataset: {scales::comma(n_samples)} samples, {num_labels} classes")
    cli::cli_alert_info("Time budget: {max_time_minutes} minutes")
    cli::cli_alert_info("Cross-validation: {cv_folds} folds")
  }

  strategy <- select_strategy(n_samples, num_labels, max_time_minutes, verbose)

  if (verbose) {
    cli::cli_h2("Selected Strategy")
    cli::cli_alert_success("Approach: {strategy$name}")
    cli::cli_alert_info("Reason: {strategy$reason}")
    if (length(strategy$candidates) > 1) {
      cli::cli_alert_info("Will test {length(strategy$candidates)} configurations via CV")
    }
  }

  best_config <- if (length(strategy$candidates) > 1) {
    perform_cv_selection(
      texts, labels, num_labels,
      strategy$candidates, cv_folds,
      start_time, max_time_minutes,
      device, verbose
    )
  } else {
    strategy$candidates[[1]]
  }

  if (verbose) {
    cli::cli_h2("Training Final Model")
    cli::cli_alert_info("Configuration: {best_config$description}")
  }

  final_model <- train_final_model(
    texts, labels, num_labels,
    best_config, device, verbose
  )

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  if (verbose) {
    cli::cli_alert_success("Total time: {round(elapsed, 1)} minutes")
    cli::cli_alert_success("Model ready for predictions!")
  }

  final_model
}

#' Select training strategy based on dataset characteristics
#' @keywords internal
select_strategy <- function(n_samples, num_labels, max_time_minutes, verbose) {
  is_binary <- num_labels == 2

  if (n_samples < 5000) {
    list(
      name = "Frozen Head-Only (Fast)",
      reason = "Small dataset - frozen backbone prevents overfitting",
      candidates = list(
        list(
          type = "standard",
          freeze_backbone = TRUE,
          learning_rate = 1e-3,
          epochs = 5,
          description = "Standard classifier, frozen backbone"
        )
      )
    )
  } else if (n_samples < 20000) {
    candidates <- list(
      list(
        type = "standard",
        freeze_backbone = TRUE,
        learning_rate = 1e-3,
        epochs = 5,
        description = "Standard frozen"
      )
    )

    if (max_time_minutes >= 20) {
      candidates[[2]] <- list(
        type = "standard",
        freeze_backbone = FALSE,
        learning_rate = 2e-5,
        epochs = 3,
        description = "Standard fine-tuned"
      )
    }

    list(
      name = "Standard Classifier with CV",
      reason = "Medium dataset - test frozen vs fine-tuned",
      candidates = candidates
    )
  } else if (n_samples < 50000 || is_binary) {
    list(
      name = "Full Fine-Tuning",
      reason = if (is_binary) "Binary task - standard is sufficient" else "Large dataset - full fine-tuning beneficial",
      candidates = list(
        list(
          type = "standard",
          freeze_backbone = FALSE,
          learning_rate = 2e-5,
          epochs = 4,
          description = "Standard full fine-tune"
        )
      )
    )
  } else {
    candidates <- list(
      list(
        type = "standard",
        freeze_backbone = FALSE,
        learning_rate = 2e-5,
        epochs = 3,
        description = "Standard fine-tuned"
      )
    )

    if (max_time_minutes >= 45 && num_labels >= 4) {
      candidates[[2]] <- list(
        type = "moe",
        freeze_backbone = FALSE,
        learning_rate = 2e-5,
        epochs = 3,
        num_experts = min(4, num_labels),
        description = "MoE fine-tuned"
      )
    }

    list(
      name = "Advanced: Standard vs MoE",
      reason = "Very large multi-class dataset - test if MoE helps",
      candidates = candidates
    )
  }
}

#' Perform cross-validation model selection
#' @keywords internal
perform_cv_selection <- function(
  texts, labels, num_labels,
  candidates, cv_folds,
  start_time, max_time_minutes,
  device, verbose
) {
  if (verbose) {
    cli::cli_h2("Cross-Validation Model Selection")
  }

  n <- length(texts)
  fold_indices <- sample(rep(1:cv_folds, length.out = n))

  results <- list()

  for (i in seq_along(candidates)) {
    config <- candidates[[i]]

    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
    if (elapsed >= max_time_minutes * 0.8) {
      if (verbose) {
        cli::cli_alert_warning("Approaching time budget - stopping CV early")
      }
      break
    }

    if (verbose) {
      cli::cli_alert_info("Testing: {config$description} ({i}/{length(candidates)})")
    }

    fold_accuracies <- numeric(cv_folds)

    for (fold in 1:cv_folds) {
      val_idx <- which(fold_indices == fold)
      train_idx <- which(fold_indices != fold)

      fold_accuracy <- train_and_evaluate_fold(
        texts[train_idx], labels[train_idx],
        texts[val_idx], labels[val_idx],
        num_labels, config, device, verbose = FALSE
      )

      fold_accuracies[fold] <- fold_accuracy
    }

    mean_accuracy <- mean(fold_accuracies)
    sd_accuracy <- sd(fold_accuracies)

    results[[i]] <- list(
      config = config,
      mean_accuracy = mean_accuracy,
      sd_accuracy = sd_accuracy
    )

    if (verbose) {
      cli::cli_alert_success(
        "{config$description}: {sprintf('%.4f', mean_accuracy)} ± {sprintf('%.4f', sd_accuracy)}"
      )
    }
  }

  best_idx <- which.max(sapply(results, function(x) x$mean_accuracy))
  best_result <- results[[best_idx]]

  if (verbose) {
    cli::cli_alert_success("Best configuration: {best_result$config$description}")
    cli::cli_alert_success("CV accuracy: {sprintf('%.4f', best_result$mean_accuracy)}")
  }

  best_result$config
}

#' Train and evaluate a single fold
#' @keywords internal
train_and_evaluate_fold <- function(
  train_texts, train_labels,
  val_texts, val_labels,
  num_labels, config, device, verbose
) {
  tryCatch({
    if (config$type == "standard") {
      clf <- classifier(
        num_labels = num_labels,
        device = device,
        freeze_backbone = config$freeze_backbone
      )
    } else {
      clf <- moe_classifier(
        num_labels = num_labels,
        num_experts = config$num_experts,
        device = device,
        freeze_backbone = config$freeze_backbone
      )
    }

    train_data <- data.frame(
      text = train_texts,
      label = train_labels,
      stringsAsFactors = FALSE
    )

    if (config$type == "standard") {
      clf <- train(
        clf, train_data, text, label,
        epochs = config$epochs,
        batch_size = 8,
        learning_rate = config$learning_rate,
        validation_split = 0,
        verbose = verbose
      )
    } else {
      clf <- train_moe(
        clf, train_data, text, label,
        epochs = config$epochs,
        batch_size = 8,
        learning_rate = config$learning_rate,
        validation_split = 0,
        verbose = verbose
      )
    }

    val_data <- data.frame(
      text = val_texts,
      label = val_labels,
      stringsAsFactors = FALSE
    )

    preds <- predict(clf, val_data, text, type = "class")
    accuracy <- mean(preds$prediction == preds$label)

    accuracy
  }, error = function(e) {
    if (verbose) {
      cli::cli_alert_danger("Fold failed: {e$message}")
    }
    return(0)
  })
}

#' Train final model on all data
#' @keywords internal
train_final_model <- function(
  texts, labels, num_labels,
  config, device, verbose
) {
  if (config$type == "standard") {
    clf <- classifier(
      num_labels = num_labels,
      device = device,
      freeze_backbone = config$freeze_backbone
    )
  } else {
    clf <- moe_classifier(
      num_labels = num_labels,
      num_experts = config$num_experts,
      device = device,
      freeze_backbone = config$freeze_backbone
    )
  }

  train_data <- data.frame(
    text = texts,
    label = labels,
    stringsAsFactors = FALSE
  )

  if (config$type == "standard") {
    clf <- train(
      clf, train_data, text, label,
      epochs = config$epochs,
      batch_size = 8,
      learning_rate = config$learning_rate,
      validation_split = 0.1,
      verbose = verbose
    )
  } else {
    clf <- train_moe(
      clf, train_data, text, label,
      epochs = config$epochs,
      batch_size = 8,
      learning_rate = config$learning_rate,
      validation_split = 0.1,
      verbose = verbose
    )
  }

  clf
}
