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
#' @param cv_folds Number of cross-validation folds. If NULL (default), automatically
#'   determined: 5 folds for <2K samples, 3 for 2-10K, 2 for >10K.
#' @param max_time_minutes Maximum time budget in minutes (default: 30)
#' @param device Device to use ("cpu" or "cuda"). Auto-detected if NULL.
#' @param verbose Show detailed progress (default: TRUE)
#'
#' @return A trained classifier (standard or MoE) ready for predictions
#'
#' @details
#' **Data-Driven Strategy**:
#'
#' Computes `samples_per_class` and `complexity_score`, then decides:
#' - **Always**: Frozen baseline (fast, prevents overfitting)
#' - **Add fine-tuning if**: 50+ samples/class AND fits time budget
#' - **Add MoE if**: 4+ classes, 200+ samples/class, complexity > 12, AND fits budget
#'
#' Model selection via cross-validation:
#' 1. Adaptive CV folds (fewer for larger datasets)
#' 2. Tests all candidates that fit in time budget
#' 3. Selects best based on validation accuracy
#' 4. Trains final model on all data
#'
#' Time budget management:
#' - Estimates training time from dataset size
#' - Only tests approaches that fit (with safety margin)
#' - Early stopping if budget exhausted during CV
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
  cv_folds = NULL,
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

  # Adaptive CV folds: fewer folds for larger datasets
  if (is.null(cv_folds)) {
    cv_folds <- if (n_samples < 2000) {
      5
    } else if (n_samples < 10000) {
      3
    } else {
      2  # Just a train/val split for very large datasets
    }
  }

  if (verbose) {
    cli::cli_h1("Auto-Classify: Automatic Model Selection")
    cli::cli_alert_info("Dataset: {scales::comma(n_samples)} samples, {num_labels} classes")
    cli::cli_alert_info("Time budget: {max_time_minutes} minutes")
    cli::cli_alert_info("Cross-validation: {cv_folds} folds")
  }

  strategy <- select_strategy(n_samples, num_labels, max_time_minutes, cv_folds, verbose)

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
select_strategy <- function(n_samples, num_labels, max_time_minutes, cv_folds, verbose) {
  # Compute data-driven metrics
  samples_per_class <- n_samples / num_labels
  complexity_score <- num_labels * log10(pmax(n_samples, 100))

  # Estimate single-fold training time (minutes) - empirically calibrated
  # These are rough estimates: frozen is fast, fine-tuning is 10x slower, MoE is 15x slower
  time_per_epoch_frozen <- n_samples / 60000  # ~1000 samples/sec
  time_per_epoch_finetuned <- n_samples / 6000  # ~100 samples/sec
  time_per_epoch_moe <- n_samples / 4000  # ~67 samples/sec

  candidates <- list()

  # Always include frozen baseline
  frozen_epochs <- pmin(10, pmax(3, ceiling(5000 / samples_per_class)))
  candidates[[1]] <- list(
    type = "standard",
    freeze_backbone = TRUE,
    learning_rate = 1e-3,
    epochs = frozen_epochs,
    description = "Standard frozen"
  )

  # Add fine-tuned if enough data per class
  # Rule: need at least 50 samples/class to benefit from fine-tuning
  if (samples_per_class >= 50) {
    finetuned_epochs <- pmin(5, pmax(2, ceiling(3000 / samples_per_class)))
    # CV cost for both models
    cv_cost <- cv_folds * (time_per_epoch_frozen * frozen_epochs + time_per_epoch_finetuned * finetuned_epochs)
    final_cost <- time_per_epoch_finetuned * finetuned_epochs

    if (cv_cost + final_cost <= max_time_minutes) {
      candidates[[length(candidates) + 1]] <- list(
        type = "standard",
        freeze_backbone = FALSE,
        learning_rate = 2e-5,
        epochs = finetuned_epochs,
        description = "Standard fine-tuned"
      )
    }
  }

  # Add MoE if: complex multi-class task with enough data
  # Rule: need 4+ classes, 200+ samples/class, and high enough complexity score
  if (num_labels >= 4 && samples_per_class >= 200 && complexity_score > 12) {
    moe_epochs <- 3
    # Estimate total CV cost with all candidates
    n_candidates_with_moe <- length(candidates) + 1
    cv_cost <- cv_folds * n_candidates_with_moe * time_per_epoch_moe * moe_epochs
    final_cost <- time_per_epoch_moe * moe_epochs

    if (cv_cost + final_cost <= max_time_minutes) {
      candidates[[length(candidates) + 1]] <- list(
        type = "moe",
        freeze_backbone = FALSE,
        learning_rate = 2e-5,
        epochs = moe_epochs,
        num_experts = pmin(8, pmax(2, ceiling(num_labels / 2))),
        description = "MoE fine-tuned"
      )
    }
  }

  # Determine strategy name and reason
  name <- if (length(candidates) == 1) {
    "Frozen Baseline"
  } else if (any(sapply(candidates, function(x) x$type == "moe"))) {
    "Multi-Strategy CV"
  } else {
    "Frozen vs Fine-tuned CV"
  }

  reason <- sprintf(
    "%.0f samples/class, complexity=%.1f, %d candidates",
    samples_per_class, complexity_score, length(candidates)
  )

  list(
    name = name,
    reason = reason,
    candidates = candidates
  )
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
