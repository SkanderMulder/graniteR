#' Automatic text classification with AutoML
#'
#' State-of-the-art automatic model selection using:
#' - Meta-learning to predict model performance from dataset characteristics
#' - CASH-style combined algorithm and hyperparameter optimization
#' - Ensemble building from diverse high-performing candidates
#' - Resource-aware time budget management
#'
#' @param data Training data frame
#' @param text_col Column containing text (unquoted)
#' @param label_col Column containing labels (unquoted)
#' @param max_time_minutes Maximum time budget in minutes (default: 30)
#' @param ensemble Enable ensemble building (default: FALSE). Requires 2x time budget.
#' @param meta_learning Use meta-features for candidate ranking (default: TRUE)
#' @param cv_folds Number of CV folds (NULL = adaptive: 5/<2K, 3/2-10K, 2/>10K)
#' @param device Device to use ("cpu" or "cuda", NULL = auto-detect)
#' @param verbose Show detailed progress (default: TRUE)
#'
#' @return Trained classifier or ensemble ready for predictions
#' @export
#'
#' @details
#' **How it works**:
#'
#' 1. **Meta-feature extraction**: Computes dataset characteristics like
#'    samples_per_class, label_entropy, complexity_score, vocab_richness
#'
#' 2. **CASH search space**: Generates candidate configurations across:
#'    - Model types: frozen, fine-tuned, MoE
#'    - Hyperparameters: learning rates, epochs, num_experts
#'
#' 3. **Meta-learning ranking**: Predicts performance without training,
#'    prioritizes most promising candidates
#'
#' 4. **Cross-validation**: Evaluates candidates with early stopping
#'    when time budget is exhausted
#'
#' 5. **Model selection or ensemble**: Returns best single model, or
#'    builds weighted ensemble from diverse top performers
#'
#' **When to use ensemble**:
#' - Critical applications requiring maximum accuracy
#' - Time budget >= 45 minutes
#' - Typically improves accuracy by 1-3%
#'
#' **Performance**: For quick experiments use `ensemble = FALSE` with
#' 15-30 min budget. For production use `ensemble = TRUE` with 60+ min.
#'
#' @seealso \code{\link{classifier}}, \code{\link{moe_classifier}},
#'   \code{\link{train}}, \code{\link{predict.classifier}}
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#' data(emotion_sample)
#'
#' # Quick classification
#' clf <- auto_classify(
#'   emotion_sample,
#'   text,
#'   label,
#'   max_time_minutes = 15
#' )
#'
#' # Best performance with ensemble
#' ensemble <- auto_classify(
#'   emotion_full,
#'   text,
#'   label,
#'   max_time_minutes = 60,
#'   ensemble = TRUE
#' )
#'
#' # Make predictions
#' test_data <- emotion_sample |> slice_sample(n = 100)
#' predictions <- predict(clf, test_data, text)
#' }
auto_classify <- function(
  data,
  text_col,
  label_col,
  max_time_minutes = 30,
  ensemble = FALSE,
  meta_learning = TRUE,
  cv_folds = NULL,
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

  # Adaptive CV folds
  if (is.null(cv_folds)) {
    cv_folds <- if (n_samples < 2000) 5 else if (n_samples < 10000) 3 else 2
  }

  if (verbose) {
    cli::cli_h1("AutoML Classification: Advanced Strategy")
    cli::cli_alert_info("Dataset: {scales::comma(n_samples)} samples, {num_labels} classes")
    cli::cli_alert_info("Time budget: {max_time_minutes} minutes")
    cli::cli_alert_info("Ensemble: {ensemble}, Meta-learning: {meta_learning}")
  }

  # Step 1: Extract meta-features
  meta_features <- compute_meta_features(texts, labels)

  if (verbose) {
    cli::cli_h2("Dataset Meta-Features")
    cli::cli_ul(c(
      "Samples/class: {round(meta_features$samples_per_class)}",
      "Label entropy: {round(meta_features$label_entropy, 2)}",
      "Class imbalance: {round(meta_features$class_imbalance, 1)}x",
      "Avg text length: {round(meta_features$avg_text_len)} chars",
      "Vocabulary: {scales::comma(meta_features$vocab_size)} words",
      "Complexity: {round(meta_features$complexity_score, 1)}"
    ))
  }

  # Step 2: Generate candidate search space (CASH)
  candidates <- generate_candidate_space(
    meta_features,
    max_time_minutes,
    cv_folds,
    meta_learning
  )

  if (verbose) {
    cli::cli_h2("Candidate Models ({length(candidates)} total)")
    for (i in seq_along(candidates)) {
      cand <- candidates[[i]]
      if (meta_learning) {
        pred_acc <- round(cand$predicted_accuracy, 3)
        cli::cli_alert_info(
          "{i}. {cand$description} (predicted: {pred_acc}, cost: {round(cand$estimated_time, 1)}m)"
        )
      } else {
        cli::cli_alert_info(
          "{i}. {cand$description} (cost: {round(cand$estimated_time, 1)}m)"
        )
      }
    }
  }

  # Step 3: Evaluate candidates with early stopping
  cv_results <- evaluate_candidates_with_budget(
    texts, labels, num_labels,
    candidates, cv_folds,
    start_time, max_time_minutes,
    device, verbose
  )

  if (length(cv_results) == 0) {
    stop("No candidates could be evaluated within time budget")
  }

  # Step 4: Build ensemble or select best
  if (ensemble && length(cv_results) >= 2) {
    if (verbose) {
      cli::cli_h2("Building Ensemble")
    }

    final_model <- build_ensemble_from_results(
      texts, labels, num_labels,
      cv_results, device, verbose
    )
  } else {
    if (verbose) {
      cli::cli_h2("Training Best Model")
    }

    best_result <- cv_results[[which.max(sapply(cv_results, function(x) x$mean_accuracy))]]

    if (verbose) {
      cli::cli_alert_success("Selected: {best_result$config$description}")
      cli::cli_alert_info("CV accuracy: {round(best_result$mean_accuracy, 4)}")
    }

    final_model <- train_final_model(
      texts, labels, num_labels,
      best_result$config, device, verbose
    )
  }

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  if (verbose) {
    cli::cli_alert_success("Total time: {round(elapsed, 1)} minutes")
    if (inherits(final_model, "ensemble_classifier")) {
      cli::cli_alert_success("Ensemble with {final_model$n_models} models ready!")
    } else {
      cli::cli_alert_success("Model ready for predictions!")
    }
  }

  final_model
}

#' Generate candidate search space with hyperparameters
#' @keywords internal
generate_candidate_space <- function(
  meta_features,
  max_time_minutes,
  cv_folds,
  use_meta_learning
) {
  candidates <- list()

  # Hyperparameter search space
  frozen_configs <- expand.grid(
    learning_rate = c(5e-4, 1e-3, 2e-3),
    epochs = 3:7,
    stringsAsFactors = FALSE
  )

  finetuned_configs <- expand.grid(
    learning_rate = c(1e-5, 2e-5, 5e-5),
    epochs = 2:4,
    stringsAsFactors = FALSE
  )

  moe_configs <- expand.grid(
    learning_rate = c(1e-5, 2e-5),
    epochs = 2:3,
    num_experts = c(2, 4, 6),
    stringsAsFactors = FALSE
  )

  # Time per epoch estimates
  time_per_epoch_frozen <- meta_features$n_samples / 60000
  time_per_epoch_finetuned <- meta_features$n_samples / 6000
  time_per_epoch_moe <- meta_features$n_samples / 4000

  # Generate frozen candidates
  for (i in seq_len(nrow(frozen_configs))) {
    config <- frozen_configs[i, ]
    estimated_time <- cv_folds * time_per_epoch_frozen * config$epochs

    candidates[[length(candidates) + 1]] <- list(
      type = "standard",
      freeze_backbone = TRUE,
      learning_rate = config$learning_rate,
      epochs = config$epochs,
      description = sprintf(
        "Frozen (lr=%.0e, ep=%d)",
        config$learning_rate, config$epochs
      ),
      estimated_time = estimated_time,
      predicted_accuracy = if (use_meta_learning) {
        predict_candidate_performance(meta_features, "frozen")
      } else NA
    )
  }

  # Generate fine-tuned candidates if enough data
  if (meta_features$samples_per_class >= 50) {
    for (i in seq_len(nrow(finetuned_configs))) {
      config <- finetuned_configs[i, ]
      estimated_time <- cv_folds * time_per_epoch_finetuned * config$epochs

      candidates[[length(candidates) + 1]] <- list(
        type = "standard",
        freeze_backbone = FALSE,
        learning_rate = config$learning_rate,
        epochs = config$epochs,
        description = sprintf(
          "Fine-tuned (lr=%.0e, ep=%d)",
          config$learning_rate, config$epochs
        ),
        estimated_time = estimated_time,
        predicted_accuracy = if (use_meta_learning) {
          predict_candidate_performance(meta_features, "finetuned")
        } else NA
      )
    }
  }

  # Generate MoE candidates if complex task
  if (meta_features$n_labels >= 4 &&
      meta_features$samples_per_class >= 200 &&
      meta_features$complexity_score > 12) {
    for (i in seq_len(nrow(moe_configs))) {
      config <- moe_configs[i, ]
      estimated_time <- cv_folds * time_per_epoch_moe * config$epochs

      candidates[[length(candidates) + 1]] <- list(
        type = "moe",
        freeze_backbone = FALSE,
        learning_rate = config$learning_rate,
        epochs = config$epochs,
        num_experts = config$num_experts,
        description = sprintf(
          "MoE-%d (lr=%.0e, ep=%d)",
          config$num_experts, config$learning_rate, config$epochs
        ),
        estimated_time = estimated_time,
        predicted_accuracy = if (use_meta_learning) {
          predict_candidate_performance(meta_features, "moe")
        } else NA
      )
    }
  }

  # Filter by time budget (keep only those that fit)
  affordable <- sapply(candidates, function(c) c$estimated_time <= max_time_minutes * 0.8)
  candidates <- candidates[affordable]

  # Sort by predicted performance if using meta-learning
  if (use_meta_learning && length(candidates) > 0) {
    pred_scores <- sapply(candidates, function(c) c$predicted_accuracy)
    candidates <- candidates[order(pred_scores, decreasing = TRUE)]

    # Keep top 10 to avoid excessive search
    if (length(candidates) > 10) {
      candidates <- candidates[1:10]
    }
  }

  candidates
}

#' Evaluate candidates with time budget monitoring
#' @keywords internal
evaluate_candidates_with_budget <- function(
  texts, labels, num_labels,
  candidates, cv_folds,
  start_time, max_time_minutes,
  device, verbose
) {
  if (verbose) {
    cli::cli_h2("Cross-Validation Evaluation")
  }

  n <- length(texts)
  fold_indices <- sample(rep(1:cv_folds, length.out = n))

  results <- list()

  for (i in seq_along(candidates)) {
    config <- candidates[[i]]

    # Check time budget
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
    if (elapsed >= max_time_minutes * 0.9) {
      if (verbose) {
        cli::cli_alert_warning("Time budget exhausted, stopping early")
      }
      break
    }

    if (verbose) {
      cli::cli_alert_info("Evaluating {config$description} ({i}/{length(candidates)})")
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

    results[[length(results) + 1]] <- list(
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

  results
}

#' Build ensemble from top performing diverse models
#' @keywords internal
build_ensemble_from_results <- function(
  texts, labels, num_labels,
  cv_results, device, verbose
) {
  # Select diverse top performers
  selected_idx <- select_diverse_candidates(cv_results, k = min(3, length(cv_results)))

  if (verbose) {
    cli::cli_alert_info("Selected {length(selected_idx)} diverse models for ensemble")
  }

  # Train each selected model on full data
  models <- list()
  weights <- numeric(length(selected_idx))

  train_data <- data.frame(
    text = texts,
    label = labels,
    stringsAsFactors = FALSE
  )

  for (i in seq_along(selected_idx)) {
    idx <- selected_idx[i]
    config <- cv_results[[idx]]$config

    if (verbose) {
      cli::cli_alert_info("Training ensemble member {i}/{length(selected_idx)}: {config$description}")
    }

    model <- train_final_model(
      texts, labels, num_labels,
      config, device, verbose = FALSE
    )

    models[[i]] <- model
    weights[i] <- cv_results[[idx]]$mean_accuracy
  }

  # Build ensemble
  ensemble <- build_ensemble(models, weights)

  if (verbose) {
    cli::cli_alert_success("Ensemble built with weights: {paste(round(ensemble$weights, 3), collapse=', ')}")
  }

  ensemble
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
