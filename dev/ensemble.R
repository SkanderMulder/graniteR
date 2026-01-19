#' Build ensemble classifier from multiple trained models
#'
#' Combines predictions from multiple classifiers using weighted averaging.
#' Weights can be based on validation performance or uniform.
#'
#' @param models List of trained classifier objects
#' @param weights Numeric vector of weights (will be normalized). If NULL, uniform weights.
#' @param method Ensemble method: "weighted_avg" (default) or "voting"
#'
#' @return Ensemble classifier object
#' @keywords internal
build_ensemble <- function(models, weights = NULL, method = "weighted_avg") {
  if (is.null(weights)) {
    weights <- rep(1, length(models))
  }

  # Normalize weights
  weights <- weights / sum(weights)

  structure(
    list(
      models = models,
      weights = weights,
      method = method,
      n_models = length(models)
    ),
    class = c("ensemble_classifier", "classifier")
  )
}

#' Predict with ensemble classifier
#'
#' @param object Ensemble classifier
#' @param data Data frame with text
#' @param text_col Column containing text
#' @param type Prediction type: "class", "prob", or "raw"
#' @param ... Additional arguments
#'
#' @return Predictions data frame
#' @export
#' @method predict ensemble_classifier
predict.ensemble_classifier <- function(object, data, text_col, type = "class", ...) {
  text_col <- rlang::enquo(text_col)

  # Get predictions from all models
  all_preds <- lapply(seq_along(object$models), function(i) {
    model <- object$models[[i]]
    preds <- predict(model, data, !!text_col, type = "prob")
    preds
  })

  if (object$method == "weighted_avg") {
    # Weighted average of probabilities
    # Extract probability columns (excluding text and label)
    prob_cols <- grep("^prob_", names(all_preds[[1]]), value = TRUE)

    # Initialize weighted probabilities
    weighted_probs <- all_preds[[1]][, prob_cols, drop = FALSE] * object$weights[1]

    # Add weighted contributions from other models
    if (length(all_preds) > 1) {
      for (i in 2:length(all_preds)) {
        weighted_probs <- weighted_probs +
          all_preds[[i]][, prob_cols, drop = FALSE] * object$weights[i]
      }
    }

    # Get final predictions
    prediction <- apply(weighted_probs, 1, which.max) - 1

    result <- data.frame(
      prediction = prediction
    )

    # Add probabilities if requested
    if (type %in% c("prob", "raw")) {
      result <- cbind(result, weighted_probs)
    }

  } else if (object$method == "voting") {
    # Majority voting
    votes <- sapply(all_preds, function(p) p$prediction)
    prediction <- apply(votes, 1, function(row) {
      as.integer(names(which.max(table(row))))
    })

    result <- data.frame(prediction = prediction)

    if (type %in% c("prob", "raw")) {
      # Compute vote-based probabilities
      vote_probs <- t(apply(votes, 1, function(row) {
        prop.table(table(factor(row, levels = 0:(max(row)))))
      }))
      colnames(vote_probs) <- paste0("prob_", 0:(ncol(vote_probs) - 1))
      result <- cbind(result, vote_probs)
    }
  }

  # Add text if it exists in data
  text_name <- rlang::as_name(text_col)
  if (text_name %in% names(data)) {
    result <- cbind(data.frame(text = dplyr::pull(data, !!text_col)), result)
  }

  result
}

#' Select top K diverse candidates for ensemble
#'
#' Given CV results, select K best performing diverse models for ensembling.
#' Diversity is measured by prediction disagreement.
#'
#' @param cv_results List of CV evaluation results
#' @param k Number of models to select
#'
#' @return Indices of selected candidates
#' @keywords internal
select_diverse_candidates <- function(cv_results, k = 3) {
  n_candidates <- length(cv_results)

  if (n_candidates <= k) {
    return(seq_len(n_candidates))
  }

  # Start with best performing model
  accuracies <- sapply(cv_results, function(x) x$mean_accuracy)
  selected <- c(which.max(accuracies))

  # Greedy selection: add most diverse high-performing model
  while (length(selected) < k) {
    remaining <- setdiff(seq_len(n_candidates), selected)

    if (length(remaining) == 0) break

    # Score = accuracy + diversity from already selected
    scores <- sapply(remaining, function(idx) {
      acc <- accuracies[idx]

      # Diversity: assume different types are diverse
      # (In real implementation, would use prediction disagreement)
      type <- cv_results[[idx]]$config$type
      selected_types <- sapply(selected, function(s) cv_results[[s]]$config$type)
      diversity <- 1 - mean(type == selected_types)

      # Combined score: 70% accuracy, 30% diversity
      0.7 * acc + 0.3 * diversity
    })

    best_idx <- remaining[which.max(scores)]
    selected <- c(selected, best_idx)
  }

  selected
}
