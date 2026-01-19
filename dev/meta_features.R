#' Compute meta-features for text classification datasets
#'
#' Extracts dataset characteristics that help predict which model architecture
#' will perform best. Used internally by auto_classify for intelligent model selection.
#'
#' @param texts Character vector of text samples
#' @param labels Integer or factor vector of labels
#'
#' @return List of meta-features
#' @keywords internal
compute_meta_features <- function(texts, labels) {
  # Basic statistics
  n_samples <- length(texts)
  n_labels <- length(unique(labels))
  samples_per_class <- n_samples / n_labels

  # Label distribution
  label_probs <- prop.table(table(labels))
  label_entropy <- -sum(label_probs * log2(label_probs + 1e-10))
  max_class_imbalance <- max(label_probs) / min(label_probs)

  # Text characteristics
  text_lengths <- nchar(texts)
  avg_text_len <- mean(text_lengths)
  median_text_len <- median(text_lengths)
  sd_text_len <- sd(text_lengths)
  cv_text_len <- sd_text_len / (avg_text_len + 1e-10)  # coefficient of variation

  # Vocabulary statistics (simple word-based)
  words <- tolower(unlist(strsplit(texts, "\\s+")))
  vocab_size <- length(unique(words))
  vocab_richness <- vocab_size / length(words)  # type-token ratio

  # Complexity indicators
  complexity_score <- n_labels * log10(pmax(n_samples, 100))
  data_density <- n_samples / (vocab_size + 1)

  list(
    n_samples = n_samples,
    n_labels = n_labels,
    samples_per_class = samples_per_class,
    label_entropy = label_entropy,
    class_imbalance = max_class_imbalance,
    avg_text_len = avg_text_len,
    median_text_len = median_text_len,
    cv_text_len = cv_text_len,
    vocab_size = vocab_size,
    vocab_richness = vocab_richness,
    complexity_score = complexity_score,
    data_density = data_density
  )
}

#' Predict model performance based on meta-features
#'
#' Simple meta-learner that predicts which model type will perform best
#' based on dataset characteristics. Uses heuristics derived from empirical
#' observations across multiple text classification benchmarks.
#'
#' @param meta_features List of meta-features from compute_meta_features
#' @param candidate_type Character: "frozen", "finetuned", or "moe"
#'
#' @return Predicted accuracy score (0-1)
#' @keywords internal
predict_candidate_performance <- function(meta_features, candidate_type) {
  # Base performance by type
  base_score <- switch(candidate_type,
    frozen = 0.70,
    finetuned = 0.82,
    moe = 0.84,
    0.70
  )

  # Adjustments based on meta-features
  score <- base_score

  # Sample size effects
  if (meta_features$samples_per_class < 50) {
    # Small data: frozen wins
    score <- score + ifelse(candidate_type == "frozen", 0.08, -0.10)
  } else if (meta_features$samples_per_class > 500) {
    # Large data: fine-tuning and MoE win
    score <- score + ifelse(candidate_type == "frozen", -0.05, 0.05)
  }

  # Class imbalance effects
  if (meta_features$class_imbalance > 5) {
    # Imbalanced: simple models suffer
    score <- score + ifelse(candidate_type == "frozen", -0.03, 0.02)
  }

  # Complexity effects
  if (meta_features$complexity_score > 20) {
    # Complex tasks: MoE can help
    score <- score + ifelse(candidate_type == "moe", 0.03, 0)
  }

  # Text length variance effects
  if (meta_features$cv_text_len > 0.5) {
    # High variance: flexible models help
    score <- score + ifelse(candidate_type == "frozen", -0.02, 0.02)
  }

  # Vocab richness effects
  if (meta_features$vocab_richness > 0.5) {
    # Rich vocabulary: benefits from full model
    score <- score + ifelse(candidate_type == "frozen", -0.03, 0.03)
  }

  # Multiclass effects
  if (meta_features$n_labels >= 6) {
    # Many classes: MoE architecture helps
    score <- score + ifelse(candidate_type == "moe", 0.04, 0)
  }

  # Clamp to valid range
  pmin(1.0, pmax(0.0, score))
}
