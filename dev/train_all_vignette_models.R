#!/usr/bin/env Rscript
# Train all models for vignettes
# This script trains standard and MoE classifiers for all full datasets

library(graniteR)
library(dplyr)

# Helper function to clear GPU memory
clear_gpu <- function() {
  torch <- reticulate::import('torch')
  if (torch$cuda$is_available()) {
    torch$cuda$empty_cache()
  }
  gc()
}

# Helper function to train and save standard classifier
train_standard <- function(data, name, num_labels, epochs = 5, batch_size = 50, lr = 1e-3) {
  cat("\n", strrep("=", 70), "\n")
  cat("Training Standard Classifier:", name, "\n")
  cat(strrep("=", 70), "\n\n")

  set.seed(42)
  n <- nrow(data)
  train_idx <- sample(n, size = floor(0.8 * n))
  train_data <- data[train_idx, ]

  cat("Training samples:", nrow(train_data), "\n")
  cat("Num labels:", num_labels, "\n\n")

  clf <- classifier(num_labels = num_labels)
  clf <- clf |> train(
    train_data,
    text,
    label,
    epochs = epochs,
    batch_size = batch_size,
    learning_rate = lr,
    validation_split = 0.2,
    verbose = TRUE
  )

  save_path <- paste0("models/", name, "_standard")
  save_classifier(clf, file = save_path)
  cat("\nSaved to:", save_path, "\n")

  clear_gpu()
  rm(clf, train_data)
  clear_gpu()

  return(save_path)
}

# Helper function to train and save MoE classifier
train_moe_model <- function(data, name, num_labels, num_experts = NULL, epochs = 5, batch_size = 8, lr = 2e-5) {
  cat("\n", strrep("=", 70), "\n")
  cat("Training MoE Classifier:", name, "\n")
  cat(strrep("=", 70), "\n\n")

  if (is.null(num_experts)) {
    num_experts <- if (num_labels > 2) 6 else 3
  }

  set.seed(42)
  n <- nrow(data)
  train_idx <- sample(n, size = floor(0.8 * n))
  train_data <- data[train_idx, ]

  cat("Training samples:", nrow(train_data), "\n")
  cat("Num labels:", num_labels, "\n")
  cat("Num experts:", num_experts, "\n\n")

  clf <- moe_classifier(
    num_labels = num_labels,
    num_experts = num_experts,
    freeze_backbone = FALSE,
    dropout = 0.2,
    expert_depth = 2
  )

  clf <- train_moe(
    clf,
    train_data,
    text,
    label,
    epochs = epochs,
    batch_size = batch_size
  )

  save_path <- paste0("models/", name, "_moe")
  save_classifier(clf, file = save_path)
  cat("\nSaved to:", save_path, "\n")

  clear_gpu()
  rm(clf, train_data)
  clear_gpu()

  return(save_path)
}

# ============================================================================
# Main Training Pipeline
# ============================================================================

cat("\n")
cat(strrep("#", 70), "\n")
cat("# Training All Vignette Models\n")
cat(strrep("#", 70), "\n")

# 1. Emotion Detection (6 classes)
cat("\n[1/8] Emotion Detection - Standard\n")
data(emotion_full)
train_standard(emotion_full, "emotion", num_labels = 6, epochs = 5, batch_size = 50)

cat("\n[2/8] Emotion Detection - MoE\n")
train_moe_model(emotion_full, "emotion", num_labels = 6, num_experts = 6, epochs = 5, batch_size = 8)

# 2. Sentiment Analysis (2 classes)
cat("\n[3/8] Sentiment Analysis - Standard\n")
data(sentiment_imdb_full)
train_standard(sentiment_imdb_full, "sentiment", num_labels = 2, epochs = 3, batch_size = 8)

cat("\n[4/8] Sentiment Analysis - MoE\n")
train_moe_model(sentiment_imdb_full, "sentiment", num_labels = 2, num_experts = 3, epochs = 3, batch_size = 8)

# 3. Hate Speech Detection (2 classes)
cat("\n[5/8] Hate Speech Detection - Standard\n")
data(hate_speech_full)
train_standard(hate_speech_full, "hate_speech", num_labels = 2, epochs = 3, batch_size = 8)

cat("\n[6/8] Hate Speech Detection - MoE\n")
train_moe_model(hate_speech_full, "hate_speech", num_labels = 2, num_experts = 3, epochs = 3, batch_size = 8)

# 4. Malicious Prompts Detection (2 classes)
cat("\n[7/8] Malicious Prompts - Standard\n")
data(malicious_prompts_full)
train_standard(malicious_prompts_full, "malicious_prompts", num_labels = 2, epochs = 3, batch_size = 8)

cat("\n[8/8] Malicious Prompts - MoE\n")
train_moe_model(malicious_prompts_full, "malicious_prompts", num_labels = 2, num_experts = 3, epochs = 3, batch_size = 8)

# ============================================================================
# Verification: Load all models and test
# ============================================================================

cat("\n")
cat(strrep("#", 70), "\n")
cat("# Verifying All Models\n")
cat(strrep("#", 70), "\n\n")

models <- c(
  "emotion_standard", "emotion_moe",
  "sentiment_standard", "sentiment_moe",
  "hate_speech_standard", "hate_speech_moe",
  "malicious_prompts_standard", "malicious_prompts_moe"
)

for (model_name in models) {
  cat("Loading:", model_name, "... ")
  tryCatch({
    clf <- load_classifier(paste0("models/", model_name))
    cat("OK\n")
    rm(clf)
  }, error = function(e) {
    cat("FAILED:", e$message, "\n")
  })
}

cat("\n")
cat(strrep("#", 70), "\n")
cat("# All Models Trained and Verified!\n")
cat(strrep("#", 70), "\n")
cat("\nModels saved in models/ directory:\n")
cat("  - emotion_standard, emotion_moe\n")
cat("  - sentiment_standard, sentiment_moe\n")
cat("  - hate_speech_standard, hate_speech_moe\n")
cat("  - malicious_prompts_standard, malicious_prompts_moe\n")
