#!/usr/bin/env Rscript
# Train a MoE classifier for malicious prompt detection

devtools::load_all()
library(dplyr)

set.seed(42)

cat(paste(rep("=", 70), collapse = ""), "\n")
cat(" MoE Classifier Training\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

# Load dataset
cat("Loading malicious_prompts_full dataset...\n")
data(malicious_prompts_full, package = "graniteR")

cat(sprintf("Total: %s rows (%s benign, %s malicious)\n\n",
            format(nrow(malicious_prompts_full), big.mark = ","),
            format(sum(malicious_prompts_full$label == 0), big.mark = ","),
            format(sum(malicious_prompts_full$label == 1), big.mark = ",")))

# Use ALL data with gradient accumulation for scalability
cat("Using full dataset (no sampling)...\n")

data_sample <- malicious_prompts_full |>
  select(text, label)

# Train/test split
n <- nrow(data_sample)
train_idx <- sample(n, floor(0.8 * n))
train_data <- data_sample[train_idx, ]
test_data <- data_sample[-train_idx, ]

cat(sprintf("Train: %s | Test: %s\n\n",
            format(nrow(train_data), big.mark = ","),
            format(nrow(test_data), big.mark = ",")))

# Train MoE model with gradient accumulation for memory efficiency
cat("Training MoE classifier with gradient accumulation...\n")
cat("Effective batch size: 4 × 8 accumulation steps = 32\n\n")

clf_moe <- moe_classifier(
  num_labels = 2,
  num_experts = 4,
  model_name = "perplexity-ai/pplx-embed-v1-0.6b",
  freeze_backbone = FALSE,
  dropout = 0.2,
  expert_depth = 2,
  trust_remote_code = TRUE
)

# Train with small batch size + gradient accumulation
clf_moe <- train_moe(
  clf_moe,
  train_data,
  text,
  label,
  epochs = 3,
  batch_size = 4,              # Small to fit in GPU
  accumulation_steps = 8,      # Accumulate 8 batches (effective=32)
  learning_rate = 2e-5,
  validation_split = 0.2
)

# Evaluate
cat("\nEvaluating on test set...\n")
preds <- predict(clf_moe, test_data, text, type = "class")

tp <- sum(preds$prediction == 1 & preds$label == 1)
fp <- sum(preds$prediction == 1 & preds$label == 0)
fn <- sum(preds$prediction == 0 & preds$label == 1)
tn <- sum(preds$prediction == 0 & preds$label == 0)

acc <- mean(preds$prediction == preds$label)
prec <- tp / (tp + fp)
rec <- tp / (tp + fn)
f1 <- 2 * prec * rec / (prec + rec)

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat(" Results\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")
cat(sprintf("Accuracy: %.1f%% | Precision: %.1f%% | Recall: %.1f%% | F1: %.3f\n",
            acc*100, prec*100, rec*100, f1))
cat(sprintf("TP: %s | TN: %s | FP: %s | FN: %s\n\n",
            format(tp, big.mark = ","),
            format(tn, big.mark = ","),
            format(fp, big.mark = ","),
            format(fn, big.mark = ",")))

# Save
model_path <- "inst/extdata/models/malicious_prompt_detector_moe"
cat(sprintf("Saving to: %s\n", model_path))
save_classifier(clf_moe, file = model_path)

cat("\n✓ Done!\n")
cat(sprintf("\nLoad with: load_classifier(find_model('malicious_prompt_detector_moe'))\n"))
