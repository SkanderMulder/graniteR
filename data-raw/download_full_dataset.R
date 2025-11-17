#!/usr/bin/env Rscript

library(reticulate)
library(readr)
library(dplyr)
library(fs)

# Configure Python environment
venv_path <- file.path(getwd(), ".venv", "bin", "python")
if (file.exists(venv_path)) {
  use_python(venv_path, required = TRUE)
  cat("Using Python from:", venv_path, "\n")
} else {
  stop("Virtual environment not found at: ", venv_path)
}

# Import required Python modules
datasets <- import("datasets")

cat("Downloading full malicious prompts dataset from HuggingFace...\n")
cat("This may take a few minutes...\n\n")

# Download the full dataset
dataset <- datasets$load_dataset("ahsanayub/malicious-prompts")

# Convert to R data frame
cat("Converting to R data frame...\n")
train_data <- py_to_r(dataset$train$to_pandas())
cat("Dataset shape:", nrow(train_data), "rows,", ncol(train_data), "columns\n")

# Rename columns if needed to match expected format
if ("is_malicious" %in% names(train_data)) {
  train_data <- train_data %>%
    rename(label = is_malicious)
}

# Display summary statistics
cat("\nLabel distribution:\n")
print(table(train_data$label))

# Save as CSV in inst/extdata
extdata_dir <- file.path("inst", "extdata")
dir_create(extdata_dir)

csv_path <- file.path(extdata_dir, "malicious_prompts_full.csv")
cat("\nSaving to:", csv_path, "\n")
write_csv(train_data, csv_path)

cat("\nFull dataset saved successfully!\n")
cat("File size:", file.size(csv_path) / (1024^2), "MB\n")

# Optionally create .rda format for faster loading in R
cat("\nCreating compressed .rda format...\n")
malicious_prompts_full <- train_data
usethis::use_data(malicious_prompts_full, overwrite = TRUE, compress = "xz")

cat("\nDone! You can now use:\n")
cat("  data(malicious_prompts_full, package = 'graniteR')\n")
