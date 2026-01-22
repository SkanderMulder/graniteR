#!/usr/bin/env Rscript
# Download Jigsaw Agile Community Rules dataset from Kaggle
#
# Prerequisites:
# 1. Install Kaggle CLI: pip install kaggle
# 2. Get API credentials from https://www.kaggle.com/settings/account
# 3. Save kaggle.json to ~/.kaggle/kaggle.json
# 4. Set permissions: chmod 600 ~/.kaggle/kaggle.json
# 5. Accept competition rules at https://www.kaggle.com/competitions/jigsaw-agile-community-rules/rules

library(readr)
library(dplyr)
library(tibble)
library(fs)

# Check Kaggle CLI is available
if (system2("kaggle", "--version", stdout = FALSE, stderr = FALSE) != 0) {
  stop("Kaggle CLI not found. Install with: pip install kaggle")
}

# Check Kaggle credentials
kaggle_json <- path_expand("~/.kaggle/kaggle.json")
if (!file_exists(kaggle_json)) {
  stop("Kaggle credentials not found. ",
       "Download kaggle.json from https://www.kaggle.com/settings/account ",
       "and save to ~/.kaggle/kaggle.json")
}

# Create temp directory for download
temp_dir <- tempdir()
data_dir <- path(temp_dir, "jigsaw-agile")
dir_create(data_dir)

cat("Downloading Jigsaw Agile Community Rules dataset...\n")
cat("This may take a few minutes...\n\n")

# Download competition data
result <- system2(
  "kaggle",
  c("competitions", "download", "-c", "jigsaw-agile-community-rules", "-p", data_dir),
  stdout = TRUE,
  stderr = TRUE
)

if (length(result) > 0) {
  cat("Download result:\n")
  cat(paste(result, collapse = "\n"), "\n\n")
}

# Unzip the data
zip_file <- path(data_dir, "jigsaw-agile-community-rules.zip")
if (file_exists(zip_file)) {
  cat("Unzipping files...\n")
  unzip(zip_file, exdir = data_dir)
} else {
  stop("Download failed. Make sure you've accepted the competition rules.")
}

# List downloaded files
files <- dir_ls(data_dir, glob = "*.csv")
cat("\nDownloaded files:\n")
print(files)

# Read training data
train_file <- path(data_dir, "train.csv")
if (!file_exists(train_file)) {
  stop("train.csv not found. Available files: ", paste(dir_ls(data_dir), collapse = ", "))
}

cat("\nReading training data...\n")
train_data <- read_csv(train_file, show_col_types = FALSE)

cat("Dataset shape:", nrow(train_data), "rows,", ncol(train_data), "columns\n")
cat("Column names:", paste(names(train_data), collapse = ", "), "\n\n")

# Display structure
cat("Dataset structure:\n")
str(train_data)

cat("\nFirst few rows:\n")
print(head(train_data, 3))

# Check label distribution
if ("label" %in% names(train_data)) {
  cat("\nLabel distribution:\n")
  print(table(train_data$label))
}

# Create full dataset
cat("\nProcessing full dataset...\n")
agile_rules_full <- train_data %>%
  select(text, label) %>%
  mutate(label = as.integer(label))

# Add label names if binary classification
if (length(unique(agile_rules_full$label)) == 2) {
  label_map <- c("non-violation", "violation")
  agile_rules_full <- agile_rules_full %>%
    mutate(label_name = label_map[label + 1])
}

cat("\nFull dataset summary:\n")
cat("Rows:", nrow(agile_rules_full), "\n")
cat("Label distribution:\n")
print(table(agile_rules_full$label))
if ("label_name" %in% names(agile_rules_full)) {
  cat("Label names:\n")
  print(table(agile_rules_full$label_name))
}

# Create sample dataset (10-15k rows)
sample_size <- min(15000, nrow(agile_rules_full))
cat("\nCreating sample dataset (", sample_size, " rows)...\n", sep = "")

set.seed(42)
agile_rules_sample <- agile_rules_full %>%
  group_by(label) %>%
  slice_sample(n = ceiling(sample_size / n_distinct(agile_rules_full$label))) %>%
  ungroup() %>%
  slice_sample(n = sample_size)

cat("Sample dataset summary:\n")
cat("Rows:", nrow(agile_rules_sample), "\n")
print(table(agile_rules_sample$label))

# Save datasets
cat("\nSaving datasets...\n")

# Save full dataset
usethis::use_data(agile_rules_full, overwrite = TRUE, compress = "xz")
cat("✓ Saved agile_rules_full.rda\n")

# Save sample dataset
usethis::use_data(agile_rules_sample, overwrite = TRUE, compress = "xz")
cat("✓ Saved agile_rules_sample.rda\n")

# Also save CSV to inst/extdata for reference
extdata_dir <- path("inst", "extdata")
dir_create(extdata_dir)
write_csv(agile_rules_sample, path(extdata_dir, "agile_rules_sample.csv"))
cat("✓ Saved inst/extdata/agile_rules_sample.csv\n")

# Clean up temp files
cat("\nCleaning up temporary files...\n")
dir_delete(data_dir)

cat("\n")
cat(strrep("=", 71), "\n")
cat("✓ Dataset creation complete!\n")
cat(strrep("=", 71), "\n\n")

cat("You can now use:\n")
cat("  data(agile_rules_full, package = 'graniteR')\n")
cat("  data(agile_rules_sample, package = 'graniteR')\n\n")

cat("Next steps:\n")
cat("1. Add documentation to R/data.R\n")
cat("2. Run devtools::document()\n")
cat("3. Test with: devtools::load_all(); data(agile_rules_full)\n")
