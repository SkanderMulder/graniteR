## Code to prepare example datasets
## This script is for internal use and is not part of the package

library(tibble)
library(readr)

# Example sentiment dataset
sentiment_example <- tibble(
  text = c(
    "I love this product",
    "This is terrible",
    "Great experience",
    "Very disappointing",
    "Excellent service",
    "Poor quality"
  ),
  label = c(1, 0, 1, 0, 1, 0)
)

# Malicious prompts dataset
# Load from inst/data/ and save as proper R dataset
# Note: The CSV file is kept in inst/data/ for reference but the package
# uses the compressed .rda format in data/ for efficiency (105K vs 1.2M)
csv_path <- file.path("inst", "extdata", "malicious_prompts_sample.csv")
if (file.exists(csv_path)) {
  malicious_prompts_sample <- read_csv(csv_path, show_col_types = FALSE)
  usethis::use_data(malicious_prompts_sample, overwrite = TRUE)
  message("Created malicious_prompts_sample dataset")
}

# Save sentiment example if needed
# usethis::use_data(sentiment_example, overwrite = TRUE)
