## Code to prepare example datasets
## This script is for internal use and is not part of the package

library(tibble)

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

# Save if needed
# usethis::use_data(sentiment_example, overwrite = TRUE)
