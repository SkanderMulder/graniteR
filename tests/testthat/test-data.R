test_that("malicious_prompts_sample dataset loads correctly", {
  data(malicious_prompts_sample, package = "graniteR")
  
  # Check dataset exists
  expect_true(exists("malicious_prompts_sample"))
  
  # Check structure
  expect_s3_class(malicious_prompts_sample, "data.frame")
  expect_named(malicious_prompts_sample, c("text", "label", "source"))
  
  # Check dimensions
  expect_equal(nrow(malicious_prompts_sample), 1000)
  expect_equal(ncol(malicious_prompts_sample), 3)
  
  # Check column types
  expect_type(malicious_prompts_sample$text, "character")
  expect_type(malicious_prompts_sample$label, "double")
  expect_type(malicious_prompts_sample$source, "character")
  
  # Check label distribution
  label_counts <- table(malicious_prompts_sample$label)
  expect_equal(as.numeric(label_counts["0"]), 500)
  expect_equal(as.numeric(label_counts["1"]), 500)
  
  # Check no missing values
  expect_false(any(is.na(malicious_prompts_sample$text)))
  expect_false(any(is.na(malicious_prompts_sample$label)))
  expect_false(any(is.na(malicious_prompts_sample$source)))
})

test_that("malicious_prompts_sample has valid content", {
  data(malicious_prompts_sample, package = "graniteR")
  
  # Check labels are binary
  expect_true(all(malicious_prompts_sample$label %in% c(0, 1)))
  
  # Check text column has content
  expect_true(all(nchar(malicious_prompts_sample$text) > 0))
  
  # Check source column has content
  expect_true(all(nchar(malicious_prompts_sample$source) > 0))
  
  # Check for unique sources
  expect_true(length(unique(malicious_prompts_sample$source)) >= 1)
})
