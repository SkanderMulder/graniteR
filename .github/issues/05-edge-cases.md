# Robust Edge Case Handling

## Overview
Improve robustness by systematically handling edge cases and providing helpful error messages.

## Current Gaps
Per overview: "Experimental tag signals possible API flux or bugs in edge cases (e.g., very long texts)"

## Edge Cases to Handle

### 1. Text Length Issues
**Very Long Texts**
- Texts exceeding model's max_length (512 tokens for most models)
- Current behavior: Silent truncation or errors
- Desired behavior: 
  ```r
  # Option 1: Automatic chunking with aggregation
  embed(data, text, max_length = 512, truncation = "chunk_mean")
  
  # Option 2: Clear warning
  embed(data, text, max_length = 512, truncation = "warn")
  
  # Option 3: Error with helpful message
  embed(data, text, max_length = 512, truncation = "error")
  ```

**Empty Texts**
- Empty strings or NA values
- Current: May cause cryptic errors
- Desired: Skip with warning or impute with special token

**Special Characters**
- Emojis, unicode, control characters
- Non-UTF8 encodings
- Mixed languages

### 2. Data Quality Issues
**Missing Values**
```r
data <- tibble(
  text = c("valid text", NA, "", "  ", "another text"),
  label = c(1, 0, NA, 1, 0)
)

# Should provide clear error or handle gracefully
clf <- classifier(num_labels = 2) |>
  train(data, text, label, na_action = "drop")  # or "error", "impute"
```

**Class Imbalance**
- Warning when severe imbalance detected
- Suggestion to use stratified sampling
- Option for class weighting

**Tiny Datasets**
- <10 samples per class
- Warning about unreliable results
- Automatic CV fold adjustment

### 3. Model Issues
**Model Not Found**
```r
# Helpful error when model doesn't exist
clf <- classifier(model_name = "nonexistent/model")
# Error: Model 'nonexistent/model' not found on Hugging Face Hub.
# Did you mean: 'ibm-granite/granite-embedding-r2'?
# See https://huggingface.co/models for available models.
```

**Incompatible Models**
- Using decoder models when encoder expected
- Models without pooling support
- Clear error messages with examples

**Memory Exhaustion**
- Detect OOM conditions early
- Suggest batch_size reduction
- Recommend CPU fallback

### 4. Device Issues
**CUDA Errors**
```r
# Graceful fallback to CPU
clf <- classifier(device = "cuda")
# Warning: CUDA device requested but not available. Falling back to CPU.
# Set device="cpu" explicitly to suppress this warning.
```

**Mixed Device Scenarios**
- Model on GPU, data on CPU
- Multiple GPUs
- MPS (Apple Silicon)

### 5. Prediction Edge Cases
**Unseen Classes**
- Predictions on data with labels not in training set
- Appropriate handling and warnings

**Distribution Shift**
- Detect when prediction text very different from training
- Optional confidence intervals or uncertainty estimates

**Batch Size Mismatches**
- Single text vs batch predictions
- Consistent API regardless of input size

## Implementation Plan

### Phase 1: Input Validation
```r
# Add to R/utils.R
validate_text_input <- function(texts, allow_na = FALSE, allow_empty = FALSE) {
  # Check encoding
  # Check length
  # Check for NAs
  # Provide helpful errors
}

validate_labels <- function(labels, num_labels = NULL) {
  # Check type
  # Check range
  # Check for NAs
  # Detect imbalance
}
```

### Phase 2: Error Messages
- Develop error message guidelines
- Use `cli` package for formatted messages
- Include suggestions in errors
- Add "Did you mean?" suggestions where appropriate

### Phase 3: Graceful Degradation
- Auto-detect issues and warn
- Provide fallback behaviors
- Document edge case handling in vignettes

### Phase 4: Testing
```r
# tests/testthat/test-edge-cases.R
test_that("handles very long texts", {
  long_text <- paste(rep("word", 10000), collapse = " ")
  expect_warning(embed(tibble(text = long_text), text))
})

test_that("handles empty texts gracefully", {
  data <- tibble(text = c("valid", "", NA))
  expect_error(embed(data, text), class = "graniteR_empty_text")
})

test_that("provides helpful error for missing model", {
  expect_error(
    classifier(model_name = "invalid/model"),
    regexp = "Model.*not found"
  )
})
```

## Documentation

### Vignette: "Handling Common Issues"
- Edge case catalog
- Troubleshooting guide
- Best practices
- FAQ section

### Error Message Index
- Document all error classes
- Recovery suggestions
- Common causes

## Benefits
- Better user experience
- Fewer GitHub issues from confused users
- Easier debugging
- Production readiness

## Testing Strategy
- Comprehensive edge case test suite
- Fuzzing with random inputs
- Integration tests with malformed data
- Performance tests under edge conditions

## Related
- CRAN preparation (#[TBD])
- Documentation improvements (#[TBD])

## Priority
High - Critical for stability and CRAN acceptance

## Success Criteria
- All edge cases have defined behavior
- No silent failures
- Helpful error messages for 90%+ of errors
- Comprehensive test coverage
