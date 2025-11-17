# Enhanced Hyperparameter Tuning Interface

## Overview
Provide user-friendly hyperparameter tuning capabilities beyond the current AutoML implementation.

## Current State
- `auto_classify()` handles hyperparameter search internally via CASH
- Manual tuning requires understanding of model internals
- No easy way to customize search space
- Limited visibility into tuning process

## Proposed Enhancements

### 1. Explicit Tuning Interface
```r
# Define custom search space
search_space <- list(
  learning_rate = c(1e-5, 5e-5, 1e-4),
  epochs = c(3, 5, 10),
  freeze_backbone = c(TRUE, FALSE),
  dropout = c(0.1, 0.2, 0.3)
)

# Run grid search
tuned <- tune_classifier(
  train_data,
  text_col = text,
  label_col = label,
  search_space = search_space,
  method = "grid",        # or "random", "bayesian"
  cv_folds = 5,
  metric = "accuracy",
  n_trials = 20
)

# View results
plot(tuned)  # Performance across trials
best_params(tuned)  # Best configuration
```

### 2. Progress Visualization
```r
# Real-time tuning dashboard
tune_results <- tune_classifier(
  ...,
  dashboard = TRUE  # Opens Shiny dashboard
)

# Or use built-in plotting
plot_tuning_history(tune_results)
plot_param_importance(tune_results)
```

### 3. Resume Interrupted Tuning
```r
# Save tuning state
save_tuning_state(tune_results, "checkpoint.rds")

# Resume later
tune_results <- resume_tuning(
  "checkpoint.rds",
  additional_trials = 10
)
```

### 4. Custom Metrics
```r
# Define custom evaluation metric
f1_macro <- function(y_true, y_pred) {
  # Custom F1 implementation
}

tune_classifier(
  ...,
  metric = f1_macro,
  maximize = TRUE
)
```

## Implementation Strategy

### Phase 1: Basic Grid/Random Search
- Add `tune_classifier()` function
- Support grid and random search
- Cross-validation with multiple metrics
- Results visualization

### Phase 2: Advanced Methods
- Bayesian optimization (using `ParBayesianOptimization` or similar)
- Hyperband for early stopping
- Multi-objective optimization (accuracy vs speed)

### Phase 3: Integration
- Integrate with `auto_classify()` as optional backend
- Allow users to plug custom optimizers
- Export results to MLflow/Weights & Biases

## API Design Principles
- Sensible defaults (works out-of-box)
- Flexible search space definition
- Compatible with existing `classifier()` and `moe_classifier()`
- Minimal dependencies (core in base package, advanced features in Suggests)

## Files to Modify
- `R/tune.R` (new file)
- `R/auto_classify.R` (integration)
- `tests/testthat/test-tune.R` (new tests)
- `vignettes/hyperparameter-tuning.Rmd` (new vignette)

## Benefits
- Lower barrier to entry for ML practitioners
- Better models through systematic tuning
- Reproducible experiments
- Educational value (understanding hyperparameter impact)

## Related
- AutoML implementation docs (`.github/AUTOML_IMPLEMENTATION.md`)
- Benchmarking suite (#[TBD])

## Priority
Medium - Enhances usability but AutoML covers basic needs
