# AutoGluon vs graniteR Auto-Classify Comparison

## Current graniteR Implementation

### Strengths
- ✅ Meta-learning for candidate ranking
- ✅ CASH-style hyperparameter optimization
- ✅ Ensemble building
- ✅ Resource-aware scheduling
- ✅ Data-driven decision making

### Weaknesses (from code review)
- ❌ Hardcoded hyperparameter grids (manual exploration ranges)
- ❌ Simple greedy CV evaluation (no early stopping per trial)
- ❌ Basic ensemble weighting (accuracy-based only)
- ❌ No Bayesian optimization (uses grid search)
- ❌ Manual candidate generation with if/else logic

## AutoGluon's Superior Approach

### 1. Search Space Definition
**AutoGluon**: Uses Ray Tune's declarative search spaces
```python
search_space = {
    "learning_rate": tune.uniform(0.00005, 0.001),
    "model": tune.choice(["bert", "roberta", "electra"]),
    "dropout": tune.quniform(0.1, 0.5, 0.05)
}
```

**graniteR (current)**: Hardcoded grids
```r
frozen_configs <- expand.grid(
    learning_rate = c(5e-4, 1e-3, 2e-3),  # Fixed values
    epochs = 3:7
)
```

### 2. Search Strategy
**AutoGluon**: Bayesian optimization via Ray Tune
- Learns from previous trials
- Focuses on promising regions
- Adaptive sampling

**graniteR (current)**: Meta-learning heuristics + grid search
- Predicts once upfront
- Exhaustive grid evaluation
- No adaptation during search

### 3. Scheduling
**AutoGluon**: ASHA (Asynchronous Successive Halving)
- Early stops poor candidates
- Allocates more resources to promising ones
- Parallel trial execution

**graniteR (current)**: Sequential CV with time budget cutoff
- Completes all CV folds per candidate
- No early stopping within trials
- Sequential only

### 4. Ensemble Strategy
**AutoGluon**: Greedy ensemble ("soup") + Multi-layer stacking
- Combines top-K checkpoints
- Stacking uses original features + predictions
- Post-hoc ensemble optimization (PSEO)

**graniteR (current)**: Weighted average by accuracy + diversity
- Simple weighted mean
- No stacking
- Fixed 3-model ensemble

## Recommended Improvements for graniteR

### Priority 1: Replace Grid Search with Bayesian Optimization
Instead of:
```r
# Current: fixed grid
frozen_configs <- expand.grid(
    learning_rate = c(5e-4, 1e-3, 2e-3),
    epochs = 3:7
)
```

Use:
```r
# Proposed: continuous search space
search_space <- list(
    frozen = list(
        learning_rate = c(1e-4, 5e-3),  # Range
        epochs = c(3, 10)                # Range
    ),
    finetuned = list(
        learning_rate = c(5e-6, 1e-4),
        epochs = c(2, 5)
    )
)

# Sample using GP (Gaussian Process) or TPE (Tree-Parzen Estimator)
```

**Benefits**:
- 10-20x fewer trials for same quality
- Finds better hyperparameters
- Adaptive to dataset characteristics

**Implementation**: Use `ParBayesianOptimization` R package or simple GP

### Priority 2: Add Trial-Level Early Stopping
Instead of completing all CV folds:
```r
# Current
for (fold in 1:cv_folds) {
    accuracy[fold] <- evaluate_fold(...)  # Always completes all folds
}
```

Use successive halving:
```r
# Proposed
candidates <- rank_by_meta_learning(candidates)
resources <- c(1, 2, 5)  # Number of CV folds

for (resource_level in resources) {
    # Evaluate top 50% from previous level
    survivors <- top_50_percent(candidates)
    for (candidate in survivors) {
        accuracy <- evaluate_with_folds(candidate, n_folds = resource_level)
    }
    candidates <- rank_by_accuracy(survivors)
}
```

**Benefits**:
- 2-3x faster search
- Eliminates bad candidates early
- More trials in time budget

### Priority 3: Improve Ensemble Strategy
Current:
```r
# Simple weighted average
weights <- accuracies / sum(accuracies)
ensemble <- weighted_mean(predictions, weights)
```

Proposed (Greedy Soup + Stacking):
```r
# Greedy soup: iteratively add best performing checkpoint
ensemble <- list()
for (i in 1:max_members) {
    best_addition <- NULL
    best_score <- -Inf

    for (candidate in remaining_candidates) {
        temp_ensemble <- c(ensemble, candidate)
        score <- evaluate_ensemble(temp_ensemble, validation_data)
        if (score > best_score) {
            best_addition <- candidate
            best_score <- score
        }
    }
    ensemble <- c(ensemble, best_addition)
}

# Then add stacking layer
meta_features <- cbind(
    original_features,
    ensemble_predictions
)
stacker <- train_meta_learner(meta_features, labels)
```

**Benefits**:
- 1-3% accuracy improvement
- Better than simple weighted average
- Leverages complementary models

### Priority 4: Declarative Search Space
Replace manual candidate generation:
```r
# Current: lots of if/else
if (meta_features$n_labels >= 4 &&
    meta_features$samples_per_class >= 200 &&
    meta_features$complexity_score > 12) {
    # Generate MoE candidates
    for (i in seq_len(nrow(moe_configs))) {
        candidates[[length(candidates) + 1]] <- list(...)
    }
}
```

With declarative config:
```r
# Proposed: data-driven search space
model_registry <- list(
    frozen = list(
        enabled = function(meta) TRUE,  # Always available
        search_space = list(
            learning_rate = c(1e-4, 5e-3),
            epochs = c(3, 10)
        ),
        time_estimate = function(n, epochs) n / 60000 * epochs
    ),
    finetuned = list(
        enabled = function(meta) meta$samples_per_class >= 50,
        search_space = list(
            learning_rate = c(5e-6, 1e-4),
            epochs = c(2, 5)
        ),
        time_estimate = function(n, epochs) n / 6000 * epochs
    ),
    moe = list(
        enabled = function(meta) {
            meta$n_labels >= 4 &&
            meta$samples_per_class >= 200 &&
            meta$complexity_score > 12
        },
        search_space = list(
            learning_rate = c(5e-6, 1e-4),
            epochs = c(2, 4),
            num_experts = c(2, 8)
        ),
        time_estimate = function(n, epochs) n / 4000 * epochs
    )
)

# Use registry to generate candidates
enabled_models <- Filter(function(m) m$enabled(meta_features), model_registry)
search_space <- lapply(enabled_models, function(m) m$search_space)
```

**Benefits**:
- Cleaner code
- Easier to extend
- Self-documenting
- Testable components

## Implementation Plan

### Phase 1 (Quick Wins - 1 day)
1. Extract candidate generation to registry pattern
2. Add simple successive halving for early stopping
3. Improve ensemble with greedy soup algorithm

### Phase 2 (Medium Effort - 2-3 days)
1. Integrate `ParBayesianOptimization` for HPO
2. Implement ASHA scheduler
3. Add checkpoint-based ensemble fusion

### Phase 3 (Advanced - 1 week)
1. Multi-layer stacking like AutoGluon
2. Post-hoc ensemble optimization
3. Transfer learning from meta-learning database

## Code Quality Improvements

Replace nested if/else with registry pattern:
```r
# Before (current)
if (samples_per_class < 50) {
    candidates <- frozen_only()
} else if (samples_per_class >= 50 && samples_per_class < 500) {
    if (time_budget > 20) {
        candidates <- frozen_and_finetuned()
    } else {
        candidates <- frozen_only()
    }
} else if (samples_per_class >= 500) {
    if (n_labels >= 4 && complexity > 12) {
        candidates <- frozen_finetuned_moe()
    } else {
        candidates <- frozen_and_finetuned()
    }
}

# After (proposed)
enabled_models <- filter_models(model_registry, meta_features, time_budget)
candidates <- generate_search_space(enabled_models, bayesian_sampler)
```

## Expected Impact

| Metric | Current | With Improvements | Gain |
|--------|---------|-------------------|------|
| Search efficiency | 10-15 trials | 30-50 trials (same time) | 3x |
| Best model accuracy | Baseline | +1-2% | Better |
| Ensemble accuracy | Baseline | +2-3% | Better |
| Code complexity | High (nested if/else) | Low (declarative) | Cleaner |
| Extensibility | Hard to add models | Easy (registry) | Easier |

## References

- AutoGluon HPO: https://auto.gluon.ai/dev/tutorials/multimodal/advanced_topics/hyperparameter_optimization.html
- ASHA paper: https://arxiv.org/abs/1810.05934
- PSEO paper: https://arxiv.org/abs/2508.05144
- ParBayesianOptimization: https://github.com/AnotherSamWilson/ParBayesianOptimization
