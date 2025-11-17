# Integration Documentation: extractoR, SpinneR, and Other Packages

## Overview
Create comprehensive documentation showing how graniteR integrates with related packages in the ecosystem, particularly extractoR and SpinneR (same author).

## Motivation
The overview mentions: "Pair it with extractoR for LLM-enhanced pipelines or SpinneR for training visuals"

## Integration Opportunities

### 1. graniteR + extractoR
**extractoR**: LLM-based data extraction from text

**Integration Use Cases**:
```r
# Pipeline: Extract structured data, then classify
library(graniteR)
library(extractoR)

# Step 1: Extract entities with extractoR
emails <- tibble(
  text = c("From: john@example.com...", "From: spam@badactor.com...")
)

extracted <- emails |>
  extract_llm(text, schema = list(
    sender = "email address",
    intent = "purchase, support, or spam",
    urgency = "high, medium, or low"
  ))

# Step 2: Embed or classify with graniteR
embeddings <- extracted |>
  embed(text)

# Step 3: Train classifier on extracted features
clf <- classifier(num_labels = 3) |>
  train(extracted, text, label_col = intent)
```

**Integration Points**:
- extractoR for schema-based extraction → graniteR for embedding/classification
- Combined feature engineering (LLM-extracted features + embeddings)
- Hybrid pipelines (rule-based extraction + ML classification)

### 2. graniteR + SpinneR
**SpinneR**: Training visualization and monitoring

**Integration Use Cases**:
```r
# Visualize graniteR training with SpinneR
library(SpinneR)

clf <- classifier(num_labels = 2) |>
  train(data, text, label, 
        epochs = 10,
        callback = spinner_callback()  # SpinneR integration
  )

# Or manually log metrics
spinner_init("graniteR Training")
for (epoch in 1:10) {
  metrics <- train_epoch(clf, data)
  spinner_update(
    epoch = epoch,
    loss = metrics$loss,
    accuracy = metrics$accuracy
  )
}
spinner_finish()
```

**Integration Points**:
- Real-time training visualization
- Loss/accuracy tracking
- Progress bars and ETAs
- Model comparison dashboards

### 3. graniteR + tidymodels
**tidymodels**: ML workflow framework

```r
library(tidymodels)
library(graniteR)

# Define graniteR as parsnip model
graniteR_spec <- function() {
  parsnip::boost_tree() |>  # Placeholder
    set_engine("graniteR") |>
    set_mode("classification")
}

# Use in tidymodels workflow
workflow() |>
  add_recipe(
    recipe(label ~ text, data = train_data)
  ) |>
  add_model(graniteR_spec()) |>
  fit(train_data)
```

### 4. graniteR + text/spacyr
**text**: Text mining in R
**spacyr**: spaCy integration

```r
# Combine linguistic features with embeddings
library(spacyr)
library(graniteR)

# Extract linguistic features
parsed <- spacy_parse(data$text)
linguistic_feats <- extract_entity(parsed)

# Get graniteR embeddings
embeddings <- embed(data, text)

# Combine for downstream tasks
combined <- bind_cols(linguistic_feats, embeddings)
```

### 5. graniteR + targets/drake
**targets**: Pipeline orchestration

```r
# _targets.R
library(targets)
library(graniteR)

tar_plan(
  # Download data
  tar_target(raw_data, download_data()),
  
  # Embed texts
  tar_target(embeddings, embed(raw_data, text)),
  
  # Train classifier
  tar_target(model, {
    classifier(num_labels = 2) |>
      train(raw_data, text, label, epochs = 5)
  }),
  
  # Save model
  tar_target(saved_model, {
    save_model(model, "models/clf.rds")
    "models/clf.rds"
  })
)
```

## Documentation Deliverables

### 1. Integration Vignettes
Create separate vignettes for each integration:

- `vignettes/integration-extractoR.Rmd`
  - LLM + Embeddings pipelines
  - Feature engineering workflows
  - End-to-end example

- `vignettes/integration-SpinneR.Rmd`
  - Training visualization
  - Model comparison
  - Hyperparameter tracking

- `vignettes/integration-tidymodels.Rmd`
  - parsnip engine implementation
  - Recipe integration
  - Tune grid search

- `vignettes/integration-targets.Rmd`
  - Pipeline definition
  - Caching strategies
  - Parallel training

### 2. Cross-Package Examples
Create `examples/integrations/` directory:

```
examples/integrations/
├── extractoR_pipeline.R
├── spinneR_dashboard.R
├── tidymodels_workflow.R
├── targets_pipeline/
│   ├── _targets.R
│   └── R/functions.R
└── README.md
```

### 3. Documentation Updates
- Update README.md with "Related Packages" section
- Add "Integrations" section to pkgdown site
- Cross-reference in function documentation

### 4. Blog Posts / Case Studies
- "Building LLM-Enhanced Text Classifiers with extractoR + graniteR"
- "Visualizing Transformer Training with SpinneR"
- "Production ML Pipelines with targets + graniteR"

## Technical Implementation

### Optional Dependencies
Update DESCRIPTION:
```
Suggests:
  extractoR,
  SpinneR,
  tidymodels,
  parsnip,
  recipes,
  targets,
  spacyr,
  ...
```

### Helper Functions
```r
# R/integrations.R

#' Convert graniteR model to tidymodels spec
#' @export
as_parsnip_spec <- function(model) {
  # Implementation
}

#' SpinneR callback for training
#' @export
spinner_callback <- function() {
  # Implementation
}

#' Extract features for targets pipeline
#' @export
granite_target <- function(...) {
  # Wrapper for targets compatibility
}
```

## Testing
- Add integration tests (if packages available)
- Use `skip_if_not_installed("extractoR")` pattern
- CI testing with integration packages

## Benefits
- Showcase graniteR in broader ecosystem
- Increase discoverability
- Provide real-world workflow examples
- Build community around related packages

## Priority
Medium - Enhances adoption but not critical for CRAN

## Timeline
- Phase 1 (extractoR + SpinneR): 1 week
- Phase 2 (tidymodels): 1 week
- Phase 3 (targets + others): 1 week
