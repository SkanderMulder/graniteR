# graniteR Examples

This directory contains practical examples demonstrating various features of the graniteR package.

## Available Examples

### 1. `basic_example.R`
Introduction to graniteR with simple embedding and classification tasks.

**Features covered:**
- Generating text embeddings
- Binary classification
- Making predictions

**Use case:** Basic sentiment analysis

---

### 2. `multiclass_classification.R` 🆕
Comprehensive guide to multi-class classification with graniteR.

**Features covered:**
- 4-class classification (critical, high, medium, low priority)
- Automatic label conversion from characters to integers
- Custom factor levels for controlled label ordering
- Probability distributions for all classes
- Understanding label mappings

**Use case:** Support ticket priority classification

**Key learnings:**
- `num_labels` must match the number of unique classes
- Character labels are converted alphabetically by default
- Use factors with explicit levels for custom ordering
- Both `type="class"` and `type="prob"` predictions supported

---

## Running Examples

All examples assume you have:
1. Installed graniteR: `devtools::install_github("skandermulder/graniteR")`
2. Set up Python dependencies: `install_granite()`

To run an example:

```r
source("examples/multiclass_classification.R")
```

Or from within the package:

```r
system.file("examples", "multiclass_classification.R", package = "graniteR") |>
  source()
```

---

## Classification Support

graniteR supports:

✅ **Binary Classification** (2 classes)
```r
clf <- classifier(num_labels = 2)
```

✅ **Multi-class Classification** (3+ classes)
```r
clf <- classifier(num_labels = 4)  # e.g., critical, high, medium, low
```

✅ **Character Labels** - automatically converted
```r
labels <- c("high", "medium", "low", "unknown")
```

✅ **Factor Labels** - respects level ordering
```r
labels <- factor(c("high", "low"), levels = c("low", "medium", "high"))
```

✅ **Integer Labels** - used directly
```r
labels <- c(0, 1, 2, 3)
```

---

## Tips for Multi-class Classification

1. **Set correct `num_labels`**: Must match the number of unique classes in your data
2. **Understand label mapping**: Check how your labels map to integers with `levels(as.factor(labels))`
3. **Use factors for control**: Create explicit factor levels when you need specific integer assignments
4. **Check class balance**: Ensure you have sufficient examples for each class
5. **Monitor all classes**: Use validation split to see performance across all categories
6. **Get probabilities**: Use `type="prob"` to see confidence scores for all classes

---

## Need More Help?

- See vignettes: `vignette("getting-started", package = "graniteR")`
- Check documentation: `?classifier`
- Multi-class is mentioned in: `vignette("technical-approaches")`

---

## Contributing Examples

Have a great use case? Contributions welcome! Please:
1. Follow the existing example format
2. Include comments explaining key concepts
3. Use realistic, relatable scenarios
4. Update this README
