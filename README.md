# graniteR

R interface to IBM Granite embedding models via Python's transformers library.

## Installation

```r
# Install from source
devtools::install_github("skandermulder/graniteR")

# Install Python dependencies
library(graniteR)
install_granite()
```

## Usage

### Generate Embeddings

```r
library(graniteR)
library(dplyr)

data <- tibble(
  id = 1:3,
  text = c(
    "This is a positive sentence",
    "This is a negative sentence",
    "This is a neutral sentence"
  )
)

embeddings <- data |>
  granite_embed(text)

head(embeddings)
```

### Text Classification

```r
train_data <- tibble(
  text = c(
    "I love this product",
    "This is terrible",
    "Great experience",
    "Very disappointing"
  ),
  label = c(1, 0, 1, 0)
)

classifier <- granite_classifier(num_labels = 2) |>
  granite_train(
    train_data,
    text,
    label,
    epochs = 3,
    batch_size = 2
  )

new_data <- tibble(
  text = c("Amazing product", "Not good")
)

predictions <- granite_predict(classifier, new_data, text)
```

## Features

- Pipe-friendly interface following tidyverse conventions
- Support for sentence embeddings with Granite-R2
- Text classification with fine-tuning
- GPU acceleration support
- Minimal dependencies

## Model

The package uses IBM's Granite Embedding English R2 model by default:
- Model: `ibm-granite/granite-embedding-english-r2`
- Size: 149M parameters
- Embedding dimension: 768
- Max sequence length: 512 tokens

## License

MIT
