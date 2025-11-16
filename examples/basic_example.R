## Basic graniteR Example
## This script demonstrates the core functionality of graniteR

library(graniteR)
library(dplyr)
library(tibble)

# Example 1: Generate embeddings
data <- tibble(
  id = 1:5,
  text = c(
    "The weather is beautiful today",
    "I love programming in R",
    "Machine learning is fascinating",
    "The sun is shining brightly",
    "Data science is my passion"
  )
)

embeddings <- data |>
  granite_embed(text_col = text)

cat("Generated embeddings with dimensions:", dim(embeddings), "\n\n")

# Example 2: Text classification
train_data <- tibble(
  text = c(
    "This movie was absolutely fantastic!",
    "Terrible film, waste of time",
    "Loved every minute of it",
    "Boring and disappointing",
    "Best movie I've seen this year",
    "Awful acting and poor plot",
    "Highly recommended!",
    "Don't waste your money"
  ),
  sentiment = c(1, 0, 1, 0, 1, 0, 1, 0)
)

cat("Training classifier...\n")
classifier <- granite_classifier(num_labels = 2) |>
  granite_train(
    train_data,
    text_col = text,
    label_col = sentiment,
    epochs = 3,
    batch_size = 2,
    learning_rate = 2e-5
  )

test_data <- tibble(
  text = c(
    "Amazing experience!",
    "Not impressed at all"
  )
)

cat("\nMaking predictions...\n")
predictions <- granite_predict(classifier, test_data, text_col = text, type = "class")
print(predictions)

probs <- granite_predict(classifier, test_data, text_col = text, type = "prob")
print(probs)
