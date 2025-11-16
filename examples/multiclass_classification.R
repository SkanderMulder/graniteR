# Multi-class Classification Example with graniteR
# Demonstrates classifying text into 4 priority levels: critical, high, medium, low

library(graniteR)
library(dplyr)
library(tibble)

# ============================================================================
# Example: Support Ticket Priority Classification
# ============================================================================

# Create training data with 4 priority levels
tickets <- tibble(
  text = c(
    "System is completely down, all users affected",
    "Password reset request for one user",
    "Feature request for dark mode",
    "Unable to login, getting error 500",
    "Question about using the API",
    "Critical security vulnerability found",
    "Typo in documentation page",
    "Dashboard not loading data",
    "Server crashed and won't restart",
    "Can you add export to PDF feature?",
    "Email notifications not working",
    "How do I change my username?",
    "Data breach - immediate attention needed",
    "Minor UI alignment issue",
    "Performance degradation noticed",
    "Documentation link is broken"
  ),
  priority = c(
    "critical", "low", "low", "high", 
    "medium", "critical", "low", "high",
    "critical", "low", "high", "medium",
    "critical", "low", "medium", "low"
  )
)

cat("=== Training Data ===\n")
print(tickets)

# Check the unique classes
unique_priorities <- unique(tickets$priority)
n_classes <- length(unique_priorities)
cat("\nUnique classes:", paste(unique_priorities, collapse = ", "), "\n")
cat("Number of classes:", n_classes, "\n")

# ============================================================================
# Create and Train Multi-class Classifier
# ============================================================================

cat("\n=== Training Classifier ===\n")

# IMPORTANT: num_labels must match the number of unique classes
classifier <- granite_classifier(num_labels = n_classes)

# Train the model
# Character labels are automatically converted to integers in alphabetical order
# critical -> 0, high -> 1, low -> 2, medium -> 3
classifier <- granite_train(
  classifier, 
  tickets, 
  text_col = text, 
  label_col = priority,
  epochs = 5,
  learning_rate = 2e-5,
  validation_split = 0.2
)

# ============================================================================
# Make Predictions
# ============================================================================

cat("\n=== Making Predictions ===\n")

# New tickets to classify
new_tickets <- tibble(
  text = c(
    "Database connection timeout for 30 minutes",
    "Update user profile picture functionality",
    "System performance is somewhat slow",
    "Complete data center failure"
  )
)

cat("\nNew tickets to classify:\n")
print(new_tickets)

# Get class predictions (returns predicted class as integer)
predictions <- granite_predict(classifier, new_tickets, text_col = text, type = "class")
cat("\nClass predictions (as integers):\n")
print(predictions)

# Get probability distributions across all classes
probabilities <- granite_predict(classifier, new_tickets, text_col = text, type = "prob")
cat("\nProbability distributions:\n")
print(probabilities)

# ============================================================================
# Understanding Label Mapping
# ============================================================================

cat("\n=== Label Mapping ===\n")
cat("The classifier automatically converts character labels to integers.\n")
cat("Conversion is based on alphabetical order:\n\n")

label_mapping <- data.frame(
  integer = 0:(n_classes - 1),
  label = levels(as.factor(tickets$priority))
)
print(label_mapping)

# ============================================================================
# Tips for Multi-class Classification
# ============================================================================

cat("\n=== Tips ===\n")
cat("1. Ensure num_labels matches the number of unique classes\n")
cat("2. Labels are converted alphabetically (use factor with explicit levels for custom order)\n")
cat("3. Use validation_split to monitor performance across all classes\n")
cat("4. Consider class imbalance - may need more data for rare classes\n")
cat("5. type='prob' gives you confidence scores for all classes\n")

# ============================================================================
# Using Factors for Custom Label Order
# ============================================================================

cat("\n=== Using Factors for Custom Order ===\n")

# Create data with factor labels in custom order
tickets_factor <- tickets |>
  mutate(priority = factor(priority, levels = c("low", "medium", "high", "critical")))

cat("Custom factor levels:", levels(tickets_factor$priority), "\n")
cat("This ensures: low=0, medium=1, high=2, critical=3\n")

# Train with factor labels
classifier_ordered <- granite_classifier(num_labels = 4)
classifier_ordered <- granite_train(
  classifier_ordered, 
  tickets_factor, 
  text_col = text, 
  label_col = priority,
  epochs = 3
)

cat("\n✓ Multi-class classification complete!\n")
