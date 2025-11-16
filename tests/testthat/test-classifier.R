test_that("granite_classifier creates a valid classifier object", {
  skip_if_no_python_or_modules()

  # Test with explicit num_labels
  classifier_explicit <- granite_classifier(num_labels = 2)
  expect_s3_class(classifier_explicit, "granite_classifier")
  expect_equal(classifier_explicit$num_labels, 2)
  expect_false(classifier_explicit$is_trained)

  # Test with inferred num_labels
  data <- tibble::tibble(
    text = c("This is a positive review", "This is a negative review"),
    label = c("positive", "negative")
  )
  classifier_inferred <- granite_classifier(data = data, label_col = label)
  expect_s3_class(classifier_inferred, "granite_classifier")
  expect_equal(classifier_inferred$num_labels, 2)
})

test_that("granite_train runs and updates the classifier", {
  skip_if_no_python_or_modules()

  data <- tibble::tibble(
    text = c("positive", "negative", "positive", "negative"),
    label = c(1, 0, 1, 0)
  )
  classifier <- granite_classifier(num_labels = 2)

  # Just check that it runs without error and returns a trained classifier
  trained_classifier <- granite_train(
    classifier,
    data,
    text_col = text,
    label_col = label,
    epochs = 1,
    batch_size = 2,
    validation_split = 0.5,
    verbose = FALSE # Keep verbose FALSE to avoid printing during tests
  )

  expect_true(trained_classifier$is_trained)
  expect_s3_class(trained_classifier, "granite_classifier")
})

test_that("granite_predict returns predictions", {
  skip_if_no_python_or_modules()

  train_data <- tibble::tibble(
    text = c("positive", "negative", "positive", "negative"),
    label = c(1, 0, 1, 0)
  )
  classifier <- granite_classifier(num_labels = 2) |>
    granite_train(train_data, text, label, epochs = 1, verbose = FALSE)

  new_data <- tibble::tibble(
    text = c("a new positive example", "a new negative one")
  )

  # Test "class" prediction
  preds_class <- granite_predict(classifier, new_data, text, type = "class")
  expect_s3_class(preds_class, "tbl_df")
  expect_true("prediction" %in% names(preds_class))
  expect_equal(nrow(preds_class), 2)
  expect_true(is.numeric(preds_class$prediction))

  # Test "prob" prediction
  preds_prob <- granite_predict(classifier, new_data, text, type = "prob")
  expect_s3_class(preds_prob, "tbl_df")
  expect_true(all(c("prob_1", "prob_2") %in% names(preds_prob)))
  expect_equal(nrow(preds_prob), 2)
  expect_true(is.numeric(preds_prob$prob_1))
})