test_that("granite_model creates model object", {
  skip_if_not(reticulate::py_module_available("transformers"))

  model <- granite_model(task = "embedding")
  expect_s3_class(model, "granite_model")
  expect_equal(model$task, "embedding")
})

test_that("granite_tokenizer creates tokenizer object", {
  skip_if_not(reticulate::py_module_available("transformers"))

  tokenizer <- granite_tokenizer()
  expect_s3_class(tokenizer, "granite_tokenizer")
})

test_that("granite_classifier requires num_labels", {
  skip_if_not(reticulate::py_module_available("transformers"))

  expect_error(
    granite_model(task = "classification"),
    "num_labels must be specified"
  )
})
