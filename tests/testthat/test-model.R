test_that("granite_model creates model object", {
  skip_if_no_python_or_modules()

  model <- granite_model(task = "embedding")
  expect_s3_class(model, "granite_model")
  expect_equal(model$task, "embedding")
})

test_that("granite_tokenizer creates tokenizer object", {
  skip_if_no_python_or_modules()

  tokenizer <- granite_tokenizer()
  expect_s3_class(tokenizer, "granite_tokenizer")
})

test_that("granite_classifier requires num_labels", {
  skip_if_no_python_or_modules()

  expect_error(
    granite_model(task = "classification"),
    "num_labels must be specified"
  )
})

test_that("granite_model creates classification model", {
  skip_if_no_python_or_modules()
  
  model <- granite_model(task = "classification", num_labels = 3)
  
  expect_s3_class(model, "granite_model")
  expect_equal(model$task, "classification")
  expect_equal(model$num_labels, 3)
})

test_that("granite_model creates regression model", {
  skip_if_no_python_or_modules()
  
  model <- granite_model(task = "regression")
  
  expect_s3_class(model, "granite_model")
  expect_equal(model$task, "regression")
})

test_that("granite_model handles CUDA device when available", {
  skip_if_no_python_or_modules()
  skip_on_cran()
  
  torch <- reticulate::import("torch")
  cuda_available <- torch$cuda$is_available()
  
  if (cuda_available) {
    model <- granite_model(device = "cuda")
    expect_equal(model$device, "cuda")
  } else {
    expect_warning(
      model <- granite_model(device = "cuda"),
      "CUDA device requested but not available"
    )
    expect_equal(model$device, "cpu")
  }
})

test_that("granite_model defaults to CPU", {
  skip_if_no_python_or_modules()
  
  model <- granite_model()
  expect_equal(model$device, "cpu")
})

test_that("granite_model handles custom model name", {
  skip_if_no_python_or_modules()
  skip_on_cran()
  
  model <- granite_model(model_name = "ibm-granite/granite-embedding-english-r2")
  expect_equal(model$model_name, "ibm-granite/granite-embedding-english-r2")
})

test_that("granite_tokenizer handles custom model name", {
  skip_if_no_python_or_modules()
  skip_on_cran()
  
  tokenizer <- granite_tokenizer(model_name = "ibm-granite/granite-embedding-english-r2")
  expect_equal(tokenizer$model_name, "ibm-granite/granite-embedding-english-r2")
})

test_that("print.granite_model outputs correctly", {
  skip_if_no_python_or_modules()
  
  model <- granite_model(task = "embedding")
  
  output <- capture.output(print(model))
  expect_true(any(grepl("Granite Model", output)))
  expect_true(any(grepl("Task:", output)))
})

test_that("print.granite_model shows num_labels for classification", {
  skip_if_no_python_or_modules()
  
  model <- granite_model(task = "classification", num_labels = 5)
  
  output <- capture.output(print(model))
  expect_true(any(grepl("Labels:", output)))
})

test_that("print.granite_tokenizer outputs correctly", {
  skip_if_no_python_or_modules()
  
  tokenizer <- granite_tokenizer()
  
  output <- capture.output(print(tokenizer))
  expect_true(any(grepl("Granite Tokenizer", output)))
  expect_true(any(grepl("Model:", output)))
})

test_that("granite_model handles invalid task", {
  skip_if_no_python_or_modules()
  
  expect_error(
    granite_model(task = "invalid_task"),
    "'arg' should be one of"
  )
})

test_that("print.granite_model without num_labels", {
  skip_if_no_python_or_modules()
  
  model <- granite_model(task = "embedding")
  
  output <- capture.output(print(model))
  expect_true(any(grepl("Granite Model", output)))
  expect_false(any(grepl("Labels:", output)))
})

test_that("granite_model handles CUDA unavailable scenario", {
  skip_if_no_python_or_modules()
  skip_on_cran()
  
  mock_torch <- list(
    cuda = list(is_available = function() FALSE)
  )
  
  with_mocked_bindings(
    torch = mock_torch,
    {
      expect_warning(
        model <- granite_model(device = "cuda"),
        "CUDA device requested but not available"
      )
      expect_equal(model$device, "cpu")
    }
  )
})
