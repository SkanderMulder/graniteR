test_that("granite_embed adds embedding columns", {
  skip_if_no_python_or_modules()

  data <- tibble::tibble(text = c("Hello", "World"))
  result <- granite_embed(data, text)

  expect_true(ncol(result) > ncol(data))
  expect_true(any(grepl("^emb_", names(result))))
})

test_that("granite_embed handles missing column", {
  skip_if_no_python_or_modules()

  data <- tibble::tibble(other = c("Hello", "World"))
  expect_error(
    granite_embed(data, text),
    "Column 'text' not found"
  )
})

test_that("granite_embed accepts custom model and tokenizer", {
  skip_if_no_python_or_modules()
  
  model <- granite_model(task = "embedding")
  tokenizer <- granite_tokenizer()
  
  data <- tibble::tibble(text = c("Hello", "World"))
  result <- granite_embed(data, text, model = model, tokenizer = tokenizer)
  
  expect_true(ncol(result) > ncol(data))
  expect_true(any(grepl("^emb_", names(result))))
})

test_that("granite_embed handles custom batch size", {
  skip_if_no_python_or_modules()
  
  data <- tibble::tibble(text = rep("test", 10))
  result <- granite_embed(data, text, batch_size = 3)
  
  expect_equal(nrow(result), 10)
  expect_true(any(grepl("^emb_", names(result))))
})

test_that("granite_embed handles single row", {
  skip_if_no_python_or_modules()
  
  data <- tibble::tibble(text = "Single text")
  result <- granite_embed(data, text)
  
  expect_equal(nrow(result), 1)
  expect_true(any(grepl("^emb_", names(result))))
})

test_that("granite_embed creates model when NULL", {
  skip_if_no_python_or_modules()
  
  data <- tibble::tibble(text = c("Test"))
  result <- granite_embed(data, text, model = NULL, tokenizer = NULL)
  
  expect_true(any(grepl("^emb_", names(result))))
})

test_that("granite_embed handles custom column name", {
  skip_if_no_python_or_modules()
  
  data <- tibble::tibble(content = c("Hello", "World"))
  result <- granite_embed(data, content)
  
  expect_true(any(grepl("^emb_", names(result))))
  expect_true("content" %in% names(result))
})

test_that("granite_embed preserves original data", {
  skip_if_no_python_or_modules()
  
  data <- tibble::tibble(
    id = c(1, 2),
    text = c("Hello", "World"),
    label = c("A", "B")
  )
  result <- granite_embed(data, text)
  
  expect_true(all(c("id", "text", "label") %in% names(result)))
  expect_equal(result$id, data$id)
  expect_equal(result$label, data$label)
})

test_that("granite_embed handles CUDA device", {
  skip_if_no_python_or_modules()
  skip_on_cran()
  
  torch <- reticulate::import("torch")
  cuda_available <- torch$cuda$is_available()
  
  if (cuda_available) {
    model <- granite_model(task = "embedding", device = "cuda")
    tokenizer <- granite_tokenizer()
    
    data <- tibble::tibble(text = c("Test"))
    result <- granite_embed(data, text, model = model, tokenizer = tokenizer)
    
    expect_true(any(grepl("^emb_", names(result))))
  } else {
    skip("CUDA not available")
  }
})

test_that("granite_embed handles multiple batches", {
  skip_if_no_python_or_modules()
  skip_on_cran()
  
  data <- tibble::tibble(text = rep("test", 15))
  result <- granite_embed(data, text, batch_size = 5)
  
  expect_equal(nrow(result), 15)
  expect_true(any(grepl("^emb_", names(result))))
})
