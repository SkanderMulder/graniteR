test_that("granite_embed adds embedding columns", {
  skip_if_not(reticulate::py_module_available("transformers"))
  skip_if_not(reticulate::py_module_available("torch"))

  data <- tibble::tibble(text = c("Hello", "World"))
  result <- granite_embed(data, text)

  expect_true(ncol(result) > ncol(data))
  expect_true(any(grepl("^emb_", names(result))))
})

test_that("granite_embed handles missing column", {
  skip_if_not(reticulate::py_module_available("transformers"))

  data <- tibble::tibble(other = c("Hello", "World"))
  expect_error(
    granite_embed(data, text),
    "Column 'text' not found"
  )
})
