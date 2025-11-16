test_that("check_model validates model", {
  expect_error(graniteR:::check_model(NULL), "Model is NULL")
})

test_that("check_model returns model invisibly", {
  model <- list(model = "test")
  result <- graniteR:::check_model(model)
  expect_identical(result, model)
})

test_that("check_tokenizer validates tokenizer", {
  expect_error(graniteR:::check_tokenizer(NULL), "Tokenizer is NULL")
})

test_that("check_tokenizer returns tokenizer invisibly", {
  tokenizer <- list(tokenizer = "test")
  result <- graniteR:::check_tokenizer(tokenizer)
  expect_identical(result, tokenizer)
})

test_that("granite_check_system reports Python availability", {
  skip_on_cran()
  
  output <- capture.output({
    result <- suppressWarnings(granite_check_system())
  })
  
  expect_type(result, "list")
  expect_true("python_available" %in% names(result))
  expect_true("transformers" %in% names(result))
  expect_true("torch" %in% names(result))
  expect_true("cuda" %in% names(result))
})

test_that("granite_check_system handles Python not available", {
  skip_on_cran()
  
  with_mocked_bindings(
    py_available = function(...) FALSE,
    .package = "reticulate",
    {
      output <- capture.output({
        result <- suppressWarnings(granite_check_system())
      })
      
      expect_false(result$python_available)
      expect_false(result$transformers)
      expect_false(result$torch)
      expect_false(result$cuda)
    }
  )
})

test_that("granite_check_system handles missing Python packages", {
  skip_on_cran()
  
  with_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() list(python = "/usr/bin/python", version = "3.8.0"),
    py_module_available = function(module) FALSE,
    .package = "reticulate",
    {
      output <- capture.output({
        result <- suppressWarnings(granite_check_system())
      })
      
      expect_true(result$python_available)
      expect_false(result$transformers)
      expect_false(result$torch)
    }
  )
})

test_that("granite_check_system detects CUDA when available", {
  skip_on_cran()
  skip_if_not(reticulate::py_module_available("torch"))
  
  mock_torch <- list(
    cuda = list(
      is_available = function() TRUE,
      device_count = function() 1L
    ),
    version = list(cuda = "11.8")
  )
  
  with_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() list(python = "/usr/bin/python", version = "3.8.0"),
    py_module_available = function(module) TRUE,
    import = function(...) mock_torch,
    .package = "reticulate",
    {
      output <- capture.output({
        result <- suppressWarnings(granite_check_system())
      })
      
      expect_true(result$python_available)
      expect_true(result$cuda)
    }
  )
})

test_that("granite_check_system handles CUDA not available", {
  skip_on_cran()
  skip_if_not(reticulate::py_module_available("torch"))
  
  mock_torch <- list(
    cuda = list(
      is_available = function() FALSE
    )
  )
  
  with_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() list(python = "/usr/bin/python", version = "3.8.0"),
    py_module_available = function(module) TRUE,
    import = function(...) mock_torch,
    .package = "reticulate",
    {
      output <- capture.output({
        result <- suppressWarnings(granite_check_system())
      })
      
      expect_true(result$python_available)
      expect_false(result$cuda)
    }
  )
})

test_that("granite_check_system handles torch import error", {
  skip_on_cran()
  
  with_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() list(python = "/usr/bin/python", version = "3.8.0"),
    py_module_available = function(module) {
      if (module == "torch") TRUE else FALSE
    },
    import = function(...) stop("Import error"),
    .package = "reticulate",
    {
      output <- capture.output({
        result <- suppressWarnings(granite_check_system())
      })
      
      expect_true(result$python_available)
      expect_false(result$cuda)
    }
  )
})

test_that("granite_check_system handles py_config version as list", {
  skip_on_cran()
  
  with_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() list(
      python = "/usr/bin/python", 
      version = list("3.8.0", "extra")
    ),
    py_module_available = function(module) FALSE,
    .package = "reticulate",
    {
      output <- capture.output({
        result <- suppressWarnings(granite_check_system())
      })
      
      expect_true(result$python_available)
      expect_true(any(grepl("3.8.0", output)))
    }
  )
})

test_that("granite_check_system handles CUDA device count", {
  skip_on_cran()
  skip_if_not(reticulate::py_module_available("torch"))
  
  mock_torch <- list(
    cuda = list(
      is_available = function() TRUE,
      device_count = function() 2L
    ),
    version = list(cuda = "12.0")
  )
  
  with_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() list(python = "/usr/bin/python", version = "3.8.0"),
    py_module_available = function(module) TRUE,
    import = function(...) mock_torch,
    .package = "reticulate",
    {
      output <- capture.output({
        result <- suppressWarnings(granite_check_system())
      })
      
      expect_true(result$python_available)
      expect_true(result$cuda)
      expect_true(any(grepl("CUDA devices: 2", output)))
    }
  )
})
