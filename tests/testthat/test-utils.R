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

test_that("to_device handles CPU device correctly", {
  encodings <- list(input_ids = "cpu_input", attention_mask = "cpu_mask")
  labels <- "cpu_labels"
  
  result <- graniteR:::to_device(encodings, labels, device = "cpu")
  
  expect_equal(result$encodings$input_ids, "cpu_input")
  expect_equal(result$encodings$attention_mask, "cpu_mask")
  expect_equal(result$labels, "cpu_labels")
})

test_that("to_device moves encodings to CUDA device when labels are NULL", {
  skip_on_cran()
  skip_if_no_python_or_modules()
  
  mock_cuda_device <- "cuda_device_object"
  mock_input_ids <- list(to = function(device) {
    expect_equal(device, mock_cuda_device)
    "cuda_input"
  })
  mock_attention_mask <- list(to = function(device) {
    expect_equal(device, mock_cuda_device)
    "cuda_mask"
  })
  
  mock_torch <- list(
    device = function(type) {
      expect_equal(type, "cuda")
      mock_cuda_device
    }
  )
  
  with_mocked_bindings(
    import = function(module) {
      if (module == "torch") mock_torch else stop("Unexpected import")
    },
    .package = "reticulate",
    {
      encodings <- list(input_ids = mock_input_ids, attention_mask = mock_attention_mask)
      result <- graniteR:::to_device(encodings, labels = NULL, device = "cuda")
      
      expect_equal(result$encodings$input_ids, "cuda_input")
      expect_equal(result$encodings$attention_mask, "cuda_mask")
      expect_null(result$labels)
    }
  )
})

test_that("to_device moves encodings and labels to CUDA device", {
  skip_on_cran()
  skip_if_no_python_or_modules()
  
  mock_cuda_device <- "cuda_device_object"
  mock_input_ids <- list(to = function(device) {
    expect_equal(device, mock_cuda_device)
    "cuda_input"
  })
  mock_attention_mask <- list(to = function(device) {
    expect_equal(device, mock_cuda_device)
    "cuda_mask"
  })
  mock_labels <- list(to = function(device) {
    expect_equal(device, mock_cuda_device)
    "cuda_labels"
  })
  
  mock_torch <- list(
    device = function(type) {
      expect_equal(type, "cuda")
      mock_cuda_device
    }
  )
  
  with_mocked_bindings(
    import = function(module) {
      if (module == "torch") mock_torch else stop("Unexpected import")
    },
    .package = "reticulate",
    {
      encodings <- list(input_ids = mock_input_ids, attention_mask = mock_attention_mask)
      result <- graniteR:::to_device(encodings, labels = mock_labels, device = "cuda")
      
      expect_equal(result$encodings$input_ids, "cuda_input")
      expect_equal(result$encodings$attention_mask, "cuda_mask")
      expect_equal(result$labels, "cuda_labels")
    }
  )
})

