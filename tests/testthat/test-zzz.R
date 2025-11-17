library(testthat)

test_that("install_pyenv checks for processx package", {
  skip_on_cran()

  # Mock environment without processx
  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) FALSE,
    .package = "base",
    {
      expect_error(
        install_pyenv(),
        "Package 'processx' is required"
      )
    }
  )
})

test_that("install_pyenv handles UV not available", {
  skip_on_cran()
  skip_if_not_installed("processx")

  with_mocked_bindings(
    Sys.which = function(x) "",
    .package = "base",
    {
      result <- suppressMessages(install_pyenv())
      expect_false(result)
    }
  )
})

test_that("install_pyenv creates venv path", {
  skip_on_cran()
  skip_if_not_installed("processx")
  skip_if_not_installed("mockery")

  temp_dir <- tempdir()
  venv_path <- file.path(temp_dir, "test_venv")

  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    .package = "base",
    {
      mockery::stub(install_pyenv, "system.file", function(...) temp_dir)
      mockery::stub(install_pyenv, "dir.exists", function(...) FALSE)
      mockery::stub(install_pyenv, "file.exists", function(...) FALSE)

      result <- suppressMessages(install_pyenv(venv_path = "test_venv"))
      expect_true(result)
    }
  )
})

test_that("install_pyenv uses existing venv", {
  skip_on_cran()
  skip_if_not_installed("processx")
  skip_if_not_installed("mockery")

  temp_dir <- tempdir()

  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    .package = "base",
    {
      mockery::stub(install_pyenv, "system.file", function(...) temp_dir)
      mockery::stub(install_pyenv, "dir.exists", function(...) TRUE)
      result <- suppressMessages(install_pyenv())
      expect_true(result)
    }
  )
})

test_that("install_pyenv uses requirements.txt when available", {
  skip_on_cran()
  skip_if_not_installed("processx")
  skip_if_not_installed("mockery")

  temp_dir <- tempdir()

  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    .package = "base",
    {
      mockery::stub(install_pyenv, "processx::run", function(cmd, args, ...) {
        list(status = 0, stdout = "uv 0.1.0")
      })

      mockery::stub(install_pyenv, "system.file", function(...) temp_dir)
      mockery::stub(install_pyenv, "dir.exists", function(...) TRUE)
      mockery::stub(install_pyenv, "file.exists", function(...) TRUE)

      result <- suppressMessages(install_pyenv())
      expect_true(result)
    }
  )
})

test_that(".onLoad initializes Python modules when available", {
  skip_on_cran()
  skip_if_not_installed("mockery")
  
  local_bindings(transformers = NULL, torch = NULL, .env = asNamespace("graniteR"))
  
  # Mock reticulate functions
  with_mocked_bindings(
    py_available = function(initialize) {
      expect_true(initialize)
      TRUE
    },
    import = function(module, delay_load) {
      expect_false(delay_load)
      if (module == "transformers") {
        "mock_transformers"
      } else if (module == "torch") {
        "mock_torch"
      } else {
        stop("Unexpected module import")
      }
    },
    .package = "reticulate",
    {
      .onLoad(NULL, "graniteR")
      
      expect_equal(get("transformers", envir = asNamespace("graniteR")), "mock_transformers")
      expect_equal(get("torch", envir = asNamespace("graniteR")), "mock_torch")
    }
  )
})

test_that(".onLoad handles Python not available", {
  skip_on_cran()
  skip_if_not_installed("mockery")
  
  local_bindings(transformers = NULL, torch = NULL, .env = asNamespace("graniteR"))
  
  with_mocked_bindings(
    py_available = function(...) FALSE,
    .package = "reticulate",
    {
      .onLoad(NULL, "graniteR")
      
      expect_null(get("transformers", envir = asNamespace("graniteR")))
      expect_null(get("torch", envir = asNamespace("graniteR")))
    }
  )
})

test_that(".onLoad handles import errors silently", {
  skip_on_cran()
  skip_if_not_installed("mockery")
  
  local_bindings(transformers = NULL, torch = NULL, .env = asNamespace("graniteR"))
  
  with_mocked_bindings(
    py_available = function(...) TRUE,
    import = function(module, delay_load) {
      if (module == "transformers") {
        stop("Mock import error for transformers")
      } else if (module == "torch") {
        "mock_torch"
      }
    },
    .package = "reticulate",
    {
      # Should not throw an error
      expect_no_error(.onLoad(NULL, "graniteR"))
      
      expect_null(get("transformers", envir = asNamespace("graniteR")))
      expect_equal(get("torch", envir = asNamespace("graniteR")), "mock_torch")
    }
  )
})

test_that(".onAttach shows message when Python dependencies are missing", {
  skip_on_cran()
  skip_if_not_installed("mockery")
  
  local_bindings(transformers = NULL, torch = NULL, .env = asNamespace("graniteR"))
  
  with_mocked_bindings(
    packageStartupMessage = function(...) {
      captured_message <<- paste0(list(...), collapse = "")
    },
    .package = "base",
    {
      captured_message <- ""
      .onAttach(NULL, "graniteR")
      
      expect_true(grepl("Python dependencies not found. Run install_pyenv() to set up.", captured_message))
    }
  )
})

test_that(".onAttach does not show message when Python dependencies are present", {
  skip_on_cran()
  skip_if_not_installed("mockery")
  
  local_bindings(transformers = "mock_transformers", torch = "mock_torch", .env = asNamespace("graniteR"))
  
  with_mocked_bindings(
    packageStartupMessage = function(...) {
      captured_message <<- paste0(list(...), collapse = "")
    },
    .package = "base",
    {
      captured_message <- ""
      .onAttach(NULL, "graniteR")
      
      expect_equal(captured_message, "")
    }
  )
})