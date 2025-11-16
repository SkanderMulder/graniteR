test_that("install_granite checks for processx package", {
  skip_on_cran()
  
  # Mock environment without processx
  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) FALSE,
    {
      expect_error(
        install_granite(),
        "Package 'processx' is required"
      )
    }
  )
})

test_that("install_granite handles UV not available", {
  skip_on_cran()
  skip_if_not_installed("processx")
  
  # Mock UV not being available
  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    {
      mockery::stub(install_granite, "processx::run", function(...) stop("command not found"))
      
      result <- suppressMessages(install_granite())
      expect_false(result)
    }
  )
})

test_that("install_granite creates venv path", {
  skip_on_cran()
  skip_if_not_installed("processx")
  
  temp_dir <- tempdir()
  venv_path <- file.path(temp_dir, "test_venv")
  
  # Mock UV being available and successful runs
  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    {
      mockery::stub(install_granite, "processx::run", function(cmd, args, ...) {
        if (cmd == "uv" && args[1] == "--version") {
          return(list(status = 0, stdout = "uv 0.1.0"))
        }
        return(list(status = 0))
      })
      
      mockery::stub(install_granite, "system.file", function(...) temp_dir)
      mockery::stub(install_granite, "dir.exists", function(...) FALSE)
      mockery::stub(install_granite, "file.exists", function(...) FALSE)
      
      result <- suppressMessages(install_granite(venv_path = "test_venv"))
      expect_true(result)
    }
  )
})

test_that("install_granite uses existing venv", {
  skip_on_cran()
  skip_if_not_installed("processx")
  
  temp_dir <- tempdir()
  
  
  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    {
      mockery::stub(install_granite, "processx::run", function(cmd, args, ...) {
        list(status = 0, stdout = "uv 0.1.0")
      })
      
      mockery::stub(install_granite, "system.file", function(...) temp_dir)
      mockery::stub(install_granite, "dir.exists", function(...) TRUE)
      mockery::stub(install_granite, "file.exists", function(...) FALSE)
      
      result <- suppressMessages(install_granite())
      expect_true(result)
    }
  )
})

test_that("install_granite uses requirements.txt when available", {
  skip_on_cran()
  skip_if_not_installed("processx")
  
  temp_dir <- tempdir()
  
  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    {
      mockery::stub(install_granite, "processx::run", function(cmd, args, ...) {
        list(status = 0, stdout = "uv 0.1.0")
      })
      
      mockery::stub(install_granite, "system.file", function(...) temp_dir)
      mockery::stub(install_granite, "dir.exists", function(...) TRUE)
      mockery::stub(install_granite, "file.exists", function(...) TRUE)
      
      result <- suppressMessages(install_granite())
      expect_true(result)
    }
  )
})

test_that(".onLoad initializes Python modules", {
  skip_on_cran()
  
  # Test that .onLoad doesn't error
  expect_no_error({
    .onLoad(NULL, "graniteR")
  })
})
