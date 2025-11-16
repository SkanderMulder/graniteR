test_that("install_pyenv checks for processx package", {
  skip_on_cran()
  
  # Mock environment without processx
  with_mock(
    requireNamespace = function(package, quietly = TRUE) FALSE,
    .env = "base", # Specify the environment of the function to mock
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
  
  with_mock(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    .env = "base", # Specify the environment of the function to mock
    {
      result <- suppressMessages(install_pyenv())
      expect_false(result)
    }
  )
})

test_that("install_pyenv creates venv path", {
  skip_on_cran()
  skip_if_not_installed("processx")
  
  temp_dir <- tempdir()
  venv_path <- file.path(temp_dir, "test_venv")
  
  with_mock(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    .env = "base", # Specify the environment of the function to mock
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
  
  temp_dir <- tempdir()
  
  
  with_mock(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    .env = "base", # Specify the environment of the function to mock
    {
      mockery::stub(install_pyenv, "system.file", function(...) temp_dir)
      mockery::stub(install_pyenv, "dir.exists", function(...) TRUE)
            result <- suppressMessages(install_pyenv())      expect_true(result)
    }
  )
})

test_that("install_pyenv uses requirements.txt when available", {
  skip_on_cran()
  skip_if_not_installed("processx")
  
  temp_dir <- tempdir()
  
  with_mock(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    .env = "base", # Specify the environment of the function to mock
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

test_that(".onAttach initializes Python modules", {
  skip_on_cran()
  
  # Test that .onAttach doesn't error
  expect_no_error({
    .onAttach(NULL, "graniteR")
  })
})