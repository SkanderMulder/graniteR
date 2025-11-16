# Development Guide

## Setup

1. Clone the repository:
```bash
git clone https://github.com/skandermulder/graniteR.git
cd graniteR
```

2. Install R package dependencies:
```r
install.packages(c("devtools", "roxygen2", "testthat", "pkgdown"))
devtools::install_deps()
```

3. Install Python dependencies:
```r
library(graniteR)
install_pyenv()
```

## Development Workflow

### Loading the package

```r
devtools::load_all()
```

### Running tests

```r
devtools::test()
```

### Building documentation

```r
devtools::document()
```

### Checking the package

```r
devtools::check()
```

### Building the website

```r
pkgdown::build_site()
```

## Code Style

- Use native R pipe `|>` (requires R >= 4.1.0)
- Follow tidyverse style guide
- All functions should be pipe-friendly
- Use tidy evaluation with `rlang::enquo()` for column arguments
- Keep functions focused and composable

## Adding New Features

1. Write the function in the appropriate R file
2. Add roxygen2 documentation
3. Add tests in `tests/testthat/`
4. Update `NEWS.md`
5. Run `devtools::check()` to ensure everything works
6. Update the vignette if needed

## File Organization

- `R/` - R source code
  - `zzz.R` - Package initialization
  - `utils.R` - Helper functions
  - `model.R` - Model creation functions
  - `embed.R` - Embedding functions
  - `classifier.R` - Classification functions
- `inst/python/` - Python helper scripts
- `tests/testthat/` - Unit tests
- `vignettes/` - Long-form documentation
- `man/` - Auto-generated documentation (don't edit directly)

## Testing

Tests should:
- Skip if Python dependencies not available
- Use `skip_if_not(reticulate::py_module_available("transformers"))`
- Test both success and failure cases
- Use small, fast examples

## Documentation

- All exported functions must have roxygen2 documentation
- Include examples in `@examples` (wrapped in `\dontrun{}` if they require Python)
- Update vignettes for major features
- Keep README.md concise and focused on quick start

## Release Process

1. Update version in DESCRIPTION
2. Update NEWS.md with changes
3. Run `devtools::check()`
4. Build and check on multiple R versions
5. Create GitHub release
6. Build pkgdown site: `pkgdown::build_site()`
