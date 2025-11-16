# Contributing to graniteR

Thanks for considering contributing to graniteR!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a new branch for your feature
4. Make your changes
5. Submit a pull request

## Development Setup

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed setup instructions.

## Code Standards
                                                                                                                                                                                                                                                
- Follow the tidyverse style guide
- Use native R pipe `|>` instead of magrittr pipe
- Write tests for new features
- Add roxygen2 documentation for all exported functions
- Keep functions focused and composable

## Pull Request Process

1. Update tests for your changes
2. Update documentation
3. Run `devtools::check()` and ensure it passes
4. Update NEWS.md if adding features or fixing bugs
5. Submit PR with clear description of changes

## Reporting Issues

When reporting issues, please include:

- Your R version
- Your Python version
- Package versions (graniteR, reticulate, transformers, torch)
- A minimal reproducible example
- Expected vs actual behavior

## Questions

For questions about usage, please open a GitHub issue with the "question" label.
