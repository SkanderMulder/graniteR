# R Package Quality Audit: graniteR

**Date:** 2025-11-16
**Package Version:** 0.1.0
**Audited Against:** [R Packages (2e)](https://r-pkgs.org/)

This document summarizes the quality audit of the `graniteR` package against current best practices.

---

## Overall Assessment

| Category | Score | Status | Notes |
|--------------------|-------|--------|--------------------------------|
| **Overall Score** | **85%** | ✅ **GOOD** | Strong foundation, needs refinement. |
| Package Structure | 95% | ✅ Excellent | Clean, conventional layout. |
| DESCRIPTION File | 85% | ✅ Good | Missing `SystemRequirements`. |
| Code Quality | 90% | ✅ Excellent | Follows modern standards. |
| Documentation | 75% | ⚠️ Good | Missing package-level docs. |
| **Testing** | **45%** | ❌ **Needs Improvement** | **Coverage is low (~30-45%).** |
| Vignettes | 95% | ✅ Excellent | Comprehensive and clear. |
| Data Management | 100% | ✅ Excellent | Perfect implementation. |
| Python Integration | 100% | ✅ Excellent | Best-in-class `reticulate` usage. |
| CI/CD | 50% | ⚠️ Needs Improvement | Actions exist but lack coverage reporting. |

---

## High-Priority Recommendations

1.  **Increase Test Coverage (to >60%)**
    -   **Problem:** The `classifier.R` module (288 lines) is completely untested.
    -   **Action:** Add a `test-classifier.R` file to test `granite_classifier()`, `granite_train()`, and `granite_predict()`.
    -   **Command:** `usethis::use_test("classifier")`

2.  **Add Package-Level Documentation**
    -   **Problem:** The package lacks a central help page (`?graniteR`).
    -   **Action:** Create a `graniteR-package.R` file with a roxygen2 block describing the package's purpose and key functions.
    -   **Command:** `usethis::use_package_doc()`

3.  **Add SystemRequirements to DESCRIPTION**
    -   **Problem:** The dependency on Python and its libraries is not declared.
    -   **Action:** Add a `SystemRequirements` field to the `DESCRIPTION` file.
    -   **Content:** `SystemRequirements: Python (>= 3.8), transformers, torch, datasets, numpy`

4.  **Set Up Test Coverage Reporting**
    -   **Problem:** CI runs tests but does not report on coverage.
    -   **Action:** Add a GitHub Action workflow for test coverage.
    -   **Command:** `usethis::use_github_action("test-coverage")`

5.  **Remove Extraneous Files**
    -   **Problem:** A duplicate logo `logo1.png` exists in the project root.
    -   **Action:** Delete the file. It should only reside in `man/figures/`.
    -   **Command:** `rm logo1.png`

---

## Detailed Audit Findings

### Strengths

-   **Python Integration:** Excellent use of `reticulate`, `uv`, and helper functions (`install_pyenv()`) for dependency management.
-   **Documentation:** High-quality vignettes, `README`, and supplementary guides (`.github/SETUP.md`).
-   **Data Management:** Follows best practices perfectly using `data-raw/`, `usethis::use_data()`, and providing clear data documentation.
-   **Code & Package Structure:** Clean, well-organized, and adheres to standard conventions.

### Areas for Improvement

| Area | Issue | Recommendation | Priority |
|---------------|------------------------------------------------|--------------------------------------------------------------------------------|----------|
| **Testing** | **No tests for the classifier module.** | Add `test-classifier.R` and test all exported functions. | **High** |
| | Low test specificity (uses `expect_true` too often). | Use `expect_equal()`, `expect_s3_class()`, and `expect_named()` for precision. | Medium |
| **Documentation** | **Missing package-level documentation.** | Use `usethis::use_package_doc()` to create a package man page. | **High** |
| | Examples use `\dontrun{}`. | Use `@examplesIf requireNamespace("transformers")` for conditional examples. | Medium |
| | Limited cross-references. | Add `@seealso` links to related functions and vignettes. | Medium |
| **DESCRIPTION** | **`SystemRequirements` field is missing.** | Declare Python and library dependencies. | **High** |
| | No minimum versions for tidyverse packages. | Specify versions (e.g., `dplyr (>= 1.1.0)`) if using recent features. | Medium |
| **CI/CD** | **No test coverage reporting.** | Add a Codecov or Coveralls workflow. | **High** |
| **Code Quality** | `.onLoad()` does not handle errors gracefully. | Wrap `py_require()` in `tryCatch` and issue a `packageStartupMessage()` on failure. | Medium |

---

**Audit Completed By:** Automated analysis based on r-pkgs.org best practices.  
**Last Updated:** 2025-11-16