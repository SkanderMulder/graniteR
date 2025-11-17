# Instructions for Adhering to R Packages Guidelines

Below is a comprehensive set of instructions derived from each chapter of *R Packages (2nd Edition)* by Hadley Wickham and Jennifer Bryan. These are formatted as actionable steps and rules for an AI agent (or developer) to follow when building, maintaining, or extending R packages. Each chapter's instructions focus on the key best practices, workflows, and guidelines outlined in the book.

## Chapter 1: The Whole Game
- Load the devtools package to initiate package development.
- Use `create_package()` to initialize a new package in a dedicated directory, ensuring it is not nested inside another project, package, or library.
- Call `library(devtools)` after package creation if needed, as it may restart the R session.
- Initialize Git version control with `use_git()` to track changes, and commit files after each significant step.
- Write functions in `.R` files within the `R/` directory, naming files after the functions initially.
- Use `use_r()` to create and open new `.R` files for function definitions.
- Employ `load_all()` to load the package for interactive testing without building or installing it.
- Run `check()` frequently to validate the package and address issues early.
- Edit the `DESCRIPTION` file to set the package title, description, and author information.
- Set a valid license using `use_mit_license()` or similar functions.
- Document functions with roxygen2 comments above definitions, including parameters, return values, exports, and examples.
- Execute `document()` to generate documentation and update the `NAMESPACE` file.
- Install the package with `install()` after ensuring it passes checks.
- Add unit tests using `use_testthat()` to set up the testing framework.
- Create test files with `use_test()` and write tests in the `tests/testthat/` directory.
- Declare dependencies with `use_package()` and reference functions from other packages using `package::function()`.
- Use `use_github()` to connect the package to a GitHub repository for version control and collaboration.
- Create an executable README with `use_readme_rmd()` to document installation and usage.
- Render the README with `build_readme()` to keep it in sync with the package.
- Commit changes after each development step, especially after adding code, tests, or documentation.
- Run `test()` to execute all unit tests and ensure code correctness.
- Use RStudio IDE features like the Build pane for `load_all()`, `check()`, `document()`, and `test()`.

## Chapter 2: System Setup
- Install the latest version of R (at least 4.5.2).
- Install required packages for package development by running `install.packages(c("devtools", "roxygen2", "testthat", "knitr"))`.
- Use a recent version of RStudio Desktop, available at https://posit.co/download/rstudio-desktop/.
- Attach devtools in your R session by running `library(devtools)` when developing packages interactively.
- When writing package code, do not depend on devtools; instead, access functions from their primary package (e.g., use `sessioninfo::session_info()` instead of `devtools::session_info()`).
- Avoid using devtools in qualified calls like `pkg::fcn()` unless devtools is the primary home of the function.
- Report bugs in the package that is the primary home of a function.
- Use usethis functions directly without qualification when attached via devtools (e.g., `use_testthat()`), or qualify them programmatically (e.g., `usethis::use_testthat()`).
- Attach devtools in your `.Rprofile` startup file to avoid repeated attachment in every session, but only in interactive sessions.
- Use `use_devtools()` to create or edit `.Rprofile` and set up devtools attachment.
- Set personal defaults for package development using `usethis` options in `.Rprofile`, such as author information and license preferences.
- Install development versions of devtools and usethis using `devtools::install_github("r-lib/devtools")` and `devtools::install_github("r-lib/usethis")`, or via `pak::pak()`.
- On Windows, install Rtools from https://cran.r-project.org/bin/windows/Rtools/ but do not select “Edit the system PATH” during installation; select “Save version information to registry.”
- On macOS, install Xcode command line tools by running `xcode-select --install` or install full Xcode from the Mac App Store.
- On Linux (e.g., Ubuntu/Debian), install R development tools with `sudo apt install r-base-dev`; on Fedora/RedHat, install R with `sudo dnf install R` to include development tools.
- Verify system preparation by running `devtools::dev_sitrep()` and update any missing or out-of-date tools or packages as indicated.

## Chapter 3: Package Structure and State
- Understand the five states an R package can be in: source, bundled, binary, installed, and in-memory.
- Work with a package in its source form during development, which consists of a directory with specific components like a DESCRIPTION file and an R/ directory containing .R files.
- Use `devtools::build()` to create a bundled package (source tarball) from a source package.
- Use `devtools::build(binary = TRUE)` to create a binary package on the appropriate operating system.
- Use `install.packages()`, `devtools::install_github()`, or `devtools::install()` to move a package from source, bundled, or binary states into the installed state.
- Use `library()` to load an installed package into memory for use.
- Use `devtools::load_all()` to load a source package directly into memory during development.
- Create a source package by setting up a directory structure with required components such as DESCRIPTION and R/.
- Explore package source code by browsing repositories on public hosting services like GitHub or via CRAN mirrors.
- Use `.Rbuildignore` to control which files from the source package are included in the bundled form; each line is a Perl-compatible regular expression.
- Use `usethis::use_build_ignore()` to add entries to `.Rbuildignore`, ensuring proper anchoring and escaping of regular expressions.
- Exclude development aids and generated files from distribution by listing them in `.Rbuildignore`, such as Rproj files, README.Rmd, data-raw directories, and GitHub workflows.
- Recognize that a bundled package is a compressed .tar.gz file that serves as an intermediary between source and installed states.
- Note that CRAN packages are available in bundled form and can be unpacked to reveal a structure similar to the source package, minus ignored files and with built vignettes.
- Understand that binary packages are platform-specific single files (.tgz for macOS, .zip for Windows) and are typically distributed by CRAN.
- Use `devtools::build(binary = TRUE)` to create binary packages, which involve compiling code and converting files into efficient formats.
- Recognize that an installed package is a decompressed binary package within a package library directory.
- Use `R CMD INSTALL` as the underlying tool for installation, accessible via `devtools::install()`.
- Install packages in a clean R session to avoid issues, especially on Windows with compiled code.
- Manage package libraries using `.libPaths()` to view active libraries, which are directories containing installed packages.
- Set up a user library by creating the directory at `R_LIBS_USER` path using `dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)` before installing add-on packages.
- Control library search paths via environment variables like `R_LIBS_USER`, `.libPaths()`, `withr::with_libpaths()`, or function arguments like `lib.loc`.
- Never use `library()` inside a package; instead, use appropriate dependency management mechanisms.

## Chapter 4: Fundamental Development Workflows
- Survey existing R packages to assess whether your functionality is already available or if your package adds value in terms of usability, defaults, or edge cases.
- Choose a unique, pronounceable package name that starts with a letter, contains only letters, numbers, and periods, and does not end with a period.
- Avoid hyphens, underscores, and mixed case in package names; prefer names that are easy to Google and do not conflict with CRAN, Bioconductor, or other ecosystems.
- Use tools like `available::available()` or `pak::pkg_name_check()` to validate name availability on CRAN, Bioconductor, GitHub, and for unintended meanings.
- Create a new package using `create_package(path)` instead of `package.skeleton()`, ensuring it includes `R/`, `DESCRIPTION`, `NAMESPACE`, and basic ignore files.
- Store source packages in a dedicated directory separate from installed packages, such as within a home directory like `~/r/packages/`.
- Use RStudio Projects for each package to enable easy launching, isolated sessions, code navigation, and development shortcuts.
- Convert an existing directory to an RStudio Project via File > New Project > Existing Directory or `usethis::use_rstudio()`.
- Keep the top-level package directory as the working directory to avoid reliance on implicit paths.
- Use explicit path helpers like `testthat::test_path()` or `fs::path_package()` for resilient file references across package states.
- Load package code into the session with `devtools::load_all()` after editing to test changes interactively under a namespace.
- Run `devtools::check()` frequently—ideally multiple times per day—during development to catch and fix issues early.
- Ensure `check()` updates documentation via `document()`, bundles the package, and sets `NOT_CRAN` appropriately.
- Fix ERRORs immediately as they indicate severe issues; address WARNINGs if submitting to CRAN; aim to eliminate NOTEs or justify them.
- Launch RStudio Projects by double-clicking the `.Rproj` file or via RStudio’s File > Open Project menu.
- Maintain consistency between the RStudio Project, usethis active project, and working directory using `proj_sitrep()` if issues arise.

## Chapter 5: The Package Within
- Refactor an existing data analysis script to isolate reusable data and logic into functions and objects, such as lookup tables and helper functions.
- Use add-on packages like the tidyverse for data wrangling, but avoid declaring dependencies via `library()`; instead, use namespace-qualified calls (e.g., `dplyr::mutate()`).
- Move reusable code and data into separate files, such as `cleaning-helpers.R`, to improve script readability and maintainability.
- When creating a package, use `usethis::create_package()` to scaffold the package structure.
- Place function definitions in `R/` directory files (e.g., `R/cleaning-helpers.R`).
- Export user-facing functions using `@export` in roxygen comments and run `devtools::document()` to update the `NAMESPACE`.
- Do not rely on `library()` to declare dependencies; instead, list required packages in the `Imports` field of `DESCRIPTION` (e.g., `Imports: dplyr`).
- Avoid reading data files (like CSVs) from external paths within package functions; define static data directly in R code (e.g., using `dplyr::tribble()`).
- Ensure functions use run-time evaluation for dynamic values (e.g., `Sys.time()` inside the function) rather than build-time assignments (e.g., `now <- Sys.time()` at top level).
- Avoid side effects that alter global state (e.g., `Sys.setlocale()` or `Sys.setenv()`) without scoping them narrowly; use tools like `withr::local_locale()` or `withr::local_timezone()` to limit changes.
- Document all exported functions with roxygen comments to avoid missing documentation warnings during `R CMD check`.
- Handle non-standard evaluation issues (e.g., with dplyr) by suppressing undefined global variables using `utils::globalVariables()` or assigning `NULL` to them.
- Run `devtools::check()` frequently during development to catch issues like missing dependencies, undefined functions, or data file problems.
- Ensure code runs at the correct time: top-level assignments (except dynamic values) occur at build time, while function logic should execute at run time.
- Prefer specific package imports over meta-packages like `tidyverse` in package dependencies.

## Chapter 6: R Code
- Store all R function definitions in `.R` files within the `R/` directory.
- Organize functions into files using meaningful names that indicate their content.
- Avoid placing all functions in a single file or each function in its own file unless justified (e.g., large functions or extensive documentation).
- Group related functions together, such as a main function with its helpers or a family of related functions.
- Use `R/utils.R` to store small utility functions used across multiple package functions.
- Use `devtools::load_all()` to load and test package code during development instead of `source()`.
- Follow the tidyverse style guide for code style.
- Enforce consistent code style using the styler package (e.g., `styler::style_pkg()`, `styler::style_dir()`, `styler::style_file()`).
- Ensure all code in `R/` is executed only at package build time; avoid top-level code that creates objects outside functions.
- Review any top-level code in `R/` files carefully, as it runs during package build and may produce environment-specific results.
- Call functions like `system.file()` inside functions at run time, not at build time.
- Avoid defining aliases via direct assignment (e.g., `foo <- pkgB::blah`) to prevent locking in a specific package version; instead, use `foo <- function(...) pkgB::blah(...)`.
- Do not use `library()` or `require()` inside package code; declare dependencies in the `DESCRIPTION` file.
- Avoid using `source()` within package code; use `load_all()` for development-time loading.
- Use `withr` package or `base::on.exit()` to manage and restore changes to global state (e.g., options, working directory, environment variables).
- Isolate side effects like plotting or printing in dedicated output-producing functions.
- Use `.onLoad()` for setup tasks when the package is loaded, and `.onAttach()` for displaying startup messages using `packageStartupMessage()`.
- Save `.onLoad()`, `.onAttach()`, and `.onUnload()` in `R/zzz.R`.
- Run `document()`, `load_all()`, `test()`, and `check()` frequently during development to catch issues early.
- Ensure all `.R` files use only ASCII characters, escaping non-ASCII characters with Unicode escapes (e.g., `\u1234`).
- Use tools like `tools::showNonASCII()` to detect and fix non-ASCII characters in code.

## Chapter 7: Data
- Store exported data in `data/` as `.rda` files, one object per file, named identically to the file.
- Use `usethis::use_data()` to create and save exported datasets to `data/`.
- Include `LazyData: true` in `DESCRIPTION` to enable lazy-loading of exported data.
- Avoid using `utils::data()` to load datasets; access them directly via `pkg::dataset` or after `library(pkg)`.
- Place code to generate exported data in `data-raw/` and use `usethis::use_data_raw()` to set up this directory.
- Document exported datasets in `R/` using roxygen2, including `@format` to describe structure and `@source` for origin.
- Never use `@export` for datasets; they are exported by default when in `data/`.
- Use UTF-8 encoding for strings in data; ensure proper encoding in `data-raw/` scripts using `enc2utf8()` or `iconv()`.
- For internal data used by package functions, store in `R/sysdata.rda` using `usethis::use_data(internal = TRUE)`.
- Keep internal data generation code in `data-raw/` and regenerate as needed.
- Do not document internal data objects in roxygen; they are not exported.
- Store raw, non-R data files in `inst/extdata/` for user access and examples.
- Use `system.file("extdata", package = "pkg")` to retrieve file paths to `extdata` contents.
- Provide helper functions like `pkg_example()` to simplify access to example files in `extdata`.
- Use environments to manage dynamic, session-specific internal state, initialized with `new.env(parent = emptyenv())`.
- Define internal environments at the top level in `R/`, preferably in `aaa.R`, before use.
- Store persistent user data using `tools::R_user_dir()` to comply with CRAN policies and XDG standards.
- Avoid writing persistent data to user home directories; use designated R user directories for config, cache, or data.
- For sensitive data, prefer integrating with established tools like keyring instead of custom storage.
- Keep exported datasets under 1MB for CRAN submission; consider separate data packages for larger datasets.
- Experiment with compression (`compress` argument in `use_data`) to optimize `.rda` file size.

## Chapter 8: Other Components
- Include a `DESCRIPTION` file as a required component of every R package.
- Add tests and documentation as highly recommended components.
- Use the `src/` directory for source and header files for compiled code (e.g., C/C++), if needed for performance or external libraries.
- Place arbitrary additional files in the `inst/` directory, such as `CITATION`, R Markdown templates, or RStudio add-ins.
- Avoid creating subdirectories in `inst/` that match official R package directories (e.g., `inst/data`, `inst/help`, `inst/html`, `inst/libs`, `inst/man`, `inst/Meta`, `inst/R`, `inst/src`, `inst/tests`, `inst/tools`, `inst/vignettes`) to prevent malformed packages.
- Use `inst/extdata` for additional external data needed for examples and vignettes.
- Access files located in `inst/` using `system.file("filename", package = "yourpackage")` in code or documentation.
- Include a `CITATION` file in `inst/` to specify how to cite the package; generate it with `usethis::use_citation()` and format it using `bibentry` with fields like `title`, `author`, `year`, `journal`, etc.
- Use the `citation()` function to display citation information for the package.
- Consider repurposing or removing content from the `demo/` directory, as demos are legacy and prone to becoming outdated; prefer vignettes or `README.Rmd` instead for active maintenance.
- Place executable scripts in `inst/` rather than `exec/`, as `inst/` is the preferred location for such files.
- Use the `tools/` directory for auxiliary files needed during configuration (e.g., with a `configure` script) or for custom maintenance scripts that are not shipped but aid development (e.g., for updating embedded resources).
- List `tools/` in `.Rbuildignore` if the scripts are for development purposes only and not included in the package bundle.
- Store instructions for creating package data in `data-raw/` and record construction methods for test fixtures to support maintainability.

## Chapter 9: DESCRIPTION
- Create a `DESCRIPTION` file for every package, as it defines the package directory.
- Use `usethis::create_package()` to automatically generate a minimal `DESCRIPTION` file.
- Customize default `DESCRIPTION` content via global option `usethis.description` if creating multiple packages.
- Format `DESCRIPTION` using DCF: field names followed by colon, values on same line or indented with 4 spaces for multi-line values.
- Set `Title` as a one-line, plain text description in title case, not ending in a period, under 65 characters.
- Write `Description` as one paragraph, up to 80 characters per line, with subsequent lines indented by 4 spaces.
- Enclose names of R packages, software, and APIs in single quotes within `Title` and `Description`.
- Expand acronyms in `Description`, avoiding them in `Title`.
- Avoid including the package name in `Title` or starting `Title`/`Description` with phrases like “A package for …” or “This package does …”.
- Use `Authors@R` field with `person()` to specify authors and maintainers, including roles (`aut`, `cre`, `ctb`, `cph`, `fnd`) and email (required for `cre`).
- Ensure at least one `aut` and one `cre` (maintainer) are listed, with `cre` having a valid email address.
- Add ORCID via `comment` argument in `person()` when applicable.
- List multiple authors using `c()` with `person()`.
- Prefer `Authors@R` over deprecated `Author` and `Maintainer` fields.
- Set `URL` for package website and source repository, separated by commas.
- Set `BugReports` to a URL like GitHub issues for bug submissions.
- Use `usethis::use_github()` or `use_github_links()` to auto-populate `URL` and `BugReports`.
- Specify `License` using standard forms recognized by R (e.g., MIT, GPL3).
- List runtime dependencies in `Imports` (e.g., `dplyr`, `tidyr`), one per line, alphabetically ordered.
- List optional or development dependencies in `Suggests` (e.g., `ggplot2`, `testthat`), one per line, alphabetically ordered.
- Use `usethis::use_package()` to add packages to `Imports` or `Suggests`, which also reminds about namespace usage.
- Specify minimum versions in dependencies (e.g., `dplyr (>= 1.0.0)`), not exact versions, to avoid conflicts.
- Use `usethis::use_package(..., min_version = "...")` to set minimum versions.
- Set minimum R version in `Depends` only when necessary (e.g., `Depends: R (>= 4.0.0)`), with testing.
- Use `LinkingTo` for packages providing C/C++ code your package uses.
- Avoid `Enhances` unless providing methods for classes in another package.
- Set `Version` to reflect package lifecycle stage.
- Add `LazyData: true` if package includes data, via `usethis::use_data()`.
- Set `Encoding: UTF-8` as default character encoding.
- Use `Collate` to control sourcing order of R files, typically managed by roxygen2.
- List vignette-building packages in `VignetteBuilder` (e.g., `knitr`).
- Describe non-R dependencies in `SystemRequirements` (e.g., `C++17`, `GNU make`).
- Avoid manually managing `Date`; let it be auto-populated during bundling.
- Prefix custom fields with `Config/` if submitting to CRAN.
- Include `Roxygen` and `RoxygenNote` fields when using roxygen2 for documentation.

## Chapter 10: Dependencies: Mindset and Background
- Evaluate dependencies holistically by considering their type, recursive dependencies, whether they are already fulfilled, installation burden (time, size, system requirements), maintenance capacity, and functionality provided.
- Prefer a balanced approach that weighs benefits (new features, bug fixes, consistency) against costs (installation time, disk space, maintenance burden from changes).
- Target the primary audience when deciding dependency scope: leaner packages for package authors, feature-rich for end users like data scientists who likely have popular packages installed.
- Use tools like `itdepends` or `pak` to quantitatively analyze dependency weight, such as `pak::pkg_deps_tree()` and `pak::pkg_deps_explain()`.
- Avoid minimizing dependency count absolutely; instead, assess whether a dependency provides critical, well-tested functionality worth the cost.
- For tidyverse-related packages, depend on specific low-level packages (e.g., rlang, cli, glue, withr, lifecycle) rather than meta-packages like tidyverse or devtools.
- Use `usethis::use_tidy_dependencies()` to add recommended tidyverse low-level dependencies.
- Ensure compliance with CRAN policies: aim for fewer than 20 non-default packages in `Imports` to avoid NOTES during `R CMD check`.
- List packages in `Imports` if they are required for the package to function; use `Suggests` for optional use cases like tests, examples, or vignettes.
- Use `Imports` for packages needed in exported functions; use `Suggests` for packages only used in tests, documentation, or non-critical paths.
- Generate the `NAMESPACE` file automatically using roxygen2 with `@export` and import-related tags in roxygen comments, rather than editing it manually.
- Use `@importFrom()` or `@import()` in roxygen comments to specify which objects from dependencies are imported into the namespace.
- Prefer `import()` over `importFrom()` when all objects from a package are needed, as roxygen2 will generate the appropriate directive.
- Use `package::function()` calling style in code under `R/` to ensure unambiguous references and avoid search path conflicts.
- Avoid using `library()` or `require()` in package code under `R/` or `tests/`; instead, use `requireNamespace()` for suggested packages when checking availability.
- Use `library()` only in user-facing contexts like scripts or vignettes where attaching packages is desired.
- Always list dependencies in `Imports` rather than `Depends` unless there is a specific reason, to keep packages self-contained and minimize search path modifications.
- Ensure that dependencies listed in `Imports` are loaded automatically when used, while `Depends` causes attachment and potential namespace pollution.

## Chapter 11: Dependencies: In Practice
- List dependencies in the DESCRIPTION file (e.g., Imports, Depends, Suggests) to ensure they are installed when the package is installed.
- Understand that listing a package in Imports does not automatically make its functions available; it only ensures installation.
- Use the package::function() syntax by default in code under R/ to call external functions, avoiding unnecessary imports.
- Import specific functions into the namespace only when necessary (e.g., for operators, frequently used functions, or performance in tight loops) using @importFrom in roxygen comments.
- Use @import to bring an entire package namespace into your package’s namespace only in rare cases, such as when a package functions like a base package (e.g., rlang in tidyverse).
- Place @importFrom tags either near the function usage or in a central location (e.g., in R/pkg-package.R) managed by usethis::use_import_from().
- Run devtools::document() (or Ctrl/Cmd + Shift + D in RStudio) to regenerate the NAMESPACE file after adding roxygen import/export tags.
- Do not use library() to attach dependencies in test code; use package::function() or directly call imported functions.
- In examples and vignettes, use library() or package::function() to access packages listed in Imports; for Suggests, guard usage with if (requireNamespace("pkg", quietly = TRUE)).
- For packages in Suggests, check availability with requireNamespace() or rlang::check_installed() before use in R/ code; provide fallbacks if optional.
- Assume suggested packages are available in tests and vignettes unless conditionally skipped using testthat::skip_if_not_installed().
- For packages in Depends, functions are available upon loading; use same calling conventions as Imports, but importing the full namespace is more common.
- No need to explicitly attach Depends packages in examples or vignettes; they are loaded automatically with your package.
- Export only functions intended for external use using @export in roxygen comments; avoid over-exporting to prevent breaking reverse dependencies.
- Use @keywords internal along with @export for functions useful to developers but not end users.
- Re-export functions from dependencies by listing the dependency in Imports, then using @importFrom and @export together in roxygen comments above a reference to the function.
- To suppress NOTES about unused Imports, include a namespace-qualified reference (e.g., pkg::fun) in an unexported function under R/ without calling it.
- Use usethis::use_import_from() and usethis::use_package() to automate adding dependencies and import tags to DESCRIPTION and source files.
- For development dependencies (e.g., dev version of another package), use Remotes field temporarily but remove it before CRAN submission.
- Use Config/Needs/* fields (e.g., Config/Needs/website) to declare non-runtime dependencies needed for websites or CI, without affecting formal dependency list.

## Chapter 12: Licensing
- **Choose a license for your own code using the appropriate function:**
  - Use `use_mit_license()` for a permissive license (MIT).
  - Use `use_gpl_license()` for a copyleft license (GPLv3, set to GPL >= 2 or >= 3 by default).
  - Use `use_cc0_license()` for data with minimal restrictions (CC0).
  - Use `use_ccby_license()` for data requiring attribution (CC BY).
  - Use `use_proprietary_license()` if the package is not open source (cannot be distributed via CRAN).

- **Set the `License` field in the `DESCRIPTION` file** to reflect the chosen license in standard form (e.g., `GPL (>= 3)`, `MIT`, or `file LICENSE`).

- **Include a `LICENSE` file**:
  - For template licenses (e.g., MIT), include required details like year and copyright holder.
  - For non-standard licenses, include the full text.
  - Do not include full text of standard open-source licenses.

- **Use `LICENSE.md` to include the full text of the license**, but ensure it is excluded from CRAN submission via `.Rbuildignore`.

- **For data-focused packages**, prefer Creative Commons licenses (CC0 or CC BY) over code licenses.

- **When relicensing an existing package**:
  - Check the `Authors@R` field in `DESCRIPTION` for bundled code.
  - Identify all contributors via Git history or GitHub.
  - Optionally exclude minor contributors (e.g., typo fixes), but err on including them.
  - Obtain explicit permission from all copyright holders (e.g., via GitHub issue).
  - Apply the new license using the appropriate function only after approval.

- **For contributed code (e.g., pull requests)**:
  - Assume the contributor agrees to your package’s license per GitHub terms.
  - Retain the contributor’s copyright; do not change their code’s license without permission.
  - Consider requiring a Contributor License Agreement (CLA) if you need flexibility to relicense.
  - Acknowledge contributions in `NEWS.md` and release announcements; optionally credit all contributors.

- **Before bundling external code**:
  - Verify license compatibility with your package’s license:
    - Compatible if both licenses are identical.
    - MIT or BSD licensed code can be bundled into any open-source package.
    - Copyleft code (e.g., GPL) cannot be bundled into permissive-licensed packages.
    - Stack Overflow code is under CC BY-SA, compatible only with GPLv3.
    - Check compatibility for other licenses using resources like Wikipedia’s license compatibility diagram.
  - For non-open-source packages, consult legal department, especially for copyleft code.

- **When including bundled code**:
  - Preserve all original copyright and license statements.
  - Place each code fragment in its own file with license/copyright at the top.
  - For multiple files, organize in a directory with a `LICENSE` file inside.
  - Add copyright holders to `Authors@R` using `role = "cph"` and describe their contribution in `comment`.

- **For CRAN submission with mixed licenses**:
  - Include a `LICENSE.note` file explaining:
    - The overall package license.
    - The specific licenses of bundled components.
  - List all copyright holders in `DESCRIPTION`.

- **When using other R packages via `Imports` or `Suggests`**:
  - No need to make your package’s license compatible with those packages, as R itself is GPL but does not impose GPL on dependent packages when only calling R functions.
  - Exercise caution with compiled code or `LinkingTo`; ensure license compatibility in those cases.

- **Avoid copying small amounts of code from other packages unless necessary**; prefer dependencies to ensure bug fixes and avoid licensing complexity.

- **Do not distribute proprietary packages via CRAN**; use `use_proprietary_license()` only for internal use.

## Chapter 13: Testing Basics
- **Adopt automated testing using testthat 3e** to transition from informal interactive testing to formal unit testing.
- **Run `usethis::use_testthat(3)` once per package** to set up testing infrastructure: create `tests/testthat/` directory, add testthat to `Suggests`, set `Config/testthat/edition: 3`, and generate `tests/testthat.R`.
- **Do not edit `tests/testthat.R`**; it is used during `R CMD check` but not in most test-running scenarios.
- **Organize test files to match source files**: place tests for `R/foofy.R` in `tests/testthat/test-foofy.R`, ensuring test file names start with `test`.
- **Use `usethis::use_test("function_name")`** to create or open corresponding test files; supports variations like `test-blarg.R`, `blarg.R`, or `blarg`.
- **If editing `R/foofy.R` in RStudio, call `use_test()` with no arguments** to automatically open or create its test file.
- **Structure tests hierarchically**: use `test_that("description", { ... })` to group expectations; each test should cover a single unit of functionality.
- **Write clear test descriptions** that read naturally (e.g., `"basic duplication works"`) to aid in identifying failures.
- **Use expectations like `expect_equal()`, `expect_identical()`, `expect_error()`** to assert behavior; prefer `expect_identical()` for exact matches and `expect_error(..., class = "...")` for condition classes over message matching.
- **Test edge cases and adversarial inputs** to prevent bugs; include checks for errors, warnings, and recycling rules.
- **Run tests at different scales**:
  - Use `devtools::load_all()` and execute individual expectations interactively during micro-iteration.
  - Run a single test file with `testthat::test_file("path/to/test-file.R")` or `devtools::test_active_file()` during mezzo-iteration.
  - Run the full test suite with `devtools::test()` and verify with `devtools::check()` during macro-iteration.
- **Avoid including `library()` calls in test files**; testthat and dependencies are loaded automatically during testing.
- **Use snapshot tests with `expect_snapshot()`** for complex outputs (e.g., error messages, UI) that are hard to specify inline; snapshots are stored in `tests/testthat/_snaps/` and reviewed non-interactively.
- **Set `error = TRUE` in `expect_snapshot()`** if testing error messages; use `transform` to handle volatile data and `variant` for platform-specific differences.
- **Do not use deprecated `testthat::context()`**; rely on file naming for context.
- **Ensure all tests pass before finalizing changes**; use test failures to guide code or test fixes.

## Chapter 14: Designing Your Test Suite
- Do not include `library()` calls in test files.
- Use testthat edition 3 for testing.
- Test the external interface of functions, not internal implementation details.
- Write one test per distinct behavior to simplify updates.
- Avoid testing simple, reliable code; focus on fragile, complex, or uncertain code.
- Always write a test when discovering a bug.
- Adopt a test-first approach: write tests before implementing code.
- Use the covr package to measure test coverage and aim for high but practical coverage.
- Prioritize testing code prone to bugs over achieving 100% coverage.
- Ensure tests are self-sufficient and self-contained, minimizing external dependencies.
- Avoid top-level code in test files; confine logic within `test_that()` blocks.
- Use `withr` package to manage temporary changes to global state (e.g., options, environment variables, packages).
- Clean up side effects like file system changes, search path modifications, and global options after tests.
- Plan for test failures by making tests easy to understand and debug in isolation.
- Use `devtools::load_all()` during interactive development to simulate package loading.
- Avoid `library()` and `source()` calls in tests; access functions via `::` or namespace.
- Define test helper functions in `tests/testthat/helper*.R` files for shared utilities.
- Use setup files (`setup*.R`) for global test configuration and teardown, executed only during automated testing.
- Store test data in subdirectories under `tests/testthat/` and access via `testthat::test_path()`.
- Write files only to the session temporary directory using `withr::local_tempfile()` or `local_tempdir()`.
- Automatically clean up temporary files and directories created during tests.
- Tolerate repetition in test code to improve clarity over strict DRY enforcement.
- Ensure test code is obvious and readable, especially when failing.
- Use `testthat::local_reproducible_output()` (implicit in edition 3) to standardize test conditions.
- Configure GitHub Actions with `usethis::use_github_action("test-coverage")` to monitor coverage.
- Keep test files focused on `test_that()` calls; move auxiliary logic to helpers or setup files.
- Do not edit `tests/testthat.R`; it is run during `R CMD check` but not interactive testing.

## Chapter 15: Advanced Testing Techniques
- Avoid including `library()` calls in test files.
- Declare testthat edition 3 in the DESCRIPTION file for real packages.
- Use test fixtures to arrange a consistent state for testing when tests are not fully self-sufficient.
- Create a helper function (e.g., `new_useful_thing()`) to encapsulate repeated code for constructing test objects.
- Define helper functions either under `R/` or in `tests/testthat/helper.R` to make them available via `devtools::load_all()`.
- Apply memoisation to helper functions if object construction is slow and computationally expensive.
- Use a custom `local_*()`-style function (e.g., `local_useful_thing()`) with `withr::defer()` when object creation has side effects requiring cleanup.
- Store costly test objects as static files (e.g., `.rds`) in `tests/testthat/fixtures/` and load them with `readRDS()` and `test_path()`.
- Include a companion script (e.g., `make-useful-things.R`) in `fixtures/` to re-create static test objects as needed for updates.
- Define hyper-local helpers inside tests only when they simplify repetitive logic and keep each helper short and simple.
- Create custom expectations (e.g., `expect_usethis_error()`, `expect_proj_file()`) to reduce repetition and improve readability in complex test scenarios.
- Use `testthat::skip()` within individual tests rather than at the top of test files to maintain clear connections between skip conditions and specific tests.
- Implement custom skip helpers (e.g., `skip_if_no_api()`) to check for unavailable resources like APIs before running tests.
- Prefer built-in skip functions such as `skip_if()`, `skip_if_not()`, `skip_if_not_installed()`, `skip_if_offline()`, `skip_on_cran()`, and `skip_on_os()` for common conditions.
- Regularly review `R CMD check` results on CI to monitor skipped tests and ensure they align with expectations.
- Avoid mocking external services unless necessary, as it adds complexity; use state-of-the-art tools like `with_mocked_bindings()` if mocking is required.
- Design packages to allow testing without live authenticated access to external services where possible.
- Use `skip_on_cran()` at the start of tests that should not run on CRAN, such as long-running or flaky tests.
- Ensure tests run quickly (ideally under one minute) and use `skip_on_cran()` for slow tests.
- Avoid testing variable conditions like execution time, parallel code behavior, or numerical precision differences across platforms; prefer `expect_equal()` over `expect_identical()` unless identicalness is required.
- Eliminate flaky tests; if unavoidable, wrap them with `skip_on_cran()`.
- Ensure tests do not write to user home filespace or outside the R session temporary directory, and clean up any temporary files or processes launched.
- Avoid accessing the clipboard during tests, as it violates CRAN policies on file system and process hygiene.
- Use `NOT_CRAN="true"` environment variable (set by devtools/testthat) to allow tests to run locally but be skipped on CRAN.
- Consider maintaining integration tests outside the package if they cannot be hosted within a single CRAN package.

## Chapter 16: Function Documentation
- Add roxygen2 comments above each exported function using `#'` to start each line.
- Write the function first, then place the cursor in it and use Code > Insert Roxygen Skeleton to generate a roxygen comment skeleton.
- Include an introduction before any tags, where the first sentence serves as the title in sentence case without a period, followed by a blank line.
- Use tags such as `@param` for arguments, `@returns` for the return value, `@examples` for executable code, and `@export` to document exported functions.
- Document all exported functions and datasets to avoid `R CMD check` warnings about missing documentation.
- Use `@noRd` tag for unexported functions to suppress `.Rd` file generation.
- Run `devtools::document()` to generate or update `.Rd` files from roxygen comments.
- Preview documentation using `?function` after running `devtools::load_all()` to ensure changes are visible during development.
- Use markdown syntax in roxygen comments, such as backticks for inline code and square brackets for auto-linked functions.
- Inherit argument documentation with `@inheritParams function_name` to avoid duplication across related functions.
- Document the return value using `@returns`, describing the shape, type, and dimensions of the output.
- Include examples in `@examples` that are self-contained, run without errors, and avoid side effects like changing options or working directory.
- Use `try()` to show expected errors in examples without stopping execution, or `\dontrun{}` to prevent running code.
- Use `@examplesIf condition()` for conditional example execution, such as when dependencies are available.
- Combine documentation for related functions in one topic using `@rdname function_name` for the second function.
- Reuse sections of documentation with `@inheritSection` or `@inherit` for shared content across functions.
- Document the package itself by including roxygen comments for the `"_PACKAGE"` sentinel in `R/{pkgname}-package.R`.
- Set `Roxygen: list(markdown = TRUE)` in the `DESCRIPTION` file to enable markdown syntax in roxygen comments.
- Reflow roxygen comments using Code > Reflow Comment to maintain consistent line length.
- Ensure examples run quickly and do not exceed 10 minutes for CRAN compliance.
- Avoid documenting unexported functions unless using `@noRd` to prevent public exposure.

## Chapter 17: Vignettes
- Run `usethis::use_vignette("my-vignette")` to create a new vignette, which sets up a `vignettes/` directory, adds knitr to `VignetteBuilder` and rmarkdown to `Suggests` in `DESCRIPTION`, creates a draft `.Rmd` file, and configures `.gitignore` for preview files.
- Write vignettes using R Markdown, integrating prose and code chunks to demonstrate workflows that solve target problems.
- Use `devtools::load_all()` during development of code chunks, but ensure the vignette is rendered against the installed or source package version, not just the installed version.
- Render the vignette periodically using methods such as `devtools::build_rmd("vignettes/my-vignette.Rmd")`, `devtools::install(build_vignettes = TRUE)`, or RStudio’s build tools to reflect current package state.
- Include YAML metadata in the vignette with fields: `title` (matching `\VignetteIndexEntry{}`), `output: rmarkdown::html_vignette`, and `vignette` block specifying engine and encoding; omit `author` and `date` unless necessary.
- Set initial knitr options in a hidden chunk: `knitr::opts_chunk$set(collapse = TRUE, comment = "#>")`.
- Attach the package in a setup chunk with `library(yourpackage)`; avoid replacing it with `load_all()`.
- Ensure any package used in a vignette is listed in `Imports` or `Suggests` in `DESCRIPTION`.
- Use `eval` option in code chunks to conditionally execute code, e.g., `eval = requireNamespace("pkg")` or set globally with `knitr::opts_chunk$set(eval = FALSE)` and override as needed.
- Use `include = FALSE` for chunks that should run but not appear in output, `echo = FALSE` to hide code, and `error = TRUE` to allow errors without stopping execution.
- Store external files needed in vignettes: figures in `vignettes/` and reference with `knitr::include_graphics()`, data in `inst/extdata/` and access via `system.file()`, or place alongside vignette and use relative paths.
- Limit file size for CRAN submission by avoiding excessive graphics; consider using articles instead if size is an issue.
- Aim for one vignette named after the package (e.g., `yourpackage.Rmd`) for simple packages to enable pkgdown “Get started” link; use multiple self-contained but linked vignettes for complex packages.
- Treat vignettes as teaching tools; adopt a beginner’s mindset, explain workflows clearly, and use in-person feedback or blog posts to improve code and documentation.
- Avoid manual modification of `inst/doc/`; let tooling manage built vignettes.
- For development, render vignettes interactively or via `build_rmd()` and treat previews as disposable; do not persist built vignettes in source control.
- For sharing development vignettes, use a pkgdown website or install with `build_vignettes = TRUE`; avoid ad hoc use of `build_vignettes()`.
- Prepare CRAN submissions without manually building vignettes; use `devtools::release()` or `submit_cran()` to handle building.
- Suppress code execution in vignettes for CRAN using `eval = FALSE` where necessary, especially for code requiring credentials, long runs, or prone to failure.
- Consider using `usethis::use_article()` for non-executable, web-only documentation when vignettes cannot be built reliably or are too large.

## Chapter 18: Other Markdown Files
- Create a `README.md` (preferably generated from `README.Rmd`) to answer: Why should I use it? How do I use it? How do I get it?
- Structure the `README` with:
  - A paragraph describing the high-level purpose of the package.
  - An example showing how to solve a simple problem using the package.
  - Installation instructions with copy-pasteable R code.
  - An overview of the package’s main components, and for complex packages, point to vignettes and describe ecosystem fit.
- Use `usethis::use_readme_rmd()` to create a `README.Rmd` template, which will generate `README.md`.
- Ensure `README.md` is included in the package bundle and `README.Rmd` is ignored via `.Rbuildignore`.
- Render `README.Rmd` to `README.md` using `devtools::build_readme()` before release or when updates are made.
- Include R code chunks in `README.Rmd` for examples and plots, and save figures to `man/figures/README-` to ensure they are included in the built package.
- Add badges (e.g., CRAN version, test coverage, CI status) using functions like `usethis::use_cran_badge()`, `usethis::use_coverage()`, or `usethis::use_github_action()`.
- Set up a pre-commit hook (via `usethis::use_readme_rmd()`) to prevent commits if `README.Rmd` is newer than `README.md`.
- Create a `NEWS.md` file to document user-visible changes across releases.
- Use `usethis::use_news_md()` to initialize the `NEWS.md` file.
- Structure `NEWS.md` with:
  - A top-level heading for each version (most recent at top), e.g., `# somepackage (development version)` or `# somepackage 1.0.0`.
  - Bulleted list of changes; use subheadings like `## Major changes` or `## Bug fixes` near release time.
- Include GitHub issue numbers (e.g., `(#10)`) or pull request numbers with author (e.g., `(#101, @hadley)`) for relevant changes.
- Omit usernames in `NEWS.md` if already in `DESCRIPTION`.
- Record user-visible changes in `NEWS.md` whenever they occur, especially for external contributions.
- Before release, compare the source of the release candidate to the previous release using version control to identify missing `NEWS` entries.
- Use the release checklist (from `usethis::use_release_issue()`) to remind to build `README` and polish `NEWS.md`.

## Chapter 19: Website
- Run `usethis::use_pkgdown()` once to initialize pkgdown site setup.
- Ensure the package has a valid structure before building the site.
- Allow `_pkgdown.yml` to be created and opened for inspection; no immediate changes required.
- Add pkgdown-specific patterns (e.g., `^_pkgdown\\.yml$`, `^docs$`, `^pkgdown$`) to `.Rbuildignore`.
- Add `docs` directory to `.gitignore` to treat it as a preview, not source.
- Use `pkgdown::build_site()` to render the website locally; output appears in `docs/`.
- Open `docs/index.html` in a browser to view the rendered site.
- For RStudio users, use Addins > pkgdown > Build pkgdown to build the site.
- Host the package on GitHub and use Git for version control.
- Deploy the site using GitHub Actions to run `pkgdown::build_site()` on each push.
- Use GitHub Pages to serve the site from the `gh-pages` branch.
- Run `usethis::use_pkgdown_github_pages()` to automate GitHub Pages setup, including:
  - Initializing an empty `gh-pages` branch in the GitHub repo.
  - Enabling GitHub Pages to serve from the `gh-pages` branch at `/`.
  - Creating `.github/workflows/pkgdown.yaml` for build-and-deploy workflow.
  - Adding `.github` to `.Rbuildignore` and ignoring HTML files in `.github/.gitignore`.
  - Setting the site URL as the GitHub repo homepage, in `DESCRIPTION`, and in `_pkgdown.yml`.
- Ensure the `URL` field in `DESCRIPTION` includes the pkgdown site URL (and optionally GitHub repo URL).
- Set `url` in `_pkgdown.yml` to the pkgdown site URL for auto-linking.
- Use square brackets around function names in roxygen comments (e.g., `[thisfunction()]`) to enable hyperlinks in the pkgdown site.
- Use backticks and parentheses when mentioning functions in vignettes or articles (e.g., `` `thisfunction()` ``).
- Qualify external package functions with namespace (e.g., `otherpkg::otherfunction()`).
- Use `vignette("some-topic")` to link to vignettes within the same package; use `vignette("some-topic", package = "somepackage")` for other packages.
- Link non-vignette articles as URLs, since `vignette()` syntax cannot be evaluated for them.
- Organize reference index by adding a `reference` field in `_pkgdown.yml` with titles, subtitles, descriptions, and `contents` lists of functions.
- Group articles in `_pkgdown.yml` for better organization, such as featuring key articles upfront.
- Use `usethis::use_article()` to create non-vignette articles for content requiring unlisted dependencies, authentication, or large figures.
- Specify dependencies for non-vignette articles in `Config/Needs/website` field of `DESCRIPTION` if they use packages not in standard dependency fields.
- Set `development` mode in `_pkgdown.yml`:
  - Use `mode: release` by default (documents current source, even if development version).
  - For packages with broad user base, use `mode: auto` to generate a main site for released version and a `dev/` subdirectory for development version.
- Review rendered examples in the reference index to identify improvements in examples or conditional execution.
- Consult `vignette("pkgdown", package = "pkgdown")` for advanced pkgdown customization.
- Refer to `?pkgdown::build_reference` and `?pkgdown::build_articles` for organizing reference and articles.

## Chapter 20: Software Development Practices
- Use an integrated development environment (IDE) with specific support for R and R package development.
- Use Git as the version control system for every R package, treating the package as a Git repository.
- Sync the local Git repository to a hosted service, with GitHub being the recommended choice.
- Host the package source on a platform like GitHub to provide structure for integrating work from multiple contributors.
- Configure automatic execution of development tasks when events like a push or pull request occur in the hosted repository.
- Run `R CMD check` automatically via continuous integration to detect breakage quickly and ensure code portability across platforms.
- Use GitHub Actions (GHA) to set up workflows, such as running `R CMD check` on Linux, macOS, and Windows.
- Utilize `usethis::use_github_action()` to configure GHA workflows, selecting options like `check-standard` for comprehensive checks.
- Commit and push changes to trigger GHA workflow runs, monitoring results via the repository's Actions tab and README badge.
- Add badges to the README to report `R CMD check` results.
- Use GitHub Issues for collecting bug reports and feature requests from users.
- Use GitHub pull requests to enable low-friction contributions from external developers to fix bugs or add features.
- Enable easy installation of the development version of the package using functions like `devtools::install_github()`.
- Deploy the package website using GitHub Pages for distribution.
- Configure additional GHA workflows for tasks like computing test coverage or building and deploying the package website.
- For solo developers, still adopt Git and CI to avoid issues like “works on my machine” and improve package quality.

## Chapter 21: Lifecycle
- **Use package version numbers to signal meaningful changes**: Increment the `Version` field in `DESCRIPTION` to mark releases, as this is the primary way to communicate evolution to users.
- **Follow R version number format**: Ensure versions consist of at least two integers separated by `.` (e.g., `1.0.0`), and use `package_version()` for parsing and comparison.
- **Adopt tidyverse versioning conventions**: Use three-part released versions (`<major>.<minor>.<patch>`), always with `.` as separator; for in-development, add a fourth component starting at `.9000` (e.g., `0.0.0.9000`).
- **Increment development version only for significant features**: Change the `.9000` component (e.g., to `9001`) when adding important features detectable by users or other packages.
- **Classify changes by backward compatibility**: Assess if changes are backward compatible; breaking changes include removing functions, arguments, or narrowing valid inputs.
- **Determine release type based on changes**: Use patch release for bug fixes without breaking changes; minor release for backward compatible additions and fixes; major release for breaking changes or when reaching `1.0.0` to indicate stable API.
- **Start packages at `0.0.0.9000`**: Initialize new packages with this version using tools like `usethis::create_package()`.
- **Increment versions intentionally**: Use `usethis::use_version()` to interactively or programmatically update `Version` in `DESCRIPTION` and add entries to `NEWS.md`.
- **Apply lifecycle stages explicitly**: Mark functions or arguments as experimental, deprecated, or superseded using the `lifecycle` package to indicate status beyond stable default.
- **Use lifecycle badges for communication**: Include badges (e.g., via `lifecycle::badge()`) in help topics or README to signal stages like experimental or deprecated.
- **Deprecate functions in phases**: Add deprecation warnings with `lifecycle::deprecate_warn()` in the function body, starting in a minor or major release, and remove in a future release after user adaptation time.
- **Deprecate arguments with defaults**: Set deprecated arguments to `lifecycle::deprecated()` and use `lifecycle::deprecate_warn()` to warn users, checking presence with `lifecycle::is_present()`.
- **Centralize deprecation logic**: Create internal helpers (e.g., `warn_for_verbose()`) to handle deprecation across multiple functions, using `lifecycle::deprecate_warn()` with details on alternatives.
- **Handle dependency changes carefully**: Check dependency versions at runtime with `packageVersion()` or `rlang::is_installed()`, and provide fallback code for compatibility with older versions.
- **Mark superseded functions**: Use the superseded lifecycle stage for legacy functions that remain for backward compatibility but receive minimal maintenance, without removal.
- **Communicate changes clearly**: Provide warnings with predictable formats, alternatives, and timing (e.g., once every 8 hours by default) to help users adapt, using `lifecycle` tools for consistency.
- **Balance stakeholder interests**: Prioritize maintainers, existing users, and future users when deciding on breaking changes, favoring backward compatibility but allowing necessary evolution.
- **Run reverse dependency checks**: Test changes against dependent packages using tools to identify breaking impacts before release.
- **Use project-specific libraries for reproducibility**: Recommend tools like renv to isolate package updates and prevent unintended breaking changes in dependent projects.

## Chapter 22: Releasing to CRAN
- Run `R CMD check` regularly during development, preferably using `devtools::check()`, on multiple platforms, and address issues promptly to reduce overall pain and improve compliance.
- Use `usethis::use_release_issue()` to generate a GitHub issue with a release checklist that evolves based on package characteristics.
- Determine the release type (major, minor, patch) early, as it affects the version number and checklist; record this in the release issue but increment the version in `DESCRIPTION` later.
- For first-time CRAN releases, create a `NEWS.md` with `usethis::use_news_md()`, initialize `cran-comments.md` with `usethis::use_cran_comments()`, ensure a `README.md` exists with CRAN installation instructions, and verify `Title`, `Description`, `Authors@R` (including `cph` role), and maintainer email.
- Ensure all exported functions have documented return values with `@returns` and include `@examples` sections; use conditional execution for examples that cannot run on CRAN, avoiding `\dontrun{}` or no examples.
- Check embedded third-party code for proper license declaration.
- Review recent CRAN submission experiences from community resources to preempt common issues.
- For updates to existing CRAN packages, check current CRAN check results for the released version and address any WARNINGs, ERRORs, or NOTEs.
- Polish `NEWS.md` by reorganizing and editing accumulated bullets for clarity and consistency.
- Run `urlchecker::url_check()` to identify and fix non-working URLs or those with permanent redirects; convert problematic URLs to verbatim text if needed.
- Re-build `README.md` from `README.Rmd` using `devtools::build_readme()` to reflect the current package state.
- Perform final `R CMD check` using `devtools::check()` across multiple operating systems (Windows, macOS, Linux) and R versions (released and development), ensuring no ERRORs or WARNINGs; eliminate as many NOTEs as possible, documenting unavoidable ones in `cran-comments.md`.
- Use continuous integration like GitHub Actions to test on multiple platforms regularly.
- For packages with reverse dependencies, identify them and run `R CMD check` on each; use `revdepcheck::revdep_check()` (or `revdepcheck::cloud_check()` if available) to automate checks, comparing CRAN and development versions.
- Review revdep check results in `revdep/` folder (e.g., `problems.md`, `failures.md`, `cran.md`); address breakages by fixing bugs, notifying maintainers, or accepting as breaking changes with advance notice.
- Update `cran-comments.md` with `R CMD check` results (stating errors, warnings, notes) and revdep summary, including explanations for any NOTEs or unchecked packages.
- Bump the version number in `DESCRIPTION` to match the planned release type.
- Submit to CRAN using `devtools::submit_cran()`, which builds the bundle, uploads to CRAN, and records submission details in `CRAN-SUBMISSION`.
- Confirm submission via email link from CRAN to validate maintainer details.
- Upon CRAN acceptance, create a GitHub release with `usethis::use_github_release()`, generating notes from `NEWS.md`.
- Increment to a development version using `usethis::use_dev_version()` and push to GitHub.
- Monitor CRAN landing page post-acceptance for build and check results; prepare patch releases if issues arise.
- Publicize the release via social media, blogs, or communities, describing the package's purpose with examples; link to `NEWS.md` for details, especially for initial or significant updates.