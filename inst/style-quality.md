# Tidyverse Style Guide for R

This document provides a concise, checkable set of rules for writing R code that follows the **tidyverse style guide** as of November 2025. It is intended for use in linters, CI/CD, and by developers.

Reference: <https://style.tidyverse.org>

---

### 1. Core Syntax & Naming

| Rule | Correct (✅) | Incorrect (❌) |
| :--- | :--- | :--- |
| **Assignment** | `x <- 5` | `x = 5` |
| **Naming** | `my_variable_name` (snake_case) | `myVariableName`, `my.variable.name` |
| **Spacing** | `x <- 1 + 2`, `mean(x, na.rm = TRUE)` | `x<-1+2`, `mean(x,na.rm=TRUE)` |
| **NA Checks** | `is.na(x)` | `x == NA` |

### 2. Pipes

-   **Always use the native pipe `|>`** (requires R >= 4.1).
-   Never use the `magrittr` pipe `%>%` in new code.

```r
# Correct
mtcars |>
  subset(cyl == 4) |>
  aggregate(. ~ vs, data = _, FUN = mean)

# Incorrect
mtcars %>%
  subset(cyl == 4) %>%
  aggregate(. ~ vs, data = ., FUN = mean)
```

### 3. Preferred Packages & Functions

Use modern `tidyverse` packages over base R or older alternatives.

| Purpose | Use This (Package) | Instead of This (Base R / Old) |
| :--- | :--- | :--- |
| **Data Import** | `readr::read_csv()` | `read.csv()` |
| **Data Manipulation** | `dplyr` verbs (`filter`, `mutate`, etc.) | `subset()`, `transform()`, `merge()` |
| **Data Tidying** | `tidyr` (`pivot_longer`, `separate_wider`) | `reshape()`, `reshape2::gather()` |
| **Functional Programming** | `purrr::map_*()` | `lapply()`, `sapply()` |
| **Strings** | `stringr` (`str_detect`, `str_c`) | `grep()`, `paste()` |
| **Factors** | `forcats` (`fct_reorder`, `fct_lump`) | `factor()`, `relevel()` |
| **Dates & Times** | `lubridate` (`ymd`, `today`) | `as.Date()`, `strptime()` |

### 4. `dplyr` Workflows

-   Use `across()` for operations on multiple columns.
-   Use `relocate()` to reorder columns.
-   Do not use older, superseded functions like `mutate_at()` or `summarise_all()`.

```r
# Correct
df |>
  mutate(across(where(is.character), as.factor)) |>
  summarise(across(where(is.numeric), mean, na.rm = TRUE))
```

### 5. `ggplot2` Style

-   Always place the `+` at the end of a line, never at the beginning of the next.

```r
# Correct
ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  labs(title = "Fuel Efficiency")

# Incorrect
ggplot(mpg, aes(displ, hwy))
  + geom_point()
```

### 6. Package Development

-   **Dependencies:** Use `@importFrom package function` in roxygen2 comments. Never use `library()` or `require()` inside a package.
-   **Data:** Always use `tibble`s (`tibble::tibble()`) instead of `data.frame()`.

---

### 7. Automated Checker Configuration

This YAML block summarizes the rules for automated linting.

```yaml
# Rules for an automated agent / linter
required_pipe: "|>"
forbidden_pipe: "%>%"
assignment_operator: "<-"
spaces_around_infix: true
ggplot_plus_at_end_of_line: true

forbidden_packages:
  - plyr
  - reshape2

forbidden_base_equivalents:
  - read.csv
  - read.table
  - transform
  - merge
  - aggregate
  - reshape
```
