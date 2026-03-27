# CRAN Release Preparation (v1.0)

## Overview
Prepare graniteR for submission to CRAN, moving from experimental lifecycle to stable release.

## Goals
- Achieve CRAN compliance
- Establish stable API
- Comprehensive documentation
- Production-ready reliability

## Tasks

### Package Structure
- [ ] Run `devtools::check()` with zero ERRORs, WARNINGs, NOTEs
- [ ] Ensure all examples run successfully or use `\dontrun{}` appropriately
- [ ] Review and update DESCRIPTION metadata
- [ ] Add `cran-comments.md` documenting check results
- [ ] Validate all URLs in documentation (use `urlchecker::url_check()`)

### Documentation
- [ ] Complete all function documentation with proper `@examples`
- [ ] Review and polish all vignettes for CRAN standards
- [ ] Ensure README.md aligns with CRAN version
- [ ] Add citation file (CITATION)
- [ ] Review LICENSE and copyright statements

### Testing
- [ ] Achieve >80% test coverage (`covr::package_coverage()`)
- [ ] Test on multiple platforms (Windows, macOS, Linux)
- [ ] Test with both CPU and CUDA configurations
- [ ] Add tests for edge cases (empty input, very long text, special characters)
- [ ] Ensure all tests properly skip when Python unavailable

### Python Dependencies
- [ ] Clarify SystemRequirements in DESCRIPTION
- [ ] Document Python setup clearly for CRAN users
- [ ] Handle graceful degradation when Python unavailable
- [ ] Test installation on fresh systems

### API Stability
- [ ] Finalize function signatures (no breaking changes post-1.0)
- [ ] Document deprecation policy
- [ ] Update lifecycle badge from experimental to stable
- [ ] Version numbering strategy (semantic versioning)

### Performance
- [ ] Profile and optimize critical paths
- [ ] Document memory requirements
- [ ] Benchmark against common datasets
- [ ] Memory leak testing with long-running processes

### Compliance
- [ ] Check for CRAN policy compliance
- [ ] Ensure no internet access in examples/tests (except with `\dontrun{}`)
- [ ] Verify package size < 5MB (or justify if larger)
- [ ] Clean up any development artifacts

## Success Criteria
- `R CMD check` passes with 0 ERRORs, 0 WARNINGs, 0 NOTEs
- All automated checks pass
- Documentation is comprehensive and polished
- Test coverage >80%
- Python integration documented clearly

## References
- [CRAN Repository Policy](https://cran.r-project.org/web/packages/policies.html)
- [R Packages Book - Release](https://r-pkgs.org/release.html)

## Timeline
Target: Q1 2026
