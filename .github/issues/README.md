# graniteR GitHub Issues - Development Roadmap

This document contains 7 comprehensive GitHub issues to guide graniteR's development from experimental to production-ready CRAN package.

## Quick Summary

| # | Title | Priority | Labels | Timeline |
|---|-------|----------|--------|----------|
| 1 | CRAN Release Preparation (v1.0) | High | enhancement, documentation, release | Q1 2026 |
| 2 | Batch Tokenization Optimization | Medium | enhancement, performance | 4-6 weeks |
| 3 | Enhanced Hyperparameter Tuning | Medium | enhancement, feature | 3-4 weeks |
| 4 | Comprehensive Benchmarking Suite | High | enhancement, testing, documentation | 5 weeks |
| 5 | Robust Edge Case Handling | High | enhancement, bug, quality | 3-4 weeks |
| 6 | Integration Documentation | Medium | documentation, enhancement | 3 weeks |
| 7 | Custom Model Support | Medium-High | documentation, enhancement | 3 weeks |

## How to Create These Issues

### Option 1: Manual Creation (Recommended)
Copy each issue from the files below and paste into GitHub's "New Issue" form:
- `/tmp/issue_cran_prep.md`
- `/tmp/issue_batch_tokenization.md`
- `/tmp/issue_hyperparameter_ui.md`
- `/tmp/issue_benchmarking.md`
- `/tmp/issue_edge_cases.md`
- `/tmp/issue_integration_docs.md`
- `/tmp/issue_custom_models.md`

### Option 2: GitHub CLI
```bash
gh issue create --title "CRAN Release Preparation (v1.0)" \
  --body-file /tmp/issue_cran_prep.md \
  --label "enhancement,documentation,release"

gh issue create --title "Batch Tokenization Optimization for Large Corpora" \
  --body-file /tmp/issue_batch_tokenization.md \
  --label "enhancement,performance"

gh issue create --title "Enhanced Hyperparameter Tuning Interface" \
  --body-file /tmp/issue_hyperparameter_ui.md \
  --label "enhancement,feature"

gh issue create --title "Comprehensive Benchmarking Suite" \
  --body-file /tmp/issue_benchmarking.md \
  --label "enhancement,testing,documentation"

gh issue create --title "Robust Edge Case Handling" \
  --body-file /tmp/issue_edge_cases.md \
  --label "enhancement,bug,quality"

gh issue create --title "Integration Documentation: extractoR, SpinneR, and Other Packages" \
  --body-file /tmp/issue_integration_docs.md \
  --label "documentation,enhancement"

gh issue create --title "Custom Model Support and Documentation" \
  --body-file /tmp/issue_custom_models.md \
  --label "documentation,enhancement"
```

### Option 3: GitHub API
```bash
# Set your GitHub token
export GITHUB_TOKEN="your_token_here"

# Run the creation script
bash /tmp/create_issues.sh
```

## Issue Details

### 1. CRAN Release Preparation (v1.0)
**Priority**: High | **Labels**: enhancement, documentation, release

Comprehensive checklist for CRAN submission including:
- Package structure compliance
- Documentation polish
- Testing (>80% coverage)
- Multi-platform testing
- API stabilization
- Performance profiling

**Key Deliverables**:
- R CMD check with 0 errors/warnings/notes
- Complete documentation
- CRAN-ready submission package
- Stable API (v1.0)

---

### 2. Batch Tokenization Optimization for Large Corpora
**Priority**: Medium | **Labels**: enhancement, performance

Enable efficient processing of ultra-large datasets (>100K documents):
- Streaming tokenization with configurable batch sizes
- Memory-mapped outputs (HDF5/Arrow)
- Progress tracking and ETA
- Memory usage optimization

**Performance Targets**:
- 10K docs in <5 min (CPU)
- 100K docs in <30 min (GPU)
- Memory <2GB (streaming mode)

---

### 3. Enhanced Hyperparameter Tuning Interface
**Priority**: Medium | **Labels**: enhancement, feature

User-friendly hyperparameter tuning beyond AutoML:
- Explicit tuning interface (grid/random/Bayesian)
- Custom search spaces
- Progress visualization
- Resume interrupted tuning
- Integration with existing AutoML

**Example API**:
```r
tuned <- tune_classifier(
  train_data,
  text_col = text,
  label_col = label,
  search_space = list(
    learning_rate = c(1e-5, 5e-5, 1e-4),
    epochs = c(3, 5, 10)
  ),
  method = "bayesian",
  cv_folds = 5
)
```

---

### 4. Comprehensive Benchmarking Suite
**Priority**: High | **Labels**: enhancement, testing, documentation

Rigorous benchmarking framework to validate performance claims:
- Throughput benchmarks (docs/second)
- Accuracy benchmarks (standard datasets)
- Memory profiling
- Training efficiency comparison
- AutoML evaluation
- Continuous benchmarking in CI/CD

**Deliverables**:
- Public benchmark results (GitHub Pages)
- Comparison vs Python baselines
- Optimization guide
- Automated regression detection

---

### 5. Robust Edge Case Handling
**Priority**: High | **Labels**: enhancement, bug, quality

Systematic edge case handling for production robustness:
- Very long texts (>512 tokens)
- Empty/NA values
- Special characters and encodings
- Class imbalance warnings
- Model compatibility validation
- Helpful error messages
- Graceful fallbacks

**Success Criteria**:
- No silent failures
- 90%+ helpful error messages
- Comprehensive test coverage
- "Handling Common Issues" vignette

---

### 6. Integration Documentation: extractoR, SpinneR, and Other Packages
**Priority**: Medium | **Labels**: documentation, enhancement

Show how graniteR integrates with ecosystem packages:
- **extractoR**: LLM + embeddings pipelines
- **SpinneR**: Training visualization
- **tidymodels**: parsnip engine integration
- **targets**: ML pipeline orchestration
- **text/spacyr**: Linguistic features + embeddings

**Deliverables**:
- Integration vignettes (4-5)
- Example projects
- Helper functions
- Cross-package documentation

---

### 7. Custom Model Support and Documentation
**Priority**: Medium-High | **Labels**: documentation, enhancement

Comprehensive guide for using custom/alternative models:
- Model compatibility guide (which models work)
- Model selection vignette
- Custom model tutorial (local/private models)
- Model utilities (`list_available_models()`, `model_info()`)
- Model comparison tool
- Domain-specific examples (legal, medical, code)
- Multilingual model examples

**Success Criteria**:
- Users can easily find appropriate models
- Clear selection guidance
- Examples with 10+ model types
- Model gallery on pkgdown site

---

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-4)
**Focus**: Stability and quality
- [ ] Issue #5: Edge case handling
- [ ] Issue #4: Benchmarking suite (Phase 1)
- [ ] Issue #7: Custom model docs (Phase 1)

### Phase 2: Enhancement (Weeks 5-8)
**Focus**: Features and documentation
- [ ] Issue #3: Hyperparameter tuning (Phase 1-2)
- [ ] Issue #6: Integration docs
- [ ] Issue #2: Batch optimization
- [ ] Issue #4: Benchmarking suite (Phase 2-3)

### Phase 3: CRAN Preparation (Weeks 9-12)
**Focus**: Release readiness
- [ ] Issue #1: CRAN preparation
- [ ] Final testing and polish
- [ ] Documentation review
- [ ] CRAN submission

## Success Metrics

By completing these issues, graniteR will achieve:
- ✅ CRAN-ready package (stable API, comprehensive docs, >80% test coverage)
- ✅ Production-ready robustness (edge case handling, helpful errors)
- ✅ Performance validation (benchmarks, optimization guide)
- ✅ Ecosystem integration (extractoR, SpinneR, tidymodels)
- ✅ Full Hugging Face Hub support (any encoder model)
- ✅ Advanced features (AutoML, MoE, ensemble, tuning)

This transforms graniteR from experimental (6/10 maturity) to production-ready (9/10), achieving the aspirational state described in the overview.

## Questions or Feedback?

These issues represent a comprehensive roadmap. Feel free to:
- Prioritize differently based on user feedback
- Split large issues into smaller ones
- Add additional issues as needs arise
- Adjust timelines based on resources

---

**Generated**: 2025-11-17
**For**: graniteR v0.1.0 → v1.0.0
**Author**: AI Assistant via Claude Code
