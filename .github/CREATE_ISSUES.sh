#!/bin/bash

# Script to create GitHub issues for graniteR development roadmap
# Generated: 2025-11-17

set -e

REPO="SkanderMulder/graniteR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISSUES_DIR="$SCRIPT_DIR/issues"

echo "=========================================="
echo "  graniteR GitHub Issues Creator"
echo "=========================================="
echo ""

# Check for gh CLI
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) not found"
    echo ""
    echo "Install it from: https://cli.github.com/"
    echo ""
    echo "Or create issues manually from files in: $ISSUES_DIR/"
    exit 1
fi

# Check authentication
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub CLI"
    echo ""
    echo "Run: gh auth login"
    exit 1
fi

echo "This script will create 7 GitHub issues in $REPO"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "Creating issues..."
echo ""

# Issue 1: CRAN Preparation
echo "[1/7] Creating: CRAN Release Preparation (v1.0)"
gh issue create \
  --repo "$REPO" \
  --title "CRAN Release Preparation (v1.0)" \
  --body-file "$ISSUES_DIR/01-cran-preparation.md" \
  --label "enhancement,documentation,release"

# Issue 2: Batch Tokenization
echo "[2/7] Creating: Batch Tokenization Optimization"
gh issue create \
  --repo "$REPO" \
  --title "Batch Tokenization Optimization for Large Corpora" \
  --body-file "$ISSUES_DIR/02-batch-tokenization.md" \
  --label "enhancement,performance"

# Issue 3: Hyperparameter Tuning
echo "[3/7] Creating: Enhanced Hyperparameter Tuning Interface"
gh issue create \
  --repo "$REPO" \
  --title "Enhanced Hyperparameter Tuning Interface" \
  --body-file "$ISSUES_DIR/03-hyperparameter-tuning.md" \
  --label "enhancement,feature"

# Issue 4: Benchmarking
echo "[4/7] Creating: Comprehensive Benchmarking Suite"
gh issue create \
  --repo "$REPO" \
  --title "Comprehensive Benchmarking Suite" \
  --body-file "$ISSUES_DIR/04-benchmarking.md" \
  --label "enhancement,testing,documentation"

# Issue 5: Edge Cases
echo "[5/7] Creating: Robust Edge Case Handling"
gh issue create \
  --repo "$REPO" \
  --title "Robust Edge Case Handling" \
  --body-file "$ISSUES_DIR/05-edge-cases.md" \
  --label "enhancement,bug,quality"

# Issue 6: Integrations
echo "[6/7] Creating: Integration Documentation"
gh issue create \
  --repo "$REPO" \
  --title "Integration Documentation: extractoR, SpinneR, and Other Packages" \
  --body-file "$ISSUES_DIR/06-integrations.md" \
  --label "documentation,enhancement"

# Issue 7: Custom Models
echo "[7/7] Creating: Custom Model Support and Documentation"
gh issue create \
  --repo "$REPO" \
  --title "Custom Model Support and Documentation" \
  --body-file "$ISSUES_DIR/07-custom-models.md" \
  --label "documentation,enhancement"

echo ""
echo "=========================================="
echo "  ✓ All issues created successfully!"
echo "=========================================="
echo ""
echo "View issues at: https://github.com/$REPO/issues"
echo ""
