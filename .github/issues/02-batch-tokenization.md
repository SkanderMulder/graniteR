# Batch Tokenization Optimization for Large Corpora

## Overview
Implement efficient batch tokenization to handle ultra-large text corpora without memory issues.

## Problem
Current implementation processes texts individually or in small batches, which can be inefficient for:
- Large datasets (>100K documents)
- Production pipelines processing streaming data
- Memory-constrained environments

## Proposed Solution

### 1. Streaming Tokenization
```r
# Process data in configurable batch sizes
embed(data, text_col, batch_size = 32, stream = TRUE)
```

### 2. Memory-Mapped Outputs
- Option to write embeddings directly to disk
- Support for HDF5 or Arrow formats
- Lazy loading for downstream tasks

### 3. Progress Tracking
- Progress bars for long-running operations
- ETA estimation
- Memory usage monitoring

## Implementation Details

### API Design
```r
# Batch processing with progress
embeddings <- data %>%
  embed(text, 
        batch_size = 64,        # Process 64 texts at a time
        show_progress = TRUE,   # Show progress bar
        output_format = "arrow" # Save to disk incrementally
  )

# Streaming mode for very large data
embeddings <- embed_stream(
  data_stream,
  text_col = text,
  chunk_size = 1000,
  output_path = "embeddings.arrow"
)
```

### Performance Targets
- Process 10K documents in <5 minutes on CPU
- Process 100K documents in <30 minutes on GPU
- Memory usage <2GB regardless of corpus size (streaming mode)

### Technical Approach
1. Modify `inst/python/granite_utils.py` to support batch processing
2. Add `batch_size` parameter to embedding functions
3. Implement chunked processing with progress callbacks
4. Add optional disk-backed storage (Arrow/HDF5)
5. Benchmarking suite to validate improvements

## Benefits
- Handle massive datasets (millions of documents)
- Predictable memory usage
- Better progress visibility for users
- Production-ready for data pipelines

## Testing Plan
- Synthetic datasets: 10K, 100K, 1M documents
- Memory profiling with different batch sizes
- Benchmarks vs current implementation
- Stress testing with concurrent operations

## Related
- Performance profiling (#[TBD])
- Edge case handling (#[TBD])

## Priority
Medium - Nice to have for production use, not blocking for CRAN
