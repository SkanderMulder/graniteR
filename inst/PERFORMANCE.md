# Performance and System Requirements

## System Requirements

### Minimum Requirements
- **R**: >= 4.1.0
- **Python**: >= 3.8
- **RAM**: 4GB (8GB+ recommended for training)
- **Storage**: ~2GB for models and dependencies

### Optional (for GPU acceleration)
- **CUDA-capable GPU**: NVIDIA GPU with compute capability 3.5+
- **CUDA Toolkit**: Version compatible with PyTorch
- **GPU Memory**: 4GB+ VRAM recommended

## CPU vs GPU Performance

### CPU Mode (Default)
- **Embedding generation**: ~100-500 texts/second (depending on CPU)
- **Training**: ~5-10 minutes per epoch on small datasets (1000 samples)
- **Inference**: Real-time for single predictions

### GPU Mode (CUDA)
- **Embedding generation**: ~1000-5000 texts/second
- **Training**: ~1-2 minutes per epoch on small datasets
- **Inference**: Much faster for batch predictions

## Common Issues and Solutions

### CUDA Warning: Driver Too Old

**Symptom:**
```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
```

**Solution:**
This is normal if you don't have an NVIDIA GPU or have old drivers. graniteR automatically falls back to CPU mode. Your code will still work, just slower.

To use GPU:
1. Update your NVIDIA drivers
2. Install CUDA toolkit matching your driver version
3. Reinstall PyTorch with CUDA support: `install_granite()`

### Model Loading Warnings

**Symptom:**
```
Some weights of ModernBertForSequenceClassification were not initialized from 
the model checkpoint and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task
```

**Solution:**
This is **expected behavior** for classification tasks! The warning appears because:
- The base Granite model is pre-trained for embeddings
- Classification layers are randomly initialized
- You MUST train the model before using it for predictions

This is the correct workflow:
```r
# 1. Create classifier (shows warning - this is OK!)
classifier <- granite_classifier(num_labels = 2)

# 2. Train it (required!)
classifier <- granite_train(classifier, train_data, text, label)

# 3. Now predictions will work properly
predictions <- granite_predict(classifier, test_data, text)
```

## Performance Tips

### 1. Batch Processing
Process multiple texts at once for better efficiency:
```r
# Good: batch processing
embeddings <- data |> granite_embed(text)

# Less efficient: one at a time
for (row in 1:nrow(data)) {
  embed <- granite_embed(data[row,], text)
}
```

### 2. Adjust Batch Size
Larger batches = faster but more memory:
```r
# Default: 32
classifier <- granite_train(classifier, data, text, label, batch_size = 32)

# More memory available? Increase batch size
classifier <- granite_train(classifier, data, text, label, batch_size = 64)

# Limited memory? Decrease batch size
classifier <- granite_train(classifier, data, text, label, batch_size = 8)
```

### 3. Use CPU for Small Tasks
For small datasets or single predictions, CPU is often sufficient:
```r
# Explicitly use CPU (default)
classifier <- granite_classifier(num_labels = 2, device = "cpu")
```

### 4. Check Your System
Run a system check to see what's available:
```r
granite_check_system()
```

## Benchmarks

### Embedding Generation (CPU - 8 cores)
| Texts | Time | Speed |
|-------|------|-------|
| 100   | 0.5s | 200/s |
| 1000  | 3s   | 333/s |
| 10000 | 25s  | 400/s |

### Training (CPU - 8 cores, binary classification)
| Dataset Size | Epochs | Time |
|--------------|--------|------|
| 500 samples  | 3      | ~2 min |
| 1000 samples | 3      | ~4 min |
| 5000 samples | 3      | ~15 min |

### Training (GPU - RTX 3080)
| Dataset Size | Epochs | Time |
|--------------|--------|------|
| 500 samples  | 3      | ~15s |
| 1000 samples | 3      | ~25s |
| 5000 samples | 3      | ~90s |

*Benchmarks are approximate and vary by hardware*

## Reducing Memory Usage

### During Training
1. Use smaller batch sizes
2. Train on a subset of data
3. Use gradient accumulation (advanced)

### During Inference
1. Process in smaller batches
2. Clear Python cache between operations:
   ```r
   reticulate::py_run_string("import gc; gc.collect()")
   ```

## Troubleshooting Slow Performance

### Vignettes Building Slowly?
Vignettes should build in 2-3 seconds because they have `eval = FALSE`.
If they're slow:
1. Check that vignettes have `eval = FALSE` in setup chunk
2. Make sure you're not accidentally running the code

### Training Taking Too Long?
1. Check if you're using CPU instead of GPU (expected behavior)
2. Reduce number of epochs
3. Use a smaller dataset for initial testing
4. Consider using the embedding-based approach instead of fine-tuning

### System Hanging?
1. Check memory usage: `free -h` (Linux) or Activity Monitor (Mac)
2. Reduce batch size
3. Close other applications
4. Check if Python process is actually running: `ps aux | grep python`

## Getting Help

If performance issues persist:
1. Run `granite_check_system()` and share output
2. Check system resources (RAM, CPU usage)
3. Open an issue: https://github.com/skandermulder/graniteR/issues
