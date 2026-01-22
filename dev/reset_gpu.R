#!/usr/bin/env Rscript
# Standalone GPU Reset Utility
# Usage: Rscript dev/reset_gpu.R

library(reticulate)

cat("\n")
cat(strrep("=", 70), "\n")
cat("GPU Memory Reset Utility\n")
cat(strrep("=", 70), "\n\n")

# Import torch
torch <- import('torch')

if (!torch$cuda$is_available()) {
  cat("CUDA is not available. Nothing to reset.\n")
  quit(status = 0)
}

# Show before state
cat("Before Reset:\n")
cat(sprintf("  Device: %s\n", torch$cuda$get_device_name(0L)))
allocated_before <- torch$cuda$memory_allocated() / 1024^3
reserved_before <- torch$cuda$memory_reserved() / 1024^3
cat(sprintf("  Allocated: %.2f GB\n", allocated_before))
cat(sprintf("  Reserved:  %.2f GB\n", reserved_before))

# Perform reset
cat("\nResetting...\n")

# Clear CUDA cache
torch$cuda$empty_cache()

# Synchronize all CUDA operations
torch$cuda$synchronize()

# Reset peak memory stats
torch$cuda$reset_peak_memory_stats()
torch$cuda$reset_accumulated_memory_stats()

# Force garbage collection multiple times
for (i in 1:5) {
  gc(verbose = FALSE, full = TRUE)
  Sys.sleep(0.3)
}

# Clear cache again
torch$cuda$empty_cache()

# Show after state
cat("\nAfter Reset:\n")
allocated_after <- torch$cuda$memory_allocated() / 1024^3
reserved_after <- torch$cuda$memory_reserved() / 1024^3
cat(sprintf("  Allocated: %.2f GB\n", allocated_after))
cat(sprintf("  Reserved:  %.2f GB\n", reserved_after))

# Show freed memory
freed_allocated <- allocated_before - allocated_after
freed_reserved <- reserved_before - reserved_after
cat(sprintf("\nFreed:\n"))
cat(sprintf("  Allocated: %.2f GB\n", freed_allocated))
cat(sprintf("  Reserved:  %.2f GB\n", freed_reserved))

cat("\n")
cat(strrep("=", 70), "\n")
cat("Reset Complete!\n")
cat(strrep("=", 70), "\n")
