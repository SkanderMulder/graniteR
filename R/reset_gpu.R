#' Reset GPU memory
#'
#' Clears CUDA cache and resets memory statistics. Useful for freeing GPU memory
#' between training runs or when encountering out-of-memory errors.
#'
#' @param verbose Logical; if TRUE, prints memory usage before and after reset
#' @return Invisibly returns TRUE if reset was successful, FALSE otherwise
#' @export
#' @examplesIf requireNamespace("reticulate")
#' \dontrun{
#' reset_gpu()
#' }
reset_gpu <- function(verbose = TRUE) {
  if (is.null(torch)) {
    if (verbose) cli::cli_alert_warning("PyTorch not loaded")
    return(invisible(FALSE))
  }

  if (!torch$cuda$is_available()) {
    if (verbose) cli::cli_alert_info("CUDA not available - nothing to reset")
    return(invisible(FALSE))
  }

  if (verbose) {
    cli::cli_h2("GPU Memory Reset")
    device_name <- torch$cuda$get_device_name(0L)
    cli::cli_alert_info("Device: {device_name}")

    allocated_before <- torch$cuda$memory_allocated() / 1024^3
    reserved_before <- torch$cuda$memory_reserved() / 1024^3
    cli::cli_alert_info("Before - Allocated: {round(allocated_before, 2)} GB, Reserved: {round(reserved_before, 2)} GB")
  } else {
    allocated_before <- torch$cuda$memory_allocated() / 1024^3
    reserved_before <- torch$cuda$memory_reserved() / 1024^3
  }

  # Clear CUDA cache
  torch$cuda$empty_cache()

  # Synchronize all CUDA operations
  torch$cuda$synchronize()

  # Reset peak memory stats
  torch$cuda$reset_peak_memory_stats()
  torch$cuda$reset_accumulated_memory_stats()

  # Force garbage collection
  for (i in 1:3) {
    gc(verbose = FALSE, full = TRUE)
  }

  # Clear cache again
  torch$cuda$empty_cache()

  if (verbose) {
    allocated_after <- torch$cuda$memory_allocated() / 1024^3
    reserved_after <- torch$cuda$memory_reserved() / 1024^3
    freed_allocated <- allocated_before - allocated_after
    freed_reserved <- reserved_before - reserved_after

    cli::cli_alert_info("After - Allocated: {round(allocated_after, 2)} GB, Reserved: {round(reserved_after, 2)} GB")
    cli::cli_alert_success("Freed {round(freed_allocated, 2)} GB allocated, {round(freed_reserved, 2)} GB reserved")
  }

  invisible(TRUE)
}
