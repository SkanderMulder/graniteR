#' Generate embeddings for text data
#'
#' @param data A data frame or tibble containing text
#' @param text_col Column name containing text (unquoted or string)
#' @param model Granite model object (if NULL, creates default embedding model)
#' @param tokenizer Granite tokenizer object (if NULL, creates default tokenizer)
#' @param batch_size Batch size for processing
#' @return Data frame with added embeddings column
#' @export
#' @seealso \code{\link{granite_model}}, \code{\link{granite_tokenizer}}
#' @examplesIf requireNamespace("transformers")
#' library(dplyr)
#' data <- tibble::tibble(text = c("Hello world", "Test sentence"))
#' data |> granite_embed(text_col = text)
granite_embed <- function(
  data,
  text_col,
  model = NULL,
  tokenizer = NULL,
  batch_size = 32
) {
  text_col <- rlang::enquo(text_col)
  text_col_name <- rlang::as_name(text_col)

  if (!text_col_name %in% names(data)) {
    stop(sprintf("Column '%s' not found in data", text_col_name))
  }

  texts <- dplyr::pull(data, .data[[text_col_name]])

  if (is.null(model)) {
    model <- granite_model(task = "embedding")
  }
  if (is.null(tokenizer)) {
    tokenizer <- granite_tokenizer(model$model_name)
  }

  check_model(model)
  check_tokenizer(tokenizer)

  model$model$eval()
  embeddings_list <- list()

  for (i in seq_len(ceiling(length(texts) / batch_size))) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(texts))

    encodings <- tokenizer$tokenizer(
      texts[start_idx:end_idx],
      padding = TRUE,
      truncation = TRUE,
      return_tensors = "pt"
    )

    moved <- to_device(encodings, device = model$device)

    with(torch$no_grad(), {
      outputs <- model$model(
        input_ids = moved$encodings$input_ids,
        attention_mask = moved$encodings$attention_mask
      )
      embeddings_list[[i]] <- outputs$last_hidden_state[, 1L, ]$cpu()$numpy()
    })
  }

  embeddings_df <- as.data.frame(do.call(rbind, embeddings_list))
  names(embeddings_df) <- paste0("emb_", seq_len(ncol(embeddings_df)))

  dplyr::bind_cols(data, embeddings_df)
}
