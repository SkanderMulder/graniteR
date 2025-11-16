#' Generate embeddings for text data
#'
#' @param data A data frame or tibble containing text
#' @param text_col Column name containing text (unquoted or string)
#' @param model Granite model object (if NULL, creates default embedding model)
#' @param tokenizer Granite tokenizer object (if NULL, creates default tokenizer)
#' @param batch_size Batch size for processing
#' @return Data frame with added embeddings column
#' @export
#' @examples
#' \dontrun{
#' library(dplyr)
#' data <- tibble(text = c("Hello world", "Test sentence"))
#' data |> granite_embed()
#' }
granite_embed <- function(
  data,
  text_col = text,
  model = NULL,
  tokenizer = NULL,
  batch_size = 32
) {
  text_col <- rlang::enquo(text_col)
  text_col_name <- rlang::as_name(text_col)

  if (!text_col_name %in% names(data)) {
    stop(sprintf("Column '%s' not found in data", text_col_name))
  }

  texts <- dplyr::pull(data, !!text_col)

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
  n_batches <- ceiling(length(texts) / batch_size)

  for (i in seq_len(n_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(texts))
    batch_texts <- texts[start_idx:end_idx]

    encodings <- tokenizer$tokenizer(
      batch_texts,
      padding = TRUE,
      truncation = TRUE,
      return_tensors = "pt"
    )

    if (model$device == "cuda") {
      encodings$input_ids <- encodings$input_ids$to(torch$device("cuda"))
      encodings$attention_mask <- encodings$attention_mask$to(torch$device("cuda"))
    }

    with(torch$no_grad(), {
      outputs <- model$model(
        input_ids = encodings$input_ids,
        attention_mask = encodings$attention_mask
      )

      batch_embeddings <- outputs$last_hidden_state[, 1L, ]$cpu()$numpy()
      embeddings_list[[i]] <- batch_embeddings
    })
  }

  all_embeddings <- do.call(rbind, embeddings_list)

  embeddings_df <- as.data.frame(all_embeddings)
  names(embeddings_df) <- paste0("emb_", seq_len(ncol(embeddings_df)))

  dplyr::bind_cols(data, embeddings_df)
}
