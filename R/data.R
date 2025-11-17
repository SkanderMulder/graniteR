#' Malicious Prompts Sample Dataset
#'
#' A sample dataset of 12,913 labeled text prompts from the HuggingFace
#' malicious-prompts dataset (ahsanayub/malicious-prompts). This dataset
#' contains examples of potentially harmful text inputs designed to bypass
#' safety mechanisms in language models, along with benign prompts.
#'
#' @format A data frame with 12,913 rows and 3 variables:
#' \describe{
#'   \item{text}{Character. The text of the prompt}
#'   \item{label}{Integer. Binary label where 1 indicates malicious and 0 indicates benign}
#'   \item{source}{Character. The source dataset from which the prompt originated}
#' }
#'
#' @source \url{https://huggingface.co/datasets/ahsanayub/malicious-prompts}
#'
#' @examples
#' \dontrun{
#' data(malicious_prompts_sample)
#' head(malicious_prompts_sample)
#' table(malicious_prompts_sample$label)
#' }
"malicious_prompts_sample"

#' Malicious Prompts Full Dataset
#'
#' The complete dataset of 373,646 labeled text prompts from the HuggingFace
#' malicious-prompts dataset (ahsanayub/malicious-prompts). This comprehensive
#' dataset contains examples of potentially harmful text inputs designed to bypass
#' safety mechanisms in language models, along with benign prompts from various sources.
#'
#' @format A data frame with 373,646 rows and 4 variables:
#' \describe{
#'   \item{text}{Character. The text of the prompt}
#'   \item{label}{Integer. Binary label where 1 indicates malicious and 0 indicates benign}
#'   \item{source}{Character. The source dataset from which the prompt originated}
#'   \item{type}{Character. Additional categorization of the prompt type}
#' }
#'
#' @details
#' Label distribution:
#' \itemize{
#'   \item Benign (0): 285,950 prompts (76.5%)
#'   \item Malicious (1): 87,696 prompts (23.5%)
#' }
#'
#' This dataset is significantly larger than \code{malicious_prompts_sample} and is
#' recommended for training production-ready models. Due to its size (453 MB uncompressed),
#' loading may take a few seconds.
#'
#' @source \url{https://huggingface.co/datasets/ahsanayub/malicious-prompts}
#'
#' @examples
#' \dontrun{
#' data(malicious_prompts_full)
#' head(malicious_prompts_full)
#' table(malicious_prompts_full$label)
#' }
"malicious_prompts_full"
