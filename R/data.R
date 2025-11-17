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
#'   \item{id}{Integer. Unique identifier for each prompt}
#'   \item{source}{Character. The source dataset from which the prompt originated}
#'   \item{text}{Character. The text of the prompt}
#'   \item{label}{Integer. Binary label where 1 indicates malicious and 0 indicates benign}
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

#' Emotion Detection Sample Dataset
#'
#' A sample dataset of 15,000 labeled text samples from the dair-ai/emotion dataset.
#' This dataset contains text samples labeled with one of six basic emotions based
#' on Ekman's emotion theory.
#'
#' @format A data frame with 15,000 rows and 3 variables:
#' \describe{
#'   \item{text}{Character. The text content}
#'   \item{label}{Integer. Emotion label (0-5)}
#'   \item{label_name}{Character. Emotion name: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)}
#' }
#'
#' @details
#' Label distribution (approximate):
#' \itemize{
#'   \item Joy: ~34%
#'   \item Sadness: ~29%
#'   \item Anger: ~14%
#'   \item Fear: ~12%
#'   \item Love: ~8%
#'   \item Surprise: ~3%
#' }
#'
#' @source \url{https://huggingface.co/datasets/dair-ai/emotion}
#'
#' @examples
#' \dontrun{
#' data(emotion_sample)
#' head(emotion_sample)
#' table(emotion_sample$label_name)
#' }
"emotion_sample"

#' Emotion Detection Full Dataset
#'
#' The complete dataset of 20,000 labeled text samples from the dair-ai/emotion
#' dataset. This dataset contains text samples labeled with one of six basic
#' emotions (sadness, joy, love, anger, fear, surprise).
#'
#' @format A data frame with 20,000 rows and 3 variables:
#' \describe{
#'   \item{text}{Character. The text content}
#'   \item{label}{Integer. Emotion label (0-5)}
#'   \item{label_name}{Character. Emotion name}
#' }
#'
#' @source \url{https://huggingface.co/datasets/dair-ai/emotion}
#'
#' @examples
#' \dontrun{
#' data(emotion_full)
#' head(emotion_full)
#' table(emotion_full$label_name)
#' }
"emotion_full"

#' Sentiment Analysis Sample Dataset (IMDB)
#'
#' A sample dataset of 10,000 movie reviews from the IMDB dataset, labeled as
#' positive or negative sentiment. This is a classic benchmark dataset for
#' binary sentiment classification.
#'
#' @format A data frame with 10,000 rows and 3 variables:
#' \describe{
#'   \item{text}{Character. The movie review text}
#'   \item{label}{Integer. Sentiment label (0 = negative, 1 = positive)}
#'   \item{label_name}{Character. Sentiment name: negative or positive}
#' }
#'
#' @details
#' The dataset is perfectly balanced with 50% positive and 50% negative reviews.
#'
#' @source \url{https://huggingface.co/datasets/imdb}
#'
#' @examples
#' \dontrun{
#' data(sentiment_imdb_sample)
#' head(sentiment_imdb_sample)
#' table(sentiment_imdb_sample$label_name)
#' }
"sentiment_imdb_sample"

#' Sentiment Analysis Full Dataset (IMDB)
#'
#' The complete IMDB dataset of 50,000 movie reviews, labeled as positive or
#' negative sentiment. This is one of the most popular benchmarks for sentiment
#' analysis in NLP.
#'
#' @format A data frame with 50,000 rows and 3 variables:
#' \describe{
#'   \item{text}{Character. The movie review text}
#'   \item{label}{Integer. Sentiment label (0 = negative, 1 = positive)}
#'   \item{label_name}{Character. Sentiment name}
#' }
#'
#' @details
#' The dataset is perfectly balanced with 25,000 positive and 25,000 negative reviews.
#'
#' @source \url{https://huggingface.co/datasets/imdb}
#'
#' @examples
#' \dontrun{
#' data(sentiment_imdb_full)
#' head(sentiment_imdb_full)
#' table(sentiment_imdb_full$label_name)
#' }
"sentiment_imdb_full"

#' Hate Speech Detection Sample Dataset
#'
#' A sample dataset of 15,000 text samples from the ucberkeley-dlab/measuring-hate-speech
#' dataset. This dataset contains text samples labeled as hate speech or non-hate speech.
#'
#' @format A data frame with 15,000 rows and 3 variables:
#' \describe{
#'   \item{text}{Character. The text content}
#'   \item{label}{Integer. Binary label (0 = non-hate, 1 = hate)}
#'   \item{label_name}{Character. Label name: non-hate or hate}
#' }
#'
#' @details
#' This dataset uses continuous hate speech scores that were binarized at a threshold
#' of 0.5 to create binary labels. The sample contains diverse text types from
#' social media and online platforms.
#'
#' @source \url{https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech}
#'
#' @examples
#' \dontrun{
#' data(hate_speech_sample)
#' head(hate_speech_sample)
#' table(hate_speech_sample$label_name)
#' }
"hate_speech_sample"

#' Hate Speech Detection Full Dataset
#'
#' The complete dataset of 135,556 text samples from the ucberkeley-dlab/measuring-hate-speech
#' dataset. This comprehensive dataset contains text samples labeled as hate speech
#' or non-hate speech.
#'
#' @format A data frame with 135,556 rows and 3 variables:
#' \describe{
#'   \item{text}{Character. The text content}
#'   \item{label}{Integer. Binary label (0 = non-hate, 1 = hate)}
#'   \item{label_name}{Character. Label name}
#' }
#'
#' @details
#' Label distribution:
#' \itemize{
#'   \item Non-hate: 86,283 samples (63.6%)
#'   \item Hate: 49,273 samples (36.4%)
#' }
#'
#' This dataset is suitable for training robust hate speech detection models.
#' Due to its size (23 MB uncompressed), loading may take a few seconds.
#'
#' @source \url{https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech}
#'
#' @examples
#' \dontrun{
#' data(hate_speech_full)
#' head(hate_speech_full)
#' table(hate_speech_full$label_name)
#' }
"hate_speech_full"

#' Example Trained MoE Classifier (Structure Reference Only - NOT for Inference)
#'
#' An example Mixture of Experts classifier object showing the structure of a
#' trained model. This is provided ONLY as a reference for understanding the object
#' structure returned by \code{moe_classifier()} and \code{auto_classify()}.
#'
#' **WARNING: This model CANNOT be used for predictions. The Python objects are null.**
#'
#' @format A trained moe_classifier object with the following components:
#' \describe{
#'   \item{model}{Python model object (pointer is NULL - not usable)}
#'   \item{tokenizer}{Tokenizer object (pointer is NULL - not usable)}
#'   \item{num_labels}{Integer. Number of output classes (6 for emotion detection)}
#'   \item{num_experts}{Integer. Number of expert networks (4)}
#'   \item{device}{Character. Device used for training ("cuda" or "cpu")}
#'   \item{is_trained}{Logical. Whether model has been trained (TRUE)}
#'   \item{model_type}{Character. Model architecture type ("moe")}
#' }
#'
#' @details
#' **CRITICAL - NOT USABLE FOR INFERENCE**:
#' \itemize{
#'   \item The Python model and tokenizer pointers are NULL after saving/loading
#'   \item \code{predict()} will fail with "no applicable method" error
#'   \item This is a fundamental limitation of R-Python integration via reticulate
#'   \item Python objects cannot be serialized and restored across R sessions
#'   \item This object is kept ONLY to show the expected structure
#' }
#'
#' **Why This Exists**:
#' \itemize{
#'   \item Shows what structure to expect from \code{moe_classifier()} training
#'   \item Demonstrates the object components and their types
#'   \item Helps users understand return values
#'   \item Useful for debugging and development
#' }
#'
#' **For Actual Inference - Train a New Model**:
#' \itemize{
#'   \item Use \code{auto_classify()} for automatic selection
#'   \item Use \code{moe_classifier()} for manual configuration
#'   \item Models must be trained in the same R session where they are used
#'   \item Cannot save and reload models for later use (reticulate limitation)
#' }
#'
#' **Model Configuration**:
#' - Task: Emotion detection (6 classes: sadness, joy, love, anger, fear, surprise)
#' - Architecture: Mixture of Experts with 4 expert networks
#' - Backbone: ibm-granite/granite-embedding-english-r2
#' - Training: Full fine-tuning (freeze_backbone = FALSE)
#'
#' **Object Structure Example**:
#' \preformatted{
#' List of 7
#'  $ model      : <Python model pointer>
#'  $ tokenizer  : List of 2
#'    ..$ tokenizer  : <Python tokenizer pointer>
#'    ..$ model_name : chr "ibm-granite/granite-embedding-english-r2"
#'  $ num_labels : num 6
#'  $ num_experts: num 4
#'  $ device     : chr "cuda"
#'  $ is_trained : logi TRUE
#'  $ model_type : chr "moe"
#'  - attr("class") = chr [1:3] "moe_classifier" "granite_classifier" "granite_model"
#' }
#'
#' @seealso
#' \code{\link{moe_classifier}}, \code{\link{auto_classify}}, \code{\link{classifier}}
#'
#' @examples
#' \dontrun{
#' # Load the example (STRUCTURE REFERENCE ONLY)
#' data(clf_moe_example)
#'
#' # Inspect structure - this is what a trained model looks like
#' str(clf_moe_example)
#' class(clf_moe_example)
#'
#' # Check metadata (these work)
#' clf_moe_example$num_labels   # 6
#' clf_moe_example$num_experts  # 4
#' clf_moe_example$is_trained   # TRUE
#'
#' # WARNING: Predictions will NOT work
#' # predict(clf_moe_example, ...) # ERROR: no applicable method
#'
#' # To create a WORKING model, train a new one in the same session:
#' data(emotion_sample)
#'
#' # Option 1: Automatic (recommended)
#' clf <- auto_classify(emotion_sample, text, label)
#' predictions <- predict(clf, emotion_sample, text)  # This works!
#'
#' # Option 2: Manual MoE
#' clf <- moe_classifier(num_labels = 6, num_experts = 4) |>
#'   train_moe(emotion_sample, text, label, epochs = 3)
#' predictions <- predict(clf, emotion_sample, text)  # This works!
#'
#' # Key limitation: Models CANNOT be saved and reloaded for inference
#' # save(clf, file = "my_model.rda")  # Saves structure
#' # load("my_model.rda")              # Loads structure
#' # predict(clf, ...)                 # FAILS - Python objects are NULL
#' }
"clf_moe_example"
