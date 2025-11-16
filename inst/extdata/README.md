# Dataset: Malicious Prompts

## Source

This dataset is a sample from the malicious-prompts dataset available on HuggingFace:
https://huggingface.co/datasets/ahsanayub/malicious-prompts

## Description

The malicious-prompts dataset contains 467,000 labeled examples of text prompts designed to detect potentially harmful inputs to language models. This package includes a balanced sample of 12,913 examples for demonstration and vignette purposes.

## Structure

- **text**: The prompt content
- **label**: Binary classification (0 = benign, 1 = malicious)
- **source**: Origin of the data

## License

MIT License (as per original dataset)

## Citation

If you use this dataset, please cite the original work:

```
@misc{malicious-prompts,
  author = {Ahsan Ayub},
  title = {Malicious Prompts Dataset},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/ahsanayub/malicious-prompts}
}
```

## Usage

The dataset is included in the graniteR package and can be loaded using:

```r
data(malicious_prompts_sample, package = "graniteR")
```

This loads a data frame with 12,913 examples. See `?malicious_prompts_sample` for more details.

For the complete dataset (467,000 examples), download directly from HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("ahsanayub/malicious-prompts")
```
