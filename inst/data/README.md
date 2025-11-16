# Dataset: Malicious Prompts

## Source

This dataset is a sample from the malicious-prompts dataset available on HuggingFace:
https://huggingface.co/datasets/ahsanayub/malicious-prompts

## Description

The malicious-prompts dataset contains 467,000 labeled examples of text prompts designed to detect potentially harmful inputs to language models. This package includes a balanced sample of 1,000 examples (500 malicious, 500 benign) for demonstration purposes.

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

The full dataset can be loaded in R using:

```r
data_path <- system.file("data/malicious_prompts_sample.csv", package = "graniteR")
prompts <- readr::read_csv(data_path)
```

For the complete dataset, download directly from HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("ahsanayub/malicious-prompts")
```
