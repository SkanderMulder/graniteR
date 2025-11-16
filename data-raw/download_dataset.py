"""
Download malicious prompts dataset from HuggingFace

After running this script, run DATASET.R to create the R data file:
    Rscript data-raw/DATASET.R
"""
import pandas as pd
from datasets import load_dataset

# Load dataset
print("Loading dataset from HuggingFace...")
dataset = load_dataset("ahsanayub/malicious-prompts")

# Get train split and sample
train_df = dataset['train'].to_pandas()

# Sample 1000 rows for vignette (balanced classes)
malicious = train_df[train_df['label'] == 1].sample(n=500, random_state=42)
benign = train_df[train_df['label'] == 0].sample(n=500, random_state=42)
sample_df = pd.concat([malicious, benign]).sample(frac=1, random_state=42)

# Select relevant columns
sample_df = sample_df[['text', 'label', 'source']].reset_index(drop=True)

# Save to CSV in inst/data/
import os
os.makedirs('../inst/data', exist_ok=True)
output_path = '../inst/data/malicious_prompts_sample.csv'
sample_df.to_csv(output_path, index=False)
print(f"Saved {len(sample_df)} samples to {output_path}")
print("\nNext step: Run 'Rscript data-raw/DATASET.R' to create the R data file")

# Print statistics
print(f"\nDataset statistics:")
print(f"Total samples: {len(sample_df)}")
print(f"Malicious (label=1): {(sample_df['label'] == 1).sum()}")
print(f"Benign (label=0): {(sample_df['label'] == 0).sum()}")
print(f"Unique sources: {sample_df['source'].nunique()}")
