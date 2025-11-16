"""
Utility functions for graniteR package
Provides helper functions for working with Granite models in Python
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embeddings(texts, model_name="ibm-granite/granite-embedding-english-r2", device="cpu"):
    """
    Generate embeddings for a list of texts
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return embeddings.cpu().numpy()


def fine_tune_step(model, batch, optimizer, device="cpu"):
    """
    Perform a single fine-tuning step
    """
    model.train()
    optimizer.zero_grad()

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss
    loss.backward()
    optimizer.step()

    return loss.item()
