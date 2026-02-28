"""
Utility functions for graniteR package
Provides helper functions for working with Granite models in Python
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
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


class EmbeddingModelForSequenceClassification(nn.Module):
    """
    Wrapper that adds a classification head to an embedding model.
    Used for models that don't have a built-in sequence classification variant.
    """
    def __init__(self, base_model, num_labels, hidden_size=None):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels

        # Try to get hidden size from model config
        if hidden_size is None:
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                hidden_size = base_model.config.hidden_size
            else:
                raise ValueError("Could not determine hidden_size. Please provide it explicitly.")

        # Classification head: dropout + linear layer
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get embeddings from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Get the last hidden state
        if hasattr(outputs, 'last_hidden_state'):
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs[0]

        # Mean pooling over sequence length
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            sum_embeddings = torch.sum(hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_state.mean(dim=1)

        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

    def parameters(self):
        """Return all parameters (base model + classifier)"""
        return super().parameters()

    def to(self, device):
        """Move model to device"""
        super().to(device)
        return self
