"""
Mixture of Experts (MoE) classifier for multi-class text classification.

This implementation provides specialized experts for different aspects of
the classification task, dynamically weighted by a gating network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MoEEmotionClassifier(nn.Module):
    """
    Mixture of Experts classifier using multiple specialized heads.

    The model consists of:
    1. A frozen/unfrozen pretrained backbone (e.g., Granite)
    2. Multiple expert networks (deeper feed-forward heads)
    3. A gating network that dynamically weights expert contributions

    Args:
        model_name: HuggingFace model identifier
        num_experts: Number of expert networks (default: 4)
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze the pretrained backbone
        hidden_dim: Hidden dimension for expert networks (default: backbone_size)
        dropout: Dropout probability for expert networks (default: 0.2)
        expert_depth: Number of layers per expert (default: 2)

    Note:
        MoE typically works best with freeze_backbone=False for multi-class tasks.
        With frozen backbone, the standard classifier often performs similarly.
    """

    def __init__(
        self,
        model_name="ibm-granite/granite-embedding-english-r2",
        num_experts=4,
        num_classes=6,
        freeze_backbone=False,
        hidden_dim=None,
        dropout=0.2,
        expert_depth=2
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone
        self.num_experts = num_experts
        self.num_classes = num_classes

        # Freeze or unfreeze backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

        hidden_size = self.backbone.config.hidden_size
        expert_hidden = hidden_dim or hidden_size

        # Experts: deeper specialized feed-forward heads
        def create_expert(depth):
            layers = []
            input_dim = hidden_size

            for i in range(depth):
                output_dim = expert_hidden if i < depth - 1 else num_classes
                layers.extend([
                    nn.Linear(input_dim, output_dim),
                ])

                if i < depth - 1:
                    layers.extend([
                        nn.LayerNorm(output_dim),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    ])

                input_dim = output_dim

            return nn.Sequential(*layers)

        self.experts = nn.ModuleList([
            create_expert(expert_depth) for _ in range(num_experts)
        ])

        # Gating network with more capacity for better routing
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_experts),
            nn.Softmax(dim=1)
        )

        # Load balancing loss coefficient (higher for frozen backbone)
        self.load_balance_coefficient = 0.05 if freeze_backbone else 0.01

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the MoE classifier.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for computing loss

        Returns:
            If labels provided: (loss, logits, gate_weights, expert_outputs)
            Otherwise: (logits, gate_weights, expert_outputs)
        """
        # Get backbone embeddings
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # Use mean pooling instead of CLS token for better representation
            pooled_output = self.mean_pooling(outputs.last_hidden_state, attention_mask)

        # Compute gating weights
        gate_weights = self.gate(pooled_output)  # (batch_size, num_experts)

        # Collect expert outputs
        expert_outputs = torch.stack(
            [expert(pooled_output) for expert in self.experts],
            dim=2
        )  # (batch_size, num_classes, num_experts)

        # Weighted average of expert outputs
        gate_weights_expanded = gate_weights.unsqueeze(1)  # (batch_size, 1, num_experts)
        logits = torch.bmm(
            expert_outputs,
            gate_weights_expanded.transpose(1, 2)
        ).squeeze(2)  # (batch_size, num_classes)

        # Compute loss if labels provided
        if labels is not None:
            # Classification loss
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, labels)

            # Load balancing loss (encourage diverse expert usage)
            load_balance_loss = self.compute_load_balance_loss(gate_weights)

            # Total loss
            loss = classification_loss + self.load_balance_coefficient * load_balance_loss

            return {
                'loss': loss,
                'logits': logits,
                'gate_weights': gate_weights,
                'expert_outputs': expert_outputs,
                'classification_loss': classification_loss,
                'load_balance_loss': load_balance_loss
            }

        return logits, gate_weights, expert_outputs

    def mean_pooling(self, last_hidden_state, attention_mask):
        """
        Mean pooling of token embeddings, considering attention mask.
        """
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def compute_load_balance_loss(self, gate_weights):
        """
        Encourage balanced expert usage across the batch.

        Computes the variance of average gate weights across experts.
        Lower variance means more balanced expert usage.
        """
        # Average gate weight per expert across batch
        avg_gate_weights = gate_weights.mean(dim=0)  # (num_experts,)

        # Compute variance (deviation from uniform distribution)
        uniform_weight = 1.0 / self.num_experts
        variance = ((avg_gate_weights - uniform_weight) ** 2).sum()

        return variance

    def get_expert_specialization(self, dataloader, device):
        """
        Analyze which experts specialize in which classes.

        Returns a dictionary with expert usage statistics per class.
        """
        self.eval()
        expert_class_counts = torch.zeros(self.num_experts, self.num_classes)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                _, gate_weights, _ = self.forward(input_ids, attention_mask)

                # Track which expert was most active for each sample
                primary_experts = gate_weights.argmax(dim=1)  # (batch_size,)

                for expert_idx in range(self.num_experts):
                    mask = primary_experts == expert_idx
                    if mask.any():
                        for class_idx in range(self.num_classes):
                            class_mask = labels == class_idx
                            combined_mask = mask & class_mask
                            expert_class_counts[expert_idx, class_idx] += combined_mask.sum().item()

        return expert_class_counts


class MoETextClassifier(nn.Module):
    """
    General-purpose MoE classifier for any number of classes.

    Similar to MoEEmotionClassifier but with a more generic name
    and slightly different default parameters.
    """

    def __init__(
        self,
        model_name="ibm-granite/granite-embedding-english-r2",
        num_experts=3,
        num_classes=2,
        freeze_backbone=True,
        hidden_dim=None,
        dropout=0.1
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone
        self.num_experts = num_experts
        self.num_classes = num_classes

        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

        hidden_size = self.backbone.config.hidden_size
        expert_hidden = hidden_dim or (hidden_size // 2)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_hidden),
                nn.LayerNorm(expert_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, num_classes)
            )
            for _ in range(num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_experts),
            nn.Softmax(dim=1)
        )

        self.load_balance_coefficient = 0.01

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = self.mean_pooling(outputs.last_hidden_state, attention_mask)

        gate_weights = self.gate(pooled_output)
        expert_outputs = torch.stack(
            [expert(pooled_output) for expert in self.experts],
            dim=2
        )

        gate_weights_expanded = gate_weights.unsqueeze(1)
        logits = torch.bmm(
            expert_outputs,
            gate_weights_expanded.transpose(1, 2)
        ).squeeze(2)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, labels)
            load_balance_loss = self.compute_load_balance_loss(gate_weights)
            loss = classification_loss + self.load_balance_coefficient * load_balance_loss

            return {
                'loss': loss,
                'logits': logits,
                'gate_weights': gate_weights,
                'expert_outputs': expert_outputs
            }

        return logits, gate_weights, expert_outputs

    def mean_pooling(self, last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def compute_load_balance_loss(self, gate_weights):
        avg_gate_weights = gate_weights.mean(dim=0)
        uniform_weight = 1.0 / self.num_experts
        variance = ((avg_gate_weights - uniform_weight) ** 2).sum()
        return variance
