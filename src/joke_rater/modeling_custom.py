import torch
import torch.nn as nn
from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel, XLMRobertaConfig


class HierarchicalConfig(XLMRobertaConfig):
    model_type = "xlm-roberta-joke-rater"

    def __init__(
        self,
        num_child_labels=10,
        class_weights_binary=None,
        class_weights_child=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_child_labels = num_child_labels
        self.class_weights_binary = class_weights_binary
        self.class_weights_child = class_weights_child


class HierarchicalClassifier(XLMRobertaPreTrainedModel):
    """
    A custom XLM-RoBERTa model with a hierarchical classification head (binary
    and child) and a combined loss function including classification and regression.
    """

    config_class = XLMRobertaConfig

    def __init__(
        self,
        config,
        num_child_labels=10,
        class_weights_binary=None,
        class_weights_child=None,
    ):
        super().__init__(config)
        h = config.hidden_size

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.5)

        self.proj1 = nn.Linear(h, h)
        self.proj2 = nn.Linear(h, h // 2)
        self.relu = nn.ReLU()

        self.binary_head = nn.Linear(h // 2, 2)
        self.child_head = nn.Linear(h // 2, num_child_labels)

        # Register class weights as buffers so they move with the model to GPU
        if class_weights_binary is not None:
            if not isinstance(class_weights_binary, torch.Tensor):
                class_weights_binary = torch.tensor(
                    class_weights_binary, dtype=torch.float
                )
            self.register_buffer("class_weights_binary", class_weights_binary)
        else:
            self.register_buffer("class_weights_binary", None)

        if class_weights_child is not None:
            if not isinstance(class_weights_child, torch.Tensor):
                class_weights_child = torch.tensor(
                    class_weights_child, dtype=torch.float
                )
            self.register_buffer("class_weights_child", class_weights_child)
        else:
            self.register_buffer("class_weights_child", None)

        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state[
            :, 0
        ]  # [CLS] token
        x = self.dropout(x)
        x = self.relu(self.proj1(x))
        x = self.relu(self.proj2(x))
        lb = self.binary_head(x)  # Logits for binary (0 vs 1-10)
        lc = self.child_head(x)  # Logits for child (1-10)

        # Compute combined logits for inference (11 classes: 0-10)
        # Use log-space for numerical stability
        log_pb = torch.log_softmax(lb, dim=-1)
        log_pc = torch.log_softmax(lc, dim=-1)

        # P(class=0) = P(binary=0)
        # P(class=k) = P(binary=1) * P(child=k) for k in 1-10
        # In log space: log P(class=k) = log P(binary=1) + log P(child=k)
        log_probs_nonzero = log_pb[:, 1].unsqueeze(-1) + log_pc  # [batch, 10]
        logits = torch.cat(
            [log_pb[:, 0].unsqueeze(-1), log_probs_nonzero], dim=-1
        )  # [batch, 11]

        loss = None
        if labels is not None:
            # Binary classification loss (main gradient signal)
            bt = (labels != 0).long()
            loss_bin = nn.CrossEntropyLoss(weight=self.class_weights_binary)(lb, bt)

            # Child classification loss (only for non-zero labels)
            nz = bt == 1
            if nz.any():
                cl = labels[nz] - 1
                loss_child = nn.CrossEntropyLoss(weight=self.class_weights_child)(
                    lc[nz], cl
                )
            else:
                # Use a zero that's connected to the computation graph
                loss_child = lc.sum() * 0.0

            # Regression loss for fine-grained ordering
            probs = torch.softmax(logits, dim=-1)
            expected = (
                probs
                * torch.arange(
                    0, logits.size(1), device=logits.device, dtype=torch.float
                )
            ).sum(dim=-1)
            reg_loss = torch.mean((expected - labels.float()) ** 2)

            # Fixed weights instead of learnable (more stable)
            loss = loss_bin + loss_child + 0.1 * reg_loss

        return {"loss": loss, "logits": logits}
