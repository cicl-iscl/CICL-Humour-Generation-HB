import torch
import torch.nn as nn
from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel, XLMRobertaConfig


class HierarchicalConfig(XLMRobertaConfig):
    model_type = "xlm-roberta-joke-rater" 
    def __init__(self, num_child_labels=10, class_weights_binary=None, class_weights_child=None, **kwargs):
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
    
    def __init__(self, config, num_child_labels=10, class_weights_binary=None, class_weights_child=None):
        super().__init__(config)
        h = config.hidden_size

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.5)
        
        self.proj1 = nn.Linear(h, h)
        self.proj2 = nn.Linear(h, h//2)
        self.relu = nn.ReLU()
        
        self.binary_head = nn.Linear(h//2, 2)
        self.child_head = nn.Linear(h//2, num_child_labels)

        self.a = nn.Parameter(torch.tensor([0.5], dtype=torch.float))
        self.b = nn.Parameter(torch.tensor([0.5], dtype=torch.float))
        
        self.loss_bin = nn.CrossEntropyLoss(weight=class_weights_binary)
        self.loss_child = nn.CrossEntropyLoss(weight=class_weights_child)

        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0] # [CLS] token
        x = self.dropout(x)
        x = self.relu(self.proj1(x))
        x = self.relu(self.proj2(x))
        lb = self.binary_head(x) # Logits for binary (0 vs 1-10)
        lc = self.child_head(x) # Logits for child (1-10)
        pb = torch.softmax(lb, dim=-1)
        pc = torch.softmax(lc, dim=-1)

        sc = pc * pb[:, 1].unsqueeze(-1)
        logits = torch.cat([pb[:, 0].unsqueeze(-1), sc], dim=-1)

        probs = torch.softmax(logits, dim=-1)
        expected = (probs * torch.arange(0, logits.size(1), device=logits.device, dtype=torch.float)).sum(dim=-1)
        reg_loss = torch.mean((expected - labels.float()) ** 2)

        loss = None
        if labels is not None:
            bt = (labels != 0).long()
            loss_b = self.loss_bin(lb, bt)
            
            nz = bt == 1
            if nz.any():
                cl = labels[nz] - 1
                loss_c = self.loss_child(lc[nz], cl)
            else:
                loss_c = 0.0 
            
            clf_loss = loss_b + loss_c
            loss = self.a * reg_loss + self.b * clf_loss

        return {"loss": loss, "logits": logits}
