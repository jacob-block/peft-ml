import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from copy import deepcopy


class AdapterLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias_bool,
        adapter_dim,
        num_tasks,
        lora_alpha,
        p_dropout,
        weight=None,
        bias=None,
    ):
        super().__init__(in_features, out_features, bias_bool)
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias

        self.adapter_dim = adapter_dim
        self.num_tasks = num_tasks
        self.scaling = lora_alpha / adapter_dim
        self.dropout = nn.Dropout(p=p_dropout) if p_dropout > 0 else nn.Identity()
        self.active_task = 0
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.in_features, adapter_dim, bias=False),
                    nn.Linear(adapter_dim, self.out_features, bias=False),
                )
                for i in range(num_tasks)
            ]
        )
        for i in range(num_tasks):
            nn.init.zeros_(self.adapters[i][1].weight)

    def forward(self, x):
        output = F.linear(x, self.weight, bias=self.bias)
        if self.adapters:
            output += self.adapters[self.active_task](self.dropout(x)) * self.scaling
        return output
    
    def set_active_task(self, task_idx):
        self.active_task = task_idx
        for adapter_idx in range(self.num_tasks):
            for p in self.adapters[adapter_idx].parameters():
                p.requires_grad_(adapter_idx == task_idx)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, adapter_dim: int, num_tasks: int, lora_alpha, p_dropout
    ) -> "AdapterLinear":
        return cls(
            linear.in_features,
            linear.out_features,
            True,
            adapter_dim,
            num_tasks,
            lora_alpha,
            p_dropout,
            linear.weight,
            linear.bias,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class RobertaHeadConfig:
    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        self.hidden_size = hidden_size
        self.classifier_dropout = None
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_labels = num_labels


class MultiHeadClassifier(nn.Module):
    def __init__(self, roberta_head: RobertaClassificationHead, num_labels_list):
        super().__init__()
        config_list = [
            RobertaHeadConfig(
                roberta_head.dense.in_features, roberta_head.dropout.p, num_labels
            )
            for num_labels in num_labels_list
        ]
        self.head = nn.ModuleList(
            [RobertaClassificationHead(config) for config in config_list]
        )
        self.active_task = 0
        self.num_tasks = len(config_list)

    def forward(self, x):
        out = self.head[self.active_task].forward(x)
        return out
    
    def set_active_task(self, task_idx):
        self.active_task = task_idx
        for head_idx in range(self.num_tasks):
            for p in self.head[head_idx].parameters():
                p.requires_grad_(head_idx == task_idx)

    @classmethod
    def from_roberta_head(
        cls, roberta_head: RobertaClassificationHead, num_labels_list
    ) -> "MultiHeadClassifier":
        return cls(roberta_head, num_labels_list)
