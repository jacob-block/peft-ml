import torch
import numpy as np
from torch import nn
from adapters import AdapterLinear, MultiHeadClassifier
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification


def add_adapters(
    net: RobertaForSequenceClassification,
    adapter_dim: int,
    num_tasks: int,
    num_labels_list: list,
    alpha: int,
    p_dropout,
    only_qv=False,
):
    """Transform Linear layers to adapter layers"""

    # Don't add adapters to classification head
    for module in list(net.roberta.modules()):
        for name, child in module.named_children():
            # Only adapt query and value matrices if specified
            if only_qv and name != "query" and name != "value":
                continue
            if isinstance(child, nn.modules.linear.Linear):
                setattr(
                    module,
                    name,
                    AdapterLinear.from_linear(
                        child,
                        adapter_dim=adapter_dim,
                        num_tasks=num_tasks,
                        lora_alpha=alpha,
                        p_dropout=p_dropout,
                    ),
                )
    net.classifier = MultiHeadClassifier.from_roberta_head(
        net.classifier, num_labels_list
    )


def add_only_head_adapters(
    net: RobertaForSequenceClassification,
    num_labels_list: list,
):
    """Transform Classifier layers to Multihead Classifier layers"""
    net.classifier = MultiHeadClassifier.from_roberta_head(
        net.classifier, num_labels_list
    )


def set_active_task(net, task_idx):
    net.zero_grad()
    for module in list(net.modules()):
        if isinstance(module, AdapterLinear) or isinstance(module, MultiHeadClassifier):
            module.set_active_task(task_idx)
    net.num_labels = net.classifier.head[task_idx].out_proj.out_features


def freeze_base_thaw_adapters(net):
    for name, param in net.named_parameters():
        if "adapters" in name or "classifier" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def freeze_adapters_thaw_base(net):
    for name, param in net.named_parameters():
        if "adapters" in name or "classifier" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)


def freeze_lora_thaw_base(net):
    for name, param in net.named_parameters():
        if "adapters" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)


def freeze_head(net):
    for name, param in net.named_parameters():
        if "classifier" in name:
            param.requires_grad_(False)


def thaw(net):
    for _, param in net.named_parameters():
        param.requires_grad_(True)


def instantiate_model(net, state_dict):
    with torch.no_grad():
        for key in state_dict.keys():
            key_list = key.split(".")
            att = getattr(net, key_list[0])
            for i in range(1, len(key_list)):
                att = getattr(att, key_list[i])
            att.copy_(state_dict[key])


def instantiate_multi_model(net, state_dict_list):
    with torch.no_grad():
        instantiate_model(net, state_dict_list[0])
        for task, state_dict in enumerate(state_dict_list[1:]):
            task = task + 1
            task_str = str(task)
            for key in state_dict.keys():
                if "adapters." in key:
                    new_key = key.replace("adapters.0", "adapters." + task_str)
                    key_list = new_key.split(".")
                    att = getattr(net, key_list[0])
                    for i in range(1, len(key_list)):
                        att = getattr(att, key_list[i])
                    att.copy_(state_dict[key])
                if "classifier.head." in key:
                    new_key = key.replace("head.0", "head." + task_str)
                    key_list = new_key.split(".")
                    att = getattr(net, key_list[0])
                    for i in range(1, len(key_list)):
                        att = getattr(att, key_list[i])
                    att.copy_(state_dict[key])


def instantiate_base_model(net, state_dict):
    with torch.no_grad():
        for key in state_dict.keys():
            if "classifier" in key or "adapters" in key:
                continue
            key_list = key.split(".")
            att = getattr(net, key_list[0])
            for i in range(1, len(key_list)):
                att = getattr(att, key_list[i])
            att.copy_(state_dict[key])
    return


def set_dropout(net, p):
    for mod in net.modules():
        if isinstance(mod, torch.nn.modules.dropout.Dropout):
            mod.p = p

def nice_list(x, num_decimals=3):
    if isinstance(x, torch.Tensor):
        x = x.cpu().tolist()
    return [round(float(xi), num_decimals) for xi in x]

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_json_safe(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    else:
        return obj  # assume it's already JSON-safe

def check_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                print(f"!! Found non-finite gradients in parameter: {name}")
                print(f"NAN? {param.grad.isnan().any()}")
                print(f"INF? {param.grad.isinf().any()}")

def validate(model, val_dloader_list, set_task=True):
    num_tasks = len(val_dloader_list)
    device = "cuda"
    model.eval()
    val_accs = torch.zeros(num_tasks, device=device)

    for task_idx in range(num_tasks):
        if set_task:
            set_active_task(model, task_idx)
        val_num = 0
        for batch in val_dloader_list[task_idx]:
            (sz, _, pad_length) = batch["input_ids"].shape

            input_ids = batch["input_ids"].reshape((-1, pad_length)).to(device)
            attention_mask = (
                batch["attention_mask"].reshape((-1, pad_length)).to(device)
            )
            ans = batch["ans"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits_resized = outputs.logits.detach().reshape(sz, 20, 2)
            preds = torch.argmax(logits_resized[:, :, 1], dim=1)

            val_num += sz
            val_accs[task_idx] += torch.sum(torch.eq(preds, ans))

        val_accs[task_idx] /= val_num
    
    return val_accs