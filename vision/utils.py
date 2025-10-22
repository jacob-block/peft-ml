import torch
from torch import nn, transpose
import numpy as np
import random
from typing import Any


def iterate_dloaders(dloaders):    
    # Create an iterator for each dataloader
    iterators = [iter(dloader) for dloader in dloaders]
    active = torch.ones(len(iterators), dtype=bool)
    
    while torch.any(active):  # Continue until all iterators are exhausted
        for task_idx, itr in enumerate(iterators):
            if active[task_idx]:
                try:
                    batch = next(itr)  # Fetch next batch
                    yield batch, task_idx # Yield the batch
                except StopIteration:
                    active[task_idx] = False
                    pass  # Skip exhausted iterators

def tp(x):
    return transpose(x,-2,-1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)