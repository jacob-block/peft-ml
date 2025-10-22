from torch import nn
import torch.nn.functional as F
from MLP import MLPMixer

class AdapterLinear(nn.Linear):
    def __init__(self,in_features, out_features, bias_bool, adapter_dim, num_tasks, weight = None, bias = None):
        super().__init__(in_features, out_features, bias_bool)
        if weight is not None:
            self.weight = weight
        
        self.adapter_dim = adapter_dim
        self.num_tasks = num_tasks
        self.active_task = 0
        self.adapters = nn.ModuleList([nn.Sequential(nn.Linear(self.in_features, adapter_dim, bias=False), nn.Linear(adapter_dim, self.out_features, bias=False)) for i in range(num_tasks)])
        for i in range(num_tasks):
            nn.init.zeros_(self.adapters[i][1].weight)
 
    def forward(self, x):
        output = F.linear(x, self.weight, bias=None)
        if self.adapters:
            output += self.adapters[self.active_task](x)
        return output
 
    @classmethod
    def from_linear(cls, linear: nn.Linear, adapter_dim: int, num_tasks: int) -> "AdapterLinear":
        return cls(linear.in_features, linear.out_features, True, adapter_dim, num_tasks, linear.weight, linear.bias)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class MetaLoraModel(MLPMixer):
    def __init__(self, k_in, T_in, depth, embedding_dim, num_classes=2):
        super().__init__(num_classes=num_classes, depth=depth, dim=embedding_dim)

        def convert_linear_to_adapter(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(module, name, AdapterLinear.from_linear(child, adapter_dim=k_in, num_tasks=T_in))
                else:
                    convert_linear_to_adapter(child)

        convert_linear_to_adapter(self)

    def set_task(self, task_idx):
        for module in self.modules():
            if isinstance(module, AdapterLinear):
                module.active_task = task_idx

    def freeze_base(self):
        visited = set()
        for name, module in self.named_modules():
            if any(name.startswith(p) for p in visited):
                continue  # Skip children of already handled AdapterLinear

            if isinstance(module, AdapterLinear):
                visited.add(name)  # Mark this branch as handled

                if module.weight is not None:
                    module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                for adapter in module.adapters:
                    for param in adapter.parameters():
                        param.requires_grad = True
            else:
                for param in module.parameters():
                    param.requires_grad = False

    def thaw(self):
        for param in self.parameters():
            param.requires_grad = True

    @classmethod
    def from_MLPMixer(cls, mlp_mixer: MLPMixer, k_in: int, T_in: int):
        new_model = cls(k_in=k_in, T_in=T_in, depth=mlp_mixer.depth, embedding_dim=mlp_mixer.dim, num_classes=2)

        new_model.load_state_dict(mlp_mixer.state_dict(), strict=False)
        return new_model
    
    @classmethod
    def from_MetaLoraModel(cls, meta_model: "MetaLoraModel", k_in: int, T_in: int):
        # Step 1: Instantiate a clean MetaLoraModel with new adapters
        new_model = cls(k_in=k_in, T_in=T_in, depth=meta_model.depth, embedding_dim=meta_model.dim, num_classes=2)
        
        # Step 2: Copy over the base weights (excluding adapters)
        base_state_dict = {
            k: v for k, v in meta_model.state_dict().items()
            if "adapters" not in k
        }
        new_model.load_state_dict(base_state_dict, strict=False)
        return new_model
    
class LastLayerModel(MLPMixer):
    def __init__(self, T_in, depth, embedding_dim, num_classes=2):
        super().__init__(num_classes=num_classes, depth=depth, dim=embedding_dim)
        self.active_task = 0

        # Replace the final Linear layer with a ModuleList of heads
        final_feat_dim = self.model[-1].in_features
        self.heads = nn.ModuleList([
            nn.Linear(final_feat_dim, num_classes) for _ in range(T_in)
        ])
        self.model = self.model[:-1]  # remove the original final layer

    def set_task(self, task_idx):
        self.active_task = task_idx

    def forward(self, x):
        feats = self.model(x)
        return self.heads[self.active_task](feats)

    def freeze_base(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def thaw(self):
        for param in self.parameters():
            param.requires_grad = True

    @classmethod
    def from_MLPMixer(cls, mlp_mixer: MLPMixer, T_in: int):
        new_model = cls(T_in=T_in, depth=mlp_mixer.depth, embedding_dim=mlp_mixer.dim, num_classes=2)

        new_model.load_state_dict(mlp_mixer.state_dict(), strict=False)
        return new_model
    
    @classmethod
    def from_LastLayerModel(cls, other_model, T_in: int):
        new_model = cls(T_in=T_in, depth=other_model.depth, dim=other_model.dim, num_classes=2)
        
        base_state_dict = {
            k: v for k, v in other_model.state_dict().items()
            if not k.startswith("heads")
        }
        new_model.load_state_dict(base_state_dict, strict=False)
        return new_model
