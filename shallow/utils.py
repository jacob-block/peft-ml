import torch
from torch import nn, transpose
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
import os

def tp(x):
    return transpose(x,-2,-1)

# Define Adapter Class
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
    

class SRModel(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Sigmoid(),
            nn.Linear(dim, 1, bias = False),
        )

    def forward(self, x):
        preds = self.layer.forward(x)
        return preds
    
    def set_task(self, task_idx):
        self.layer[0].active_task = task_idx
        return
    
    def set_base_params(self, A, c):
        state_dict = self.state_dict()
        state_dict["layer.0.weight"] = A
        state_dict["layer.2.weight"] = c
        self.load_state_dict(state_dict)
        return

    def get_base_params(self):
        A = self.layer[0].weight.detach()
        c = self.layer[2].weight.detach()
        return A,c

    def freeze_base(self):
        self.layer[0].weight.requires_grad = False
        self.layer[2].weight.requires_grad = False
    
    def thaw_base(self):
        self.layer[0].weight.requires_grad = True
        self.layer[2].weight.requires_grad = True


    
class MultiModel(SRModel):
    def __init__(self, dim, adapter_dim, num_tasks):
        super().__init__(dim)
        self.layer[0] = AdapterLinear.from_linear(self.layer[0], adapter_dim=adapter_dim, num_tasks = num_tasks)

    def set_task(self, task_idx):
        self.layer[0].active_task = task_idx
        return
    
    def set_adapter_params(self,U):
        # U is Txdxk
        state_dict = self.state_dict()

        T = U.shape[0]
        for i in range(T):
            self.layer[0].adapters[i][0].weight
            state_dict["layer.0.adapters." + str(i) + ".0.weight"] = U[i].T
            state_dict["layer.0.adapters." + str(i) + ".1.weight"] = U[i]

        self.load_state_dict(state_dict)
        return


class FineTuner(MultiModel):
    def __init__(self, dim, adapter_dim, A, c):
        super().__init__(dim, adapter_dim, 1)
        self.set_task(0)
        self.set_base_params(A,c)
        self.freeze_base()


def evaluate(X_ret, y_ret, y_clean_ret, X_ft, y_clean_ft, y_ft, As, k, lr, num_epochs, num_epochs_ft, device, multi=True):

    T,_,d = X_ret.shape

    if multi:
        model = MultiModel(d,k,T)
    else:
        model = SRModel(d)
    
    optimizer_ret = AdamW(params=model.parameters(), lr=lr)
    loss = nn.MSELoss()
    model.to(device)

    losses_ret = torch.zeros((num_epochs,T))
    pred_errs_ret = torch.zeros((num_epochs,T))
    A_diff = torch.zeros(num_epochs)

    for i in tqdm(range(num_epochs)):
        for task_idx in range(T):
            if multi:
                model.set_task(task_idx)
            preds = torch.flatten(model.forward(X_ret[task_idx]))
            output = loss(preds,y_ret[task_idx])
            output.backward()
            losses_ret[i,task_idx] = output.detach().to("cpu")
            with torch.no_grad():
                pred_errs_ret[i,task_idx] = loss(preds, y_clean_ret[task_idx]).detach().to("cpu")
        optimizer_ret.step()
        optimizer_ret.zero_grad()

        with torch.no_grad():
            A_iter, _ = model.get_base_params()
            A_diff[i] = torch.norm(A_iter - As).detach().to("cpu")
    
    with torch.no_grad():
        Ahat, chat = model.get_base_params()
    
    finetuner = FineTuner(d,k,Ahat,chat).to(device)
    optimizer_ft = AdamW(params=finetuner.parameters(), lr=lr)

    losses_ft = torch.zeros((num_epochs_ft))
    pred_errs_ft = torch.zeros((num_epochs_ft))

    for i in range(num_epochs_ft):
        preds = torch.flatten(finetuner.forward(X_ft))
        output = loss(preds,y_ft)
        output.backward()
        with torch.no_grad():
            losses_ft[i] = output.detach().to("cpu")
            pred_errs_ft[i] = loss(preds, y_clean_ft).detach().to("cpu")
        optimizer_ft.step()
        optimizer_ft.zero_grad()

    return [losses_ret, pred_errs_ret, A_diff, losses_ft, pred_errs_ft]

def resolve_path(path_str, default_filename="shallow-plot.png"):
    folder, filename = os.path.split(path_str)

    if filename:
        # folder may be empty string if just filename is passed
        folder_path = folder if folder else "."
        file_name = filename
    else:  # path is just a folder
        folder_path = path_str
        file_name = default_filename

    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)

    return os.path.join(folder_path,file_name)