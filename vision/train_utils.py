import torch
from torch.optim import lr_scheduler, Adam, SGD
from torch.nn import CrossEntropyLoss
import numpy as np
from copy import deepcopy

from MLP import MLPMixer
from model import MetaLoraModel, LastLayerModel
from utils import iterate_dloaders

def train_standard(train_loader, val_loader, num_epochs=100, lr=5e-4, weight_decay=1e-5, depth=1, embedding_dim=128, device="cuda", **kwargs):
    sr_model = MLPMixer(num_classes=2, depth=depth, dim=embedding_dim).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(sr_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    num_samples = len(train_loader.dataset)
    num_val_samples = len(val_loader.dataset)
    losses = torch.zeros(num_epochs, device=device)
    accs = torch.zeros(num_epochs, device=device)
    val_accs = torch.zeros(num_epochs, device=device)
    scaler = torch.amp.GradScaler(device)
    best_model = deepcopy(sr_model)
    best_val_acc = 0
    for epoch in range(num_epochs):
        digits_correct = 0
        sr_model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device):
                outputs = sr_model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses[epoch] += loss.detach()
            digits_correct += torch.sum(torch.argmax(outputs,dim=1) == labels).detach()

        scheduler.step()
        accs[epoch] = digits_correct/num_samples

        num_val_correct = 0
        sr_model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device):
                    outputs = sr_model(imgs)
                num_val_correct += torch.sum(torch.argmax(outputs,dim=1) == labels).detach()
        
        val_accs[epoch] = num_val_correct/num_val_samples
        if val_accs[epoch] > best_val_acc:
            best_val_acc = val_accs[epoch]
            best_model = deepcopy(sr_model)
        
    return best_model, losses.cpu().numpy(), accs.cpu().numpy(), val_accs.cpu().numpy()

def train_lora_ml(train_loaders, val_loaders, k=1, num_epochs=100, lr=5e-4, weight_decay=1e-5, depth=1, embedding_dim=128, device="cuda", **kwargs):
    
    # Set Training Params
    meta_model = MetaLoraModel(k_in=k, T_in=len(train_loaders), depth=depth, embedding_dim=embedding_dim).to(device)
    meta_model.thaw()
    optimizer = Adam(meta_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = CrossEntropyLoss()
    num_samples = np.sum([len(dloader.dataset) for dloader in train_loaders])
    num_val_samples = np.sum([len(dloader.dataset) for dloader in val_loaders])

    losses = torch.zeros(num_epochs, device=device)
    accs = torch.zeros(num_epochs, device=device)
    val_accs = torch.zeros(num_epochs, device=device)
    best_val_acc = 0
    best_model = deepcopy(meta_model)

    scaler = torch.amp.GradScaler(device)
    for epoch in range(num_epochs):
        digits_correct = 0
        meta_model.train()
        for (imgs, labels), task_idx in iterate_dloaders(train_loaders):
            imgs, labels = imgs.to(device), labels.to(device)
            meta_model.set_task(task_idx)

            with torch.amp.autocast(device):
                outputs = meta_model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if task_idx == 3:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            losses[epoch] += loss.detach()
            digits_correct += torch.sum(torch.argmax(outputs,dim=1) == labels).detach()

        scheduler.step()
        accs[epoch] = digits_correct/num_samples

        num_val_correct = 0
        meta_model.eval()
        with torch.no_grad():
            for (imgs, labels), task_idx in iterate_dloaders(val_loaders):
                imgs, labels = imgs.to(device), labels.to(device)
                meta_model.set_task(task_idx)
                with torch.amp.autocast(device):
                    outputs = meta_model(imgs)
                num_val_correct += torch.sum(torch.argmax(outputs,dim=1) == labels).detach()
            
            val_accs[epoch] = num_val_correct/num_val_samples
            if val_accs[epoch] > best_val_acc:
                best_val_acc = val_accs[epoch]
                best_model = deepcopy(meta_model)

    return best_model, losses.cpu().numpy(), accs.cpu().numpy(), val_accs.cpu().numpy()

def train_reptile(train_loaders, val_loaders, num_epochs=100, num_epochs_inner=1, lr=5e-4, inner_lr=.01, weight_decay=1e-5, depth=1, embedding_dim=128, device="cuda", **kwargs):   
    reptile_model = MLPMixer(num_classes=2, depth=depth, dim=embedding_dim).to(device)
    train_iters = [iter(dloader) for dloader in train_loaders]
    optimizer = Adam(reptile_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = CrossEntropyLoss()
    num_tasks = len(train_loaders)
    num_val_samples = np.sum([len(dloader.dataset) for dloader in val_loaders])

    val_accs = torch.zeros(num_epochs, device=device)
    best_val_acc = 0
    best_model = deepcopy(reptile_model)

    scaler = torch.amp.GradScaler(device)
    for epoch in range(num_epochs):
        reptile_model.train()
        for task_idx in range(num_tasks):
            inner_model = deepcopy(reptile_model)
            inner_optimizer = SGD(inner_model.parameters(), lr=inner_lr, momentum=0.0, weight_decay=0.0)
            for _ in range(num_epochs_inner):
                try:
                    (imgs, labels) = next(train_iters[task_idx])
                except StopIteration:
                    train_iters[task_idx] = iter(train_loaders[task_idx])
                    (imgs, labels) = next(train_iters[task_idx])

                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device):
                    outputs = inner_model(imgs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(inner_optimizer)
                scaler.update()

                inner_optimizer.zero_grad()

            optimizer.zero_grad()
            reptile_model.reptile_update(inner_model, num_tasks)
            optimizer.step()
        scheduler.step()

        num_val_correct = 0
        reptile_model.eval()
        with torch.no_grad():
            for (imgs, labels), task_idx in iterate_dloaders(val_loaders):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device):
                    outputs = reptile_model(imgs)
                num_val_correct += torch.sum(torch.argmax(outputs,dim=1) == labels).detach()
            
            val_accs[epoch] = num_val_correct/num_val_samples
            if val_accs[epoch] > best_val_acc:
                best_val_acc = val_accs[epoch]
                best_model = deepcopy(reptile_model)

    return best_model, 0, 0, val_accs.cpu().numpy()

def train_last_layer_ml(train_loaders, val_loaders, num_epochs=100, lr=5e-4, weight_decay=1e-5, depth=1, embedding_dim=128, device="cuda", **kwargs):
    
    # Set Training Params
    last_layer_model = LastLayerModel(T_in=len(train_loaders), depth=depth, embedding_dim=embedding_dim).to(device)
    optimizer = Adam(last_layer_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = CrossEntropyLoss()
    num_samples = np.sum([len(dloader.dataset) for dloader in train_loaders])
    num_val_samples = np.sum([len(dloader.dataset) for dloader in val_loaders])

    losses = torch.zeros(num_epochs, device=device)
    accs = torch.zeros(num_epochs, device=device)
    val_accs = torch.zeros(num_epochs, device=device)
    best_val_acc = 0
    best_model = deepcopy(last_layer_model)

    scaler = torch.amp.GradScaler(device)
    for epoch in range(num_epochs):
        digits_correct = 0
        last_layer_model.train()
        for (imgs, labels), task_idx in iterate_dloaders(train_loaders):
            imgs, labels = imgs.to(device), labels.to(device)
            last_layer_model.set_task(task_idx)

            with torch.amp.autocast(device):
                outputs = last_layer_model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if task_idx == 3:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            losses[epoch] += loss.detach()
            digits_correct += torch.sum(torch.argmax(outputs,dim=1) == labels).detach()

        scheduler.step()
        accs[epoch] = digits_correct/num_samples

        num_val_correct = 0
        last_layer_model.eval()
        with torch.no_grad():
            for (imgs, labels), task_idx in iterate_dloaders(val_loaders):
                imgs, labels = imgs.to(device), labels.to(device)
                last_layer_model.set_task(task_idx)
                with torch.amp.autocast(device):
                    outputs = last_layer_model(imgs)
                num_val_correct += torch.sum(torch.argmax(outputs,dim=1) == labels).detach()
            
            val_accs[epoch] = num_val_correct/num_val_samples
            if val_accs[epoch] > best_val_acc:
                best_val_acc = val_accs[epoch]
                best_model = deepcopy(last_layer_model)
    
    return best_model, losses.cpu().numpy(), accs.cpu().numpy(), val_accs.cpu().numpy()

def lora_ft(model, test_loader, test_test_loader, k=1, num_epochs_ft=100, lr_ft=5e-4, weight_decay=1e-5, **kwargs):

    num_test_test_samples = len(test_test_loader.dataset)
    device = model.get_device()

    if isinstance(model, MLPMixer):
        finetuner = MetaLoraModel.from_MLPMixer(model, k_in=k, T_in=1).to(device)
    elif isinstance(model, MetaLoraModel):
        finetuner = MetaLoraModel.from_MetaLoraModel(model, k_in=k, T_in=1).to(device)

    finetuner.set_task(0)
    finetuner.freeze_base()
    optimizer = Adam(finetuner.parameters(), lr=lr_ft, weight_decay=weight_decay)
    scheduler  = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_ft)
    criterion = CrossEntropyLoss()

    test_losses = torch.zeros(num_epochs_ft, device=device)
    test_test_accs = torch.zeros(num_epochs_ft, device=device)
    scaler = torch.amp.GradScaler(device)

    for epoch in range(num_epochs_ft):
        finetuner.train()
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(str(device)):
                outputs = finetuner(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            test_losses[epoch] += loss.detach()

        scheduler.step()
        finetuner.eval()
        with torch.no_grad():
            digits_correct = 0
            for imgs, labels in test_test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = finetuner(imgs)
                digits_correct += torch.sum(torch.argmax(outputs,dim=1) == labels)
            test_test_accs[epoch] = digits_correct/num_test_test_samples

    return test_losses.cpu().numpy(), test_test_accs.cpu().numpy()


def last_layer_ft(model, test_loader, test_test_loader, num_epochs_ft=100, lr_ft=5e-4, weight_decay=1e-5, **kwargs):

    num_test_test_samples = len(test_test_loader.dataset)
    device = model.get_device()

    if isinstance(model, MLPMixer):
        finetuner = LastLayerModel.from_MLPMixer(model, T_in=1).to(device)
    elif isinstance(model, LastLayerModel):
        finetuner = LastLayerModel.from_LastLayerModel(model, T_in=1).to(device)
    else:
        raise RuntimeError("Input Model to head_ft not a MLPMixer Model Type")

    finetuner.set_task(0)
    finetuner.thaw()
    finetuner.freeze_base()
    optimizer = Adam(finetuner.parameters(), lr=lr_ft, weight_decay=weight_decay)
    scheduler  = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_ft)
    criterion = CrossEntropyLoss()

    test_losses = torch.zeros(num_epochs_ft, device=device)
    test_test_accs = torch.zeros(num_epochs_ft, device=device)
    scaler = torch.amp.GradScaler(device)

    for epoch in range(num_epochs_ft):
        finetuner.train()
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(str(device)):
                outputs = finetuner(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            test_losses[epoch] += loss.detach()

        scheduler.step()
        finetuner.eval()
        with torch.no_grad():
            digits_correct = 0
            for imgs, labels in test_test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = finetuner(imgs)
                digits_correct += torch.sum(torch.argmax(outputs,dim=1) == labels)
            test_test_accs[epoch] = digits_correct/num_test_test_samples

    return test_losses.cpu().numpy(), test_test_accs.cpu().numpy()

retrain_fn_map = {
    "lora-ml": train_lora_ml,
    "last-layer-ml": train_last_layer_ml,
    "standard": train_standard,
    "reptile": train_reptile
}

finetune_fn_map = {
    "lora": lora_ft,
    "last-layer": last_layer_ft
}
