import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from time import time
from datetime import timedelta
import os
import json

from utils import EasyDict
from train_utils import retrain_fn_map, finetune_fn_map


def main(args):
    args = EasyDict(vars(args))

    # Standard normalization for CIFAR-10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010))

    #Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    print("Collecting Data...")

    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)

    meta_train_01 = []
    meta_train_23 = []
    meta_train_45 = []
    meta_train_67 = []
    meta_train_val_01 = []
    meta_train_val_23 = []
    meta_train_val_45 = []
    meta_train_val_67 = []
    meta_train_total = []
    meta_test = []
    meta_test_test = []

    for i in range(len(trainset)):
        img, label = trainset[i]
        if label < 2:
            meta_train_total.append((img,label%2))
            meta_train_01.append((img,label%2))
        elif label < 4:
            meta_train_total.append((img,label%2))
            meta_train_23.append((img,label%2))
        elif label < 6:
            meta_train_total.append((img,label%2))
            meta_train_45.append((img,label%2))
        elif label < 8:
            meta_train_total.append((img,label%2))
            meta_train_67.append((img,label%2))
        else:
            meta_test.append((img,label%2))
            
    for i in range(len(testset)):
        img, label = testset[i]
        if label < 2:
            meta_train_val_01.append((img,label%2))
        elif label < 4:
            meta_train_val_23.append((img,label%2))
        elif label < 6:
            meta_train_val_45.append((img,label%2))
        elif label < 8:
            meta_train_val_67.append((img,label%2))
        if label in [8,9]:
            meta_test_test.append((img,label%2))

    meta_train_list = [meta_train_01,meta_train_23,meta_train_45,meta_train_67]
    meta_train_val_list = [meta_train_val_01,meta_train_val_23,meta_train_val_45,meta_train_val_67]
    meta_train_val = meta_train_val_01 + meta_train_val_23 + meta_train_val_45 + meta_train_val_67
    seeds = range(args.seed_start, args.seed_end+1)

    valid_peft_methods = ["lora", "last-layer"]
    peft_methods = args.peft_methods.split()
    for peft_method in peft_methods:
        assert peft_method in valid_peft_methods, f"PEFT method {peft_method} specified in {args.peft_methods} not a valid choice from {valid_peft_methods}"

    retrain = retrain_fn_map[args.retrain_method]

    last_accs = np.zeros((len(peft_methods), len(seeds)))
    best_accs = np.zeros((len(peft_methods), len(seeds)))

    print("Starting Training...")

    for seed in seeds:
        start = time()

        # set_seed(seed)
        torch.manual_seed(seed)

        idx = seed - 1

        if args.retrain_method in ["lora-ml","last-layer-ml","reptile"]:
            train_loader = [DataLoader(meta_train, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True) for meta_train in meta_train_list]
            val_loader = [DataLoader(meta_val, batch_size=args.batchsz, shuffle=False, num_workers=4, pin_memory=True) for meta_val in meta_train_val_list]

        else:
            train_loader = DataLoader(meta_train_total, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(meta_train_val, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)
        
        
        test_loader = DataLoader(meta_test, batch_size=args.batchsz, shuffle=True, num_workers=4, pin_memory=True)
        test_test_loader = DataLoader(meta_test_test, batch_size=args.batchsz, shuffle=False, num_workers=4, pin_memory=True)

        model_rt, losses_rt, train_accs_rt, val_accs_rt = retrain(train_loader, val_loader, **args)
        rt_end = time()
        elapsed_str = str(timedelta(seconds=int(rt_end - start)))
        print(f"Finished retraining in {elapsed_str}")

        for i,peft_method in enumerate(peft_methods):
            ft_start = time()
            print(f"Starting PEFT method {peft_method}")
            finetune = finetune_fn_map[peft_method]
            losses_ft, accs_ft_test = finetune(model_rt, test_loader, test_test_loader, **args)
            last_accs[i,idx] = accs_ft_test[-1]
            best_accs[i,idx] = np.max(accs_ft_test)
            elapsed_str = str(timedelta(seconds=int(time() - ft_start)))
            print(f"Finished {peft_method} finetuning in {elapsed_str}")


        elapsed_str = str(timedelta(seconds=int(time()-start)))
        print(f"Finished seed {seed}/{len(seeds)} in {elapsed_str}")

    # Save args as JSON
    os.makedirs(args.save, exist_ok=True)
    args_file = os.path.join(args.save, "args.json")
    with open(args_file, "w") as f:
        json.dump(args, f, indent=2)

    for i,peft_method in enumerate(peft_methods):
        save_path = os.path.join(args.save, peft_method)
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "results.txt")
        with open(save_file, "w") as f:
            f.write("========== Results ==========\n")
            f.write(f"Retrain method: {args.retrain_method}\n")
            f.write(f"PEFT method: {peft_method}\n")
            f.write(f"Seeds: {list(seeds)}\n\n")

            f.write("Last Accuracies:\n")
            f.write(f"{last_accs[i].tolist()}\n")
            f.write(f"Mean: {np.mean(last_accs[i]):.4f} +- {np.std(last_accs[i], ddof=1) / np.sqrt(len(seeds))}\n")
            f.write(f"Median: {np.median(last_accs[i]):.4f}\n\n")

            f.write("Best Accuracies:\n")
            f.write(f"{best_accs[i].tolist()}\n")
            f.write(f"Mean: {np.mean(best_accs[i]):.4f} +- {np.std(best_accs[i], ddof=1) / np.sqrt(len(seeds))}\n")
            f.write(f"Median: {np.median(best_accs[i]):.4f}\n\n")
            f.write("=============================\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PEFT-ML algorithms on CIFAR")
    retraining_methods = ["lora-ml", "last-layer-ml", "standard", "reptile"]
    parser.add_argument("--retrain-method", action="store", choices=retraining_methods, required=True)
    parser.add_argument("--peft-methods", action="store", type=str, required=True)
    parser.add_argument("--num-tasks", action="store", type=int, required=True)
    parser.add_argument("--lora-dim", action="store", type=int, default=1)
    parser.add_argument("--embedding-dim", action="store", type=int, default=512)
    parser.add_argument("--depth", action="store", type=int, default=1)
    parser.add_argument("--batchsz", action="store", type=int, required=True)
    parser.add_argument("--lr", action="store", type=float, required=True)
    parser.add_argument("--lr-ft", action="store", type=float, required=True)
    parser.add_argument("--inner-lr", action="store", type=float, default=1e-2)
    parser.add_argument("--weight-decay", action="store", type=float, required=True)
    parser.add_argument("--num-epochs", action="store", type=int, required=True)
    parser.add_argument("--num-epochs-ft", action="store", type=int, required=True)
    parser.add_argument("--num-epochs-inner", action="store", type=int, default=20)
    parser.add_argument("--device", action="store", default="cuda")
    parser.add_argument("--save", action="store", required=True)
    parser.add_argument("--seed-start", action="store", type=int, default=1)
    parser.add_argument("--seed-end", action="store", type=int, default=5)

    args = parser.parse_args()
    main(args)
