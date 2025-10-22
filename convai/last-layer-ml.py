import numpy as np
import argparse
import torch
from torch.optim import AdamW
from torch.nn import NLLLoss, LogSoftmax
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
import os
import json
from time import time
import datetime

from conv_utils import file_to_dict_lists, get_dloader_list
from utils import add_only_head_adapters, set_active_task, instantiate_base_model, thaw, to_json_safe, nice_list, validate

def main(args):
    # Parse arguments
    model_name = args.model
    pretrained_base_path = args.from_pretrained_base
    if args.persona_nums_rng[1] > args.persona_nums_rng[0]:
        p_idxs = np.arange(args.persona_nums_rng[0], args.persona_nums_rng[1] + 1)
    else:
        p_idxs = args.persona_nums
    num_epochs = args.num_epochs
    device = torch.device(args.device)
    save_path = args.save
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "args.txt"), "w") as f:
        json.dump(to_json_safe(vars(args)), f, indent=2)
    
    verb = args.verbose
    if verb:
        print(f"Command Line Args: \n {args}")
        print(f"Persona IDXs: {p_idxs}")

    # Set seed for dataset shuffle
    torch.manual_seed(args.seed)

    train_dict_list, valid_dict_list = file_to_dict_lists(args.train_file, args.num_valid, persona_map_path=args.persona_map_path, split="train")

    print("Size of Trains:")
    print([len(train_dict_list[p_idx]["ans"]) for p_idx in p_idxs])

    print(f"Size of Valids: ")
    print([len(valid_dict_list[p_idx]["ans"]) for p_idx in p_idxs])

    tokenizer = RobertaTokenizer.from_pretrained(
       model_name, padding_side="right", cache_dir=args.cache_dir + "/tokenizers"
    )
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2, cache_dir=args.cache_dir + "/transformers"
    )

    tt_dloader_list = get_dloader_list(
        [train_dict_list[p_idx] for p_idx in p_idxs], tokenizer, args.batchsz
    )
    tv_dloader_list = get_dloader_list(
        [valid_dict_list[p_idx] for p_idx in p_idxs], tokenizer, args.batchsz
    )

    num_tasks = len(tt_dloader_list)
    add_only_head_adapters(model, num_labels_list=[2 for i in range(num_tasks)])
    thaw(model)

    if len(pretrained_base_path) > 0:
        checkpoint_base = torch.load(pretrained_base_path, map_location="cpu")
        instantiate_base_model(model, checkpoint_base["model_state_dict"])
        base_metrics = checkpoint_base["val_metrics"]
        print("Loading Saved Model...")
        print(f"Base Validation Metrics: {base_metrics}")
    else:
        print("Training from Downloaded Model...")

    optimizer = AdamW(params=model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.1 * len(tt_dloader_list[0]) * num_epochs,
        num_training_steps=len(tt_dloader_list[0]) * num_epochs * args.sm,
    )
    log_softmax = LogSoftmax(dim=1)
    loss_fn = NLLLoss()

    model.to(device)
    best_val_accs = torch.zeros(num_tasks, device=device)
    best_avg_acc = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        start_time = time()
        model.train()
        losses = torch.zeros(num_tasks, device=device)
        tr_accs = torch.zeros(num_tasks, device=device)
        tr_num = torch.zeros(num_tasks, device=device)

        keep_training_arr = np.ones(num_tasks, dtype=bool)
        iter_dloaders = [iter(tt_dloader_list[i]) for i in range(num_tasks)]

        while np.any(keep_training_arr):
            for task_idx in range(num_tasks):
                if not keep_training_arr[task_idx]:
                    continue
                try:
                    batch = next(iter_dloaders[task_idx])
                except StopIteration:
                    keep_training_arr[task_idx] = False
                    break
                set_active_task(model, task_idx)
                (sz, _, pad_length) = batch["input_ids"].shape

                input_ids = batch["input_ids"].reshape((-1, pad_length)).to(device)
                attention_mask = (batch["attention_mask"].reshape((-1, pad_length)).to(device))
                ans = batch["ans"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                targets = torch.zeros(
                    (sz, 20), dtype=torch.long, requires_grad=False
                ).to(device)
                for i in range(sz):
                    targets[i][ans[i]] = 1
                targets = targets.reshape(-1)

                loss = loss_fn(log_softmax(logits), targets)
                losses[task_idx] += loss.detach()

                logits_resized = outputs.logits.detach().reshape(sz, 20, 2)
                preds = torch.argmax(logits_resized[:, :, 1], dim=1)
                tr_num[task_idx] += sz
                tr_accs[task_idx] += torch.sum(torch.eq(preds, ans))

                # Accumulate gradients for each task
                loss.backward()

                # Update parameters using all gradients from a batch
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

        tr_accs = torch.div(tr_accs, tr_num)
        val_accs = validate(model, tv_dloader_list) # torch tensor on gpu
        best_val_accs = torch.maximum(best_val_accs, val_accs)

        mean = torch.mean(val_accs).item()
        if mean > best_avg_acc:
            best_avg_acc = mean
            best_epoch = epoch

            # Save model
            path = os.path.join(save_path, f"{model_name}.pt")
            stat_dict = {
                "seed": args.seed,
                "epoch": epoch,
                "lr": args.lr,
                "batchsz": args.batchsz,
                "persona_idxs": p_idxs,
                "val_metrics": val_accs.cpu().tolist(),
            }
            model_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_dict = {**stat_dict, **model_dict}
            torch.save(save_dict, path)
            with open(os.path.join(save_path, "info.json"), "w") as f:
                json.dump(to_json_safe(stat_dict), f, indent=2)

        if verb:
            elapsed = time() - start_time
            print(f"Finished Epoch {epoch+1} in {str(datetime.timedelta(seconds=int(elapsed)))}",flush=True)
            print()
            print(f"Epoch {epoch+1} Training Losses: {nice_list(losses)}")
            print(f"Epoch {epoch+1} Training Accuracys: {nice_list(tr_accs)}")
            print(f"Epoch {epoch+1} Validation Accuracy: {nice_list(val_accs)}")
            print()
            print("--------------------------------------------------------")
            print()

    print(f"Best Val Accuracys: {nice_list(best_val_accs)}")
    print(f"Best Avg Accuracy: {best_avg_acc:.4f}")
    print(f"Best Epoch: {best_epoch+1}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run meta-head algorithm using Roberta")
    valid_models = ["roberta-large", "roberta-base"]
    parser.add_argument("--model", action="store", choices=valid_models, help="base model to use", required=True)
    parser.add_argument("--from_pretrained_base", action="store", type=str, help="path to load pretrained saved model", default="")
    parser.add_argument(
        "--persona_nums",
        metavar="p",
        action="store",
        type=int,
        nargs="+",
        help="list of persona indices",
        default=0,
    )
    parser.add_argument(
        "--persona_nums_rng",
        metavar="p",
        action="store",
        type=int,
        nargs=2,
        help="start and end persona indices",
        default=[0, 0],
    )
    
    parser.add_argument("--batchsz", action="store", type=int, help="batch size", required=True)
    parser.add_argument("--num_valid", action="store", type=int, default=1)
    parser.add_argument("--num_epochs", action="store", type=int, help="number of epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="learning rate", required=True)
    parser.add_argument("--sm", action="store", type=float, help="schedule multiplier", default=1.0)
    parser.add_argument("--device", action="store", help="device to train on", default="cuda")
    parser.add_argument("--save", action="store", help="save model checkpoint path", required=True)
    parser.add_argument("--seed", action="store", help="random seed", type=int, default=613)
    parser.add_argument("--verbose", action="store_true", help="verbose output", default=True)
    parser.add_argument("--train_file", action="store", type=str, help="path to training data file", required=True)
    parser.add_argument("--cache_dir", action="store", type=str, help="path to cache directory", required=True)
    parser.add_argument("--persona_map_path", action="store", type=str, help="path to persona map directory", required=True)

    args = parser.parse_args()
    main(args)
