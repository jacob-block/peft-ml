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
from conv_utils import get_dloader_list, file_to_dict_lists
from utils import add_only_head_adapters, set_active_task, freeze_base_thaw_adapters, instantiate_base_model, nice_list, to_json_safe, validate

def main(args):
    # Parse arguments
    model_name = args.model
    pretrained_base_path = args.from_pretrained_base
    if args.persona_nums_rng[1] > args.persona_nums_rng[0]:
        p_idxs = np.arange(args.persona_nums_rng[0], args.persona_nums_rng[1] + 1)
    else:
        p_idxs = args.persona_nums
    batch_size = args.batchsz
    num_epochs = args.num_epochs
    device = args.device
    save_path = args.save
    os.makedirs(save_path, exist_ok=True)
    verb = args.verbose

    if verb:
        print(f"Command Line Args: \n {args}")

    # Set seed for dataset shuffle
    torch.manual_seed(args.seed)

    train_dict_list, valid_dict_list = file_to_dict_lists(args.valid_file, args.num_valid, persona_map_path=args.persona_map_path)

    print("Size of Trains:")
    print([len(train_dict_list[p_idx]["ans"]) for p_idx in p_idxs])

    print(f"Size of Valids: ")
    print([len(valid_dict_list[p_idx]["ans"]) for p_idx in p_idxs])

    tokenizer = RobertaTokenizer.from_pretrained(model_name, padding_side="right", cache_dir=args.cache_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2, cache_dir=args.cache_dir)

    tr_dloader_list = get_dloader_list([train_dict_list[p_idx] for p_idx in p_idxs], tokenizer, batch_size)
    val_dloader_list = get_dloader_list([valid_dict_list[p_idx] for p_idx in p_idxs], tokenizer, batch_size)

    num_tasks = len(tr_dloader_list)
    add_only_head_adapters(model, num_labels_list=[2 for _ in range(num_tasks)])

    if len(pretrained_base_path) > 0:
        checkpoint_base = torch.load(
            os.path.join(pretrained_base_path,f"{model_name}.pt"), map_location="cpu", weights_only=False
        )
        instantiate_base_model(model, checkpoint_base["model_state_dict"])
        base_metrics = checkpoint_base["val_metrics"]
        print("Loading Saved Model...")
        print(f"Base Validation Metrics: {base_metrics}")
        
        # Print args in output location
        src_path = os.path.join(pretrained_base_path,"args.txt")
        assert os.path.isfile(src_path)
        dest_path = os.path.join(save_path,"base_args.txt")
        assert os.path.isdir(save_path)
        with open(src_path, "r", encoding="utf-8") as f_in:
            content = f_in.read()
        with open(dest_path, "w", encoding="utf-8") as f_out:
            f_out.write(content)
    else:
        print("Training from Downloaded Model...")
    
    freeze_base_thaw_adapters(model)

    optimizers = [AdamW(params=model.parameters(), lr=args.lr) for _ in range(num_tasks)]
    lr_schedulers = [
        get_linear_schedule_with_warmup(
            optimizer=optimizers[i],
            num_warmup_steps=int(0.1 * (len(tr_dloader_list[i]) * num_epochs)),
            num_training_steps=int(len(tr_dloader_list[i]) * num_epochs * args.sm),
        )
        for i in range(num_tasks)
    ]
    log_softmax = LogSoftmax(dim=1)
    loss_fn = NLLLoss()

    model.to(device)
    best_val_accs = torch.zeros(num_tasks)
    best_avg_acc = 0

    init_val_accs = validate(model, val_dloader_list).cpu().numpy()
    print("-------------------------------------------")
    print(f"Initial Validation Accuracy: {nice_list(init_val_accs)}")
    print(f"Mean: {init_val_accs.mean().item():.2f}")
    print("-------------------------------------------")

    for epoch in range(num_epochs):
        start_time = time()
        model.train()
        losses = torch.zeros(num_tasks, device=device)
        tr_accs = torch.zeros(num_tasks, device=device)

        for task_idx in range(num_tasks):
            epoch_loss = 0
            tr_num_correct = 0
            tr_num = 0
            set_active_task(model, task_idx)
            for _, batch in enumerate(tr_dloader_list[task_idx]):
                (sz, _, pad_length) = batch["input_ids"].shape

                input_ids = batch["input_ids"].reshape((-1, pad_length)).to(device)
                attention_mask = (
                    batch["attention_mask"].reshape((-1, pad_length)).to(device)
                )
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
                epoch_loss += loss.detach()

                logits_resized = outputs.logits.detach().reshape(sz, 20, 2)
                preds = torch.argmax(logits_resized[:, :, 1], dim=1)
                tr_num += sz
                tr_num_correct += torch.sum(torch.eq(preds, ans))

                loss.backward()
                optimizers[task_idx].step()
                optimizers[task_idx].zero_grad()
                lr_schedulers[task_idx].step()
                del outputs
                del loss
                del batch

            losses[task_idx] = epoch_loss
            tr_accs[task_idx] = (tr_num_correct / tr_num)

        val_accs = validate(model, val_dloader_list).cpu()
        best_val_accs = torch.maximum(best_val_accs, val_accs)
        val_accs = val_accs.cpu().numpy()

        mean = np.mean(val_accs)
        if mean > best_avg_acc:
            best_avg_acc = mean
        if verb:
            elapsed = time() - start_time
            print(f"Finished Epoch {epoch+1} in {str(datetime.timedelta(seconds=int(elapsed)))}")
            print()
            print(f"Epoch {epoch+1} Training Losses:", nice_list(losses))
            print(f"Epoch {epoch+1} Training Accuracys:", nice_list(tr_accs))
            print(f"Epoch {epoch+1} Validation Accuracy:", nice_list(val_accs))
            print()
            print("--------------------------------------------------------")
            print()

    print("Best Val Accuracys: {}".format(best_val_accs))
    print(f"Best Avg Accuracy: {best_avg_acc}")

    if len(save_path) > 0:
        os.makedirs(save_path, exist_ok=True)
        save_dict = {
            "best_val_accs":nice_list(best_val_accs),
            "best_avg_acc":best_avg_acc,
            "last_val_accs":nice_list(val_accs),
            "last_avg_acc":mean,
            "num_epochs":num_epochs,
            "lr": args.lr,
            "batchsz": batch_size,
            "persona_idxs": p_idxs.tolist(),
        }
        torch.save(save_dict, os.path.join(save_path, "results.pt"))
        with open(os.path.join(save_path, "info.txt"), "w") as f:
            json.dump(to_json_safe(save_dict), f, indent=2)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run last layer (head) fine-tuning")
    valid_models = ["roberta-large", "roberta-base"]
    parser.add_argument("--model", action="store", choices=valid_models, help="base model to use", required=True)
    parser.add_argument("--from_pretrained_base", action="store", type=str, help="path to load pretrained saved model", default="")
    parser.add_argument("--persona_nums",action="store", type=int, nargs="+", help="which persona indices to use", default=0)
    parser.add_argument("--persona_nums_rng", action="store", type=int, nargs=2, help="start and end persona indices", default=[0, 0])
    parser.add_argument("--batchsz", action="store", type=int, help="batch size", default=6)
    parser.add_argument("--num_valid", action="store", type=int, default=1)
    parser.add_argument("--num_epochs", action="store", type=int, help="number of epochs", default=10)
    parser.add_argument("--lr", action="store", type=float, help="learning rate", default=5e-5)
    parser.add_argument("--sm", action="store", type=float, help="lr schedule multiplier", default=1)
    parser.add_argument("--device", action="store", help="device to train on", default="cuda:0")
    parser.add_argument("--save", action="store", help="save model checkpoint path", default="")
    parser.add_argument("--seed", action="store", help="random seed", type=int, default=613)
    parser.add_argument("--verbose", action="store_true", help="verbose output", default=True)
    parser.add_argument("--valid_file",action="store", type=str, help="path to validation data file", required=True)
    parser.add_argument("--cache_dir", action="store", type=str, help="path to cache directory", required=True)
    parser.add_argument("--persona_map_path", action="store", type=str, help="path to persona map directory", required=True)

    args = parser.parse_args()
    main(args)
