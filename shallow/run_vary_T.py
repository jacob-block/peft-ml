import torch
from torch import matmul, nn
from torch.optim import AdamW
import argparse
from tqdm import tqdm
import pickle

from utils import SRModel, MultiModel, FineTuner, tp


def evaluate_T(X_ret, y_ret, y_clean_ret, X_ft, y_clean_ft, y_ft, As, k, lr, num_epochs, num_epochs_ft, device, multi=True):

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

    for i in range(num_epochs):
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

    if T == 2:
        losses_ft_3k = torch.zeros((num_epochs_ft))
        pred_errs_ft_3k = torch.zeros((num_epochs_ft))
        finetuner_3k = FineTuner(d,3*k,Ahat,chat).to(device)
        optimizer_ft_3k = AdamW(params=finetuner_3k.parameters(), lr=lr)

        for i in range(num_epochs_ft):
            preds = torch.flatten(finetuner_3k.forward(X_ft))
            output = loss(preds,y_ft)
            output.backward()
            with torch.no_grad():
                losses_ft_3k[i] = output.detach().to("cpu")
                pred_errs_ft_3k[i] = loss(preds, y_clean_ft).detach().to("cpu")
            optimizer_ft_3k.step()
            optimizer_ft_3k.zero_grad()
        
        losses_ft = torch.cat((losses_ft.unsqueeze(0), losses_ft_3k.unsqueeze(0)))
        pred_errs_ft = torch.cat((pred_errs_ft.unsqueeze(0), pred_errs_ft_3k.unsqueeze(0)))
    else:
        losses_ft = torch.cat((losses_ft.unsqueeze(0),torch.zeros(losses_ft.shape).unsqueeze(0)))
        pred_errs_ft = torch.cat((pred_errs_ft.unsqueeze(0), torch.zeros(pred_errs_ft.shape).unsqueeze(0)))


    return [losses_ret, pred_errs_ret, A_diff, losses_ft, pred_errs_ft]


def main(args):
    # Parse arguments
    Ts = [2,3,5,10]
    N = args.N
    n = args.n
    d = args.d
    k = args.k
    noise_sigma = args.noise
    num_epochs = args.num_epochs
    num_epochs_ft = args.num_epochs_ft
    lr = args.lr
    device = args.device
    num_trials = args.num_trials
    seed = args.seed

    torch.manual_seed(seed)
    sigmoid = nn.Sigmoid()

    meta_stats_list = []
    sr_stats_list = []

    for j,T in enumerate(Ts):
        meta_stats = {
            "losses_ret": torch.zeros((num_trials,num_epochs,T)),
            "pred_errs_ret": torch.zeros((num_trials,num_epochs,T)),
            "A_diff": torch.zeros(num_trials,num_epochs),
            "losses_ft": torch.zeros(num_trials,2,num_epochs_ft),
            "pred_errs_ft": torch.zeros(num_trials,2,num_epochs_ft)
            }
        
        sr_stats = {
            "losses_ret": torch.zeros((num_trials,num_epochs,T)),
            "pred_errs_ret": torch.zeros((num_trials,num_epochs,T)),
            "A_diff": torch.zeros(num_trials,num_epochs),
            "losses_ft": torch.zeros(num_trials,2,num_epochs_ft),
            "pred_errs_ft": torch.zeros(num_trials,2,num_epochs_ft)
            }

        for i in tqdm(range(num_trials)):
            # Generate Data
            A = torch.randn((d,d)).to(device)
            Us = torch.randn((T+1,d,k)).to(device)
            c = torch.randn(d).to(device)
            X_retrain = torch.randn((T,N,d)).to(device)

            y_clean_retrain = matmul(c,sigmoid(torch.matmul(A + torch.matmul(Us[:T],tp(Us[:T])),tp(X_retrain))))
            y_retrain = y_clean_retrain + noise_sigma*torch.randn((T,N)).to(device) #T x n

            X_ft = torch.randn((n,d)).to(device)
            y_clean_ft = torch.flatten(matmul(c,sigmoid(torch.matmul(A + torch.matmul(Us[T],tp(Us[T])),tp(X_ft)))))
            y_ft = y_clean_ft + noise_sigma*torch.randn((n)).to(device)

            # train multi
            stats = evaluate_T(X_retrain, y_retrain, y_clean_retrain, X_ft, y_clean_ft, y_ft, A, k, lr, num_epochs, num_epochs_ft, device, multi=True)
            losses_ret, pred_errs_ret, A_diff, losses_ft, pred_errs_ft = stats

            meta_stats["losses_ret"][i] = losses_ret
            meta_stats["pred_errs_ret"][i] = pred_errs_ret
            meta_stats["A_diff"][i] = A_diff
            meta_stats["losses_ft"][i] = losses_ft
            meta_stats["pred_errs_ft"][i] = pred_errs_ft

            # train sr
            stats = evaluate_T(X_retrain, y_retrain, y_clean_retrain, X_ft, y_clean_ft, y_ft, A, k, lr, num_epochs, num_epochs_ft, device, multi=False)
            losses_ret, pred_errs_ret, A_diff, losses_ft, pred_errs_ft = stats

            sr_stats["losses_ret"][i] = losses_ret
            sr_stats["pred_errs_ret"][i] = pred_errs_ret
            sr_stats["A_diff"][i] = A_diff
            sr_stats["losses_ft"][i] = losses_ft
            sr_stats["pred_errs_ft"][i] = pred_errs_ft

        meta_stats_list.append(meta_stats)
        sr_stats_list.append(sr_stats)

    # Saving the objects:
    with open("vary_T.pkl", "wb") as f:
        pickle.dump([meta_stats_list, sr_stats_list, N, n, d, k, Ts, noise_sigma, num_epochs, num_epochs_ft, lr, num_trials],f)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Compare LoRA-ML to SR on 2-Layer Network Data")
    parser.add_argument("--T", action = "store", type=int, help = "number of retraining tasks",default = 3)
    parser.add_argument("--d", action = "store", type=int, help = "ambient dimension",default = 10)
    parser.add_argument("--N", action = "store", type=int, help = "number of retraining samples per task",default = 1000)
    parser.add_argument("--n", action = "store", type=int, help = "number of fine-tuning samples", default = 100)
    parser.add_argument("--k", action = "store", type=int, help = "lora adapter dimension", default = 1)
    parser.add_argument("--noise", action="store", type=float, help = "noise std dev", default = .1)
    parser.add_argument("--num_epochs", action = "store", type=int, help = "number of training epochs", default = 40000)
    parser.add_argument("--num_epochs_ft", action = "store", type=int, help = "number of training epochs", default = 15000)
    parser.add_argument("--num_trials", action = "store", type=int, help = "number of trials to run", default = 10)
    parser.add_argument("--device", action="store", metavar="d", type=str, help="device to train on",default = "cuda:0")
    parser.add_argument("--lr", action="store", type=float, help = "learning rate", default = 1e-3)
    parser.add_argument("--path", action = "store", type=str, help="path to folder to store figs", default = "plots/")
    parser.add_argument("--seed", action = "store", type=int, help = "random seed", default = 613)

    args = parser.parse_args()
    main(args)