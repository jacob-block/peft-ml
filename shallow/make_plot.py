import pickle
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import cycle
import argparse
from utils import resolve_path

def main(args):
    font = {
        'family': 'Times',
        'weight': 'bold',
        'size': 12
    }
    mpl.rc('font', **font)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    our_markers = cycle(['v', '^', '<'])
    ptf_markers = cycle(['o', '8', 's'])
    our_colors = cycle(['C0', 'C1', 'C2'])
    ptf_colors = cycle(['C4', 'C5', 'C6'])

    with open(args.pickle_path,'rb') as f:
        meta_stats_list, sr_stats_list, N, n, ds, k, T, noise_sigma, num_epochs, num_epochs_ft, lr, num_trials = pickle.load(f)

    # ------------------------------------------------------------
    # Change this for different parameter sweeps
    labels = [f"d = {ds[i]}" for i in range(len(meta_stats_list))]
    # ------------------------------------------------------------

    for i in range(len(meta_stats_list)):
        meta_stats = meta_stats_list[i]
        pred_errs = meta_stats['pred_errs_ft']
        medians = torch.median(pred_errs,dim=0).values.tolist()
        color = next(our_colors)
        plt.semilogy(medians, label=f"{labels[i]}, LoRA-ML",color=color, marker = next(our_markers),markevery=num_epochs_ft // 20)


        last_epoch_errs = torch.sort(pred_errs[:,-1]).values
        print(f"{labels[i]} Meta Last Value Median: {torch.median(last_epoch_errs):.3f} with central range ({last_epoch_errs[3]:.3f},{last_epoch_errs[-3]:.3f})")


    for i in range(len(sr_stats_list)):
        sr_stats = sr_stats_list[i]
        pred_errs = sr_stats['pred_errs_ft']
        medians = torch.median(pred_errs,dim=0).values.tolist()
        color = next(ptf_colors)
        plt.semilogy(medians, label=f"{labels[i]}, SR", color=color, marker = next(ptf_markers),markevery=num_epochs_ft // 20)
        last_epoch_errs = torch.sort(pred_errs[:,-1]).values
        print(f"{labels[i]} Meta Last Value Median: {torch.median(last_epoch_errs):.3f} with central range ({last_epoch_errs[3]:.3f},{last_epoch_errs[-3]:.3f})")

    plt.legend()
    plt.grid()
    plt.xlabel("Iteration",fontsize=16)
    plt.ylabel("Test Task Loss",fontsize=16)
    plt.ylim([1e-3,5])
    ax = plt.gca()
    ax.set_xticks([0,4000,8000,12000])
    plt.savefig(resolve_path(args.save_path))
    plt.show()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Plot shallow results")
    parser.add_argument("--pickle-path", action = "store", type=str, help = "path to saved pickle file", required=True)
    parser.add_argument("--save-path", action = "store", type=str, help = "path to save plot", default="./")

    args = parser.parse_args()
    main(args)