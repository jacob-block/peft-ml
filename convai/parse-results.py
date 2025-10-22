import os, json
import numpy as np
import argparse

def parse_results(path, num_tasks=10):
    """
    Parses result files within a given directory.
    Creates a summary file (summary.txt) showing last_val_accs and best_val_accs
    per task (two aligned rows), along with the average across tasks.
    """
    last_val_accs = np.zeros((5, num_tasks))
    best_val_accs = np.zeros((5, num_tasks))
    
    for i in range(5):

        seed_folder = f"seed{i}"
        seed_path = os.path.join(path, seed_folder)
        info_json = os.path.join(seed_path, "info.json")
        info_txt = os.path.join(seed_path, "info.txt")

        if os.path.exists(info_txt):
            os.rename(info_txt, info_json)

        with open(info_json) as f:
            results = json.load(f)

        last_val_accs[i,:] = results["last_val_accs"]
        best_val_accs[i,:] = results["best_val_accs"]

    last_val_accs_mean = np.mean(last_val_accs, axis=0)
    best_val_accs_mean = np.mean(best_val_accs, axis=0)
    last_val_accs_stderr = np.std(last_val_accs, axis=0, ddof=1) / np.sqrt(5)
    best_val_accs_stderr = np.std(best_val_accs, axis=0, ddof=1) / np.sqrt(5)

    # Write summary.txt
    summary_path = os.path.join(path, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Validation Accuracy Summary\n")
        f.write("=" * 35 + "\n\n")

        # Print as two aligned rows
        f.write("Last: " + " ".join(f"{x:.4f} +- {s:.4f}" for (x,s) in zip(last_val_accs_mean, last_val_accs_stderr)) + "\n")
        f.write("Best: " + " ".join(f"{x:.4f} +- {s:.4f}" for (x,s) in zip(best_val_accs_mean,best_val_accs_stderr)) + "\n\n")

        # Averages at the bottom
        f.write(f"Average last acc: {last_val_accs.mean():.4f} +- {last_val_accs.mean(axis=1).std(ddof=1) / np.sqrt(5):.4f}\n")
        f.write(f"Average best acc: {best_val_accs.mean():.4f} +- {best_val_accs.mean(axis=1).std(ddof=1) / np.sqrt(5):.4f}\n")

    return last_val_accs, best_val_accs



def main():
    parser = argparse.ArgumentParser(description="Create summary file for experiment results.")

    # Define arguments
    parser.add_argument("-ft", "--ft-type", choices=["lora","last-layer"], type=str, default="lora", help="Which fine-tuning type results to parse")

    # Parse arguments
    args = parser.parse_args()
    if args.ft_type == "lora":
        ranks = [1, 4, 8, 12, 16]
        methods = ["lora-ml", "standard", "reptile"]

        for method in methods:
            for rank in ranks:
                if method == "lora-ml":
                    path = f"results/{method}/rank{rank}/lora-ft"
                else:
                    path = f"results/{method}/lora-ft/rank{rank}"
                parse_results(path)
    else:
        methods = ["last-layer-ml", "standard", "reptile"]
        for method in methods:
            path = f"results/{method}/head-ft"
            parse_results(path)

if __name__ == "__main__":
    main()

