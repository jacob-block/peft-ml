import numpy as np
from learners import MetaLearner
from utilities import mmt, compute_A_loss, compute_U_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product, cycle
import matplotlib as mpl
import argparse
import os

def main(args):
    rng = np.random.default_rng()

    name = args.name
    dims_list = [5, 10, 20] if name == "d" else [10]
    num_tasks_list = [2, 3, 5] if name == "T" else [3]
    num_samples_list = [100, 1000, 10000] if name == "N" else [5000]
    post_sample_num_list = [100, 1000, 10000] if name == "N_prime" else [100]
    low_rank_dim_list = [1, 2 , 3] if name == "k" else [1]
    sigma_list = [0.1]
    num_runs = 5

    max_dim = max(dims_list)
    max_num_tasks = max(num_tasks_list)
    max_low_rank_dim = max(low_rank_dim_list)

    inner_step_size = 0.003
    outer_step_size = 0.03

    A_star = rng.normal(0, 1, size=(1, max_dim, max_dim))
    A_star = (A_star + A_star.transpose((0, 2, 1))) / 2
    U_star = rng.normal(0, 1, size=(max_num_tasks, max_dim, max_low_rank_dim))
    test_U_star = rng.normal(0, 1, size=(1, max_dim, max_low_rank_dim))
    start_Us = rng.normal(0, 0.1, size=(max_num_tasks, max_dim, max_low_rank_dim * 3))
    start_Vs = rng.normal(0, 0.1, size=(max_num_tasks, max_dim, max_low_rank_dim * 3))

    end_map = {"T": 10000, "d": 1400, "N": 1000, "N_prime": 1000, "k": 10000}
    num_inner_steps = 10
    num_outer_steps = 3000
    final_num_steps = end_map[name]

    prods = product(range(num_runs), dims_list, num_tasks_list, num_samples_list, low_rank_dim_list,
                sigma_list)
    enum = product(range(num_runs), range(len(dims_list)), range(len(num_tasks_list)),
                range(len(num_samples_list)),
                range(len(low_rank_dim_list)), range(len(sigma_list)))
    losses = np.zeros((num_runs, len(dims_list), len(num_tasks_list), len(num_samples_list), len(low_rank_dim_list),
                   len(sigma_list), num_outer_steps + 1))

    A_losses = np.zeros_like(losses)
    U_losses = np.zeros_like(losses)
    post_loss = np.zeros((num_runs, len(dims_list), len(num_tasks_list), len(num_samples_list), len(low_rank_dim_list),
                        len(sigma_list), len(post_sample_num_list), final_num_steps + 1))
    losses_sr = np.zeros_like(losses)
    post_loss_sr = np.zeros_like(post_loss)

    for prod, counts in zip(prods, enum):
        run, dim, num_tasks, num_samples, low_rank_dim, sigma = prod
        num_samples_per_task = num_samples  #// num_tasks
        tasks = A_star[:, :dim, :dim] + mmt(U_star[:num_tasks, :dim, :low_rank_dim])
        data = None

        # LoRA-ML
        learner = MetaLearner(dim, num_samples_per_task, low_rank_dim, num_tasks, tasks, sigma, inner_step_size,
                            outer_step_size, data=data)
        losses[*counts, 0] = learner.semi_true_loss()
        A_losses[*counts, 0] = compute_A_loss(A_star[:, :dim, :dim], learner.A, dim)
        U_losses[*counts, 0] = compute_U_loss(U_star[:num_tasks, :dim, :low_rank_dim], learner.U)
        print(run, dim, num_tasks, num_samples, low_rank_dim, sigma)
        for i in tqdm(range(num_outer_steps)):
            for j in range(num_inner_steps):
                learner.inner_gradient_step()
            learner.outer_gradient_step()
            losses[*counts, i + 1] = learner.semi_true_loss()
            A_losses[*counts, i + 1] = compute_A_loss(A_star[:, :dim, :dim], learner.A, dim)
            U_losses[*counts, i + 1] = compute_U_loss(U_star[:num_tasks, :dim, :low_rank_dim], learner.U)

        # SRT
        data = (np.concatenate([learner.X[i] for i in range(num_tasks)], axis=-1),
                np.concatenate([learner.Y[i] for i in range(num_tasks)], axis=-1))
        data = (data[0].reshape(1, *data[0].shape), data[1].reshape(1, *data[1].shape))
        sr_base_learner = MetaLearner(dim, num_samples_per_task, low_rank_dim, num_tasks, tasks, sigma,
                                    inner_step_size, outer_step_size, data=data)
        
        for i in tqdm(range(num_outer_steps)):
            sr_base_learner.outer_gradient_step()
            losses_sr[*counts, i + 1] = sr_base_learner.semi_true_loss()

        # Test run
        test_tasks = A_star[:, :dim, :dim] + mmt(test_U_star[:, :dim, :low_rank_dim])

        for j in range(len(post_sample_num_list)):
            mult = 3 if num_tasks == 2 else 1
            new_learner = learner.detach(test_tasks, post_sample_num_list[j], mult, new_learning_rate=0.005)
            a_lrd = new_learner.low_rank_dim
            new_learner.U = start_Us[:1, :dim, :a_lrd].copy()
            new_learner.V = start_Vs[:1, :dim, :a_lrd].copy()
            post_loss[*counts, j, 0] = new_learner.semi_true_loss()
            for i in tqdm(range(final_num_steps)):
                new_learner.inner_gradient_step()
                post_loss[*counts, j, i + 1] = new_learner.semi_true_loss()
            
            ft_data = (new_learner.X, new_learner.Y)
            new_sr_learner = sr_base_learner.detach(test_tasks, post_sample_num_list[j], mult,
                                                    new_learning_rate=0.005, data=ft_data)
            post_loss_sr[*counts, j, 0] = new_sr_learner.semi_true_loss()
            for i in tqdm(range(final_num_steps)):
                new_sr_learner.inner_gradient_step()
                post_loss_sr[*counts, j, i + 1] = new_sr_learner.semi_true_loss()

    if args.do_save:
        np.savez(f"./data/{name}.npz", losses=losses, post_loss=post_loss, losses_sr=losses_sr, post_loss_sr=post_loss_sr)

    post_loss_mean = post_loss.mean(axis=0)
    post_loss_lower = post_loss.min(axis=0)
    post_loss_upper = post_loss.max(axis=0)
    post_loss_sr_mean = post_loss_sr.mean(axis=0)
    post_loss_sr_lower = post_loss_sr.min(axis=0)
    post_loss_sr_upper = post_loss_sr.max(axis=0)

    font = {"size": 12}
    mpl.rc("font", **font)
    our_markers = cycle(["v", "^", "<", ">"])
    sr_markers = cycle(["o", "8", "s", "p", "*", "h"])
    our_colors = cycle(["C0", "C1", "C2", "C3"])
    sr_colors = cycle(["C4", "C5", "C6", "C7"])

    def string_producer_post(dim, num_tasks, num_samples, low_rank_dim, post_sample_num):
        mapping = {"d": dim, "T": num_tasks, "N": num_samples, "N_prime": post_sample_num, "k": low_rank_dim}
        type_name = name if name != "N_prime" else "n"
        return type_name + f"={mapping[name]}"

    end_map = {"T": 10000, "d": 1400, "N": 1000, "N_prime": 1000, "k": 1000}
    end = end_map[name]
    markevery = end // 20
    prods = product(dims_list, num_tasks_list, num_samples_list, low_rank_dim_list, sigma_list)
    enum = product(range(len(dims_list)), range(len(num_tasks_list)), range(len(num_samples_list)),
                range(len(low_rank_dim_list)), range(len(sigma_list)))
    
    plt.figure("Losses on New Task")
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Test Task Loss", fontsize=16)
    for prod, counts in zip(prods, enum):
        dim, num_tasks, num_samples, low_rank_dim, sigma = prod
        for j in range(len(post_sample_num_list)):
            label = string_producer_post(dim, num_tasks, num_samples, low_rank_dim, post_sample_num_list[j])
            color = next(our_colors)
            plt.semilogy(post_loss_mean[*counts, j, :end],
                        label=label + ", LoRA-ML", marker=next(our_markers), markevery=markevery, color=color)
            plt.fill_between(np.arange(end), post_loss_lower[*counts, j, :end], post_loss_upper[*counts, j, :end],
                            alpha=0.2, color=color)

    prods = product(dims_list, num_tasks_list, num_samples_list, low_rank_dim_list, sigma_list)
    enum = product(range(len(dims_list)), range(len(num_tasks_list)), range(len(num_samples_list)),
                range(len(low_rank_dim_list)), range(len(sigma_list)))
    for prod, counts in zip(prods, enum):
        dim, num_tasks, num_samples, low_rank_dim, sigma = prod
        for j in range(len(post_sample_num_list)):
            label = string_producer_post(dim, num_tasks, num_samples, low_rank_dim, post_sample_num_list[j])
            color = next(sr_colors)
            plt.fill_between(np.arange(end), post_loss_sr_lower[*counts, j, :end], post_loss_sr_upper[*counts, j, :end],
                            alpha=0.2, color=color)
            plt.semilogy(post_loss_sr_mean[*counts, j, :end],
                        label=label + ", SR", marker=next(sr_markers), markevery=markevery, color=color)
    
    plt.legend(loc="upper right")
    plt.grid()
    os.makedirs(args.save_dir, exist_ok=True)
    plt.savefig(os.path.join(args.save_dir, f"vary-{name}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Linear regression experiments")
    parser.add_argument("--name", action="store", choices=["d","T","N","N_prime","k"], help="which parameter to vary", required=True)
    parser.add_argument("--do-save", action="store_true", help="Flag to save data file")
    parser.add_argument("--save-dir", action="store_true", default="./")

    args = parser.parse_args()
    main(args)