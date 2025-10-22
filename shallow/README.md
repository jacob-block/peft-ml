Shallow Network Experiments
===========================

------------------------------------------------------------
Running Experiments
------------------------------------------------------------

- `run.py` runs experiments for varying any of the experimental parameters besides the number of tasks T. It is currently set up to vary the ambient dimension `d`.  
- `run_vary_T.py` runs experiments for varying number of tasks `T`. 
  - Note: when `T=2`, we use a LoRA rank of 3k rather than `k` during fine-tuning.

- `make_plots.py` generates the figures in the paper. It is currently setup to plot results for varying d.

The scripts `run.py` and `make_plots.py` can be modified to generate plots for the other experiment variations.
