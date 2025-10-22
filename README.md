Provable Meta-Learning with Low-Rank Adaptations
=================================================

Official implementation of the paper:
"Provable Meta-Learning with Low-Rank Adaptations" (NeurIPS 2025)
Paper: https://arxiv.org/abs/2410.22264

This repository contains the code for reproducing all experiments and results from the paper.

------------------------------------------------------------
Repository Structure
------------------------------------------------------------

Each subdirectory corresponds to one experiment suite used in the paper.

```
peft-ml/
├── convai/        # Language experiments
├── vision/        # Vision experiments
├── linear/        # Synthetic linear regression experiments
└── shallow/       # Synthetic shallow network experiments
```

Each subdirectory contains its own README.md describing the setup.

------------------------------------------------------------
Dependencies
------------------------------------------------------------

The core dependencies are listed in requirements.txt at the root of the repository.
You can install them with:

    pip install -r requirements.txt

------------------------------------------------------------
Citation
------------------------------------------------------------

```bibtex
@inproceedings{provableMetaLearningLoRA,
  title={Provable Meta-Learning with Low-Rank Adaptations},
  author={Jacob L. Block and Sundararajan Srinivasan and Liam Collins and Aryan Mokhtari and Sanjay Shakkottai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

------------------------------------------------------------
License
------------------------------------------------------------

This project is licensed under the Apache 2.0 License (see LICENSE file).
