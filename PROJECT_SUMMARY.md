# Project Summary

`world-model-lab-v2` is a compact PyTorch research prototype for studying whether a world model can learn a belief-like latent representation in a partially observed synthetic environment with hidden causal structure. The environment is a 2D grid world with a hidden episode mode and an unobserved charge state, so visually similar observations can imply different futures depending on prior interactions. The model learns to encode observation-action history into a latent state, predict future observations, compare counterfactual futures under alternative action sequences, and estimate its own predictive uncertainty. The strongest results are that the learned latent space supports nontrivial hidden-state structure, counterfactual branch separation, and uncertainty-guided exploration, while the main remaining limitation is conservative decoding of some successful terminal outcomes.

This project studies a concrete technical question: can a compact latent dynamics model preserve hidden but behaviorally relevant information strongly enough to support prediction, imagination, interpretability, and targeted exploration? What makes the setup interesting is that the world is small enough to inspect precisely, but rich enough to force temporal inference through a hidden mode and delayed activation rule. The most important outcomes are the low one-step and rollout error, the latent PCA and t-SNE structure by hidden mode, the counterfactual gap between correct and incorrect futures, and the finding that uncertainty-seeking exploration reaches more informative interactions than a random policy. The main limitation is that the uncertainty mechanism is still heuristic and the model remains somewhat under-confident on certain terminal counterfactual events.

## How to describe this project in an application or interview

`1-sentence version`

I built a PyTorch world-model research prototype that learns latent state in a partially observed synthetic world, then evaluates that latent with rollout, counterfactual reasoning, latent-space analysis, and uncertainty-guided exploration.

`3-sentence version`

The project asks whether a compact latent dynamics model can carry hidden causal information through time rather than only predict the next observation. I designed a synthetic environment with a hidden world mode and delayed activation rule, trained a world model with latent filtering, transition prediction, decoding, and uncertainty estimation, and then evaluated it with rollout, counterfactual branch comparison, PCA/t-SNE latent analysis, and exploration experiments. The result is a small but serious research artifact showing that the model learns usable hidden structure, though calibration and uncertainty quality still leave room for improvement.

`30-second spoken explanation`

This project is a controlled world-model experiment built to test whether a learned latent state can behave like a belief state under partial observability. The environment has hidden episode modes and delayed effects, so the agent cannot infer the future from a single observation alone. I trained a PyTorch latent dynamics model to predict future observations, compare counterfactual futures, and estimate its own uncertainty, then analyzed the learned representation with rollout metrics, latent-space plots, and exploration experiments. The main takeaway is that the latent captures meaningful hidden structure and supports nontrivial counterfactual reasoning, while uncertainty calibration and conservative decoding remain open problems.
