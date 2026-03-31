# Project Summary

`Hidden Dynamics` is a compact PyTorch research prototype for studying whether a world model can learn a belief-like latent state in a partially observed environment with hidden causal structure. The environment is small, but it includes an unobserved `armed` variable that determines whether reaching the beacon has any effect, so identical visible states can imply different futures. The model combines latent filtering, action-conditioned transition dynamics, observation decoding, and hidden-state probes, then is evaluated with rollout, counterfactual, probe, and latent-space analyses. The strongest results are that the latent state supports nontrivial hidden-state recovery, interpretable structure in PCA, and counterfactual branch separation, while the main limitation is conservative calibration and weaker performance on filtered mid-trajectory counterfactuals.

This project studies a concrete technical question: whether a compact latent dynamics model can preserve hidden but behaviorally relevant information across time strongly enough to support prediction, counterfactual reasoning, and interpretable state tracking. What is interesting about the setup is that the hidden rule is simple enough to understand precisely but rich enough to force genuine partial observability, making it possible to inspect what the latent state is actually doing rather than just reporting prediction loss. The most important results are the hidden-state probe metrics, the linear probe on posterior latents, the open-loop rollout analysis, and the counterfactual benchmark showing that the model often scores the switch-first branch above the direct branch even when binary calibration remains conservative. The main remaining limitation is that exact counterfactual outcome prediction from filtered mid-trajectory belief states is still the weakest regime.

## How to describe this project in an application or interview

`1-sentence version`

I built a small world-model research prototype in PyTorch that learns latent state in a partially observed grid world, then tests whether that latent is rich enough for rollout, counterfactual reasoning, and interpretable hidden-state tracking.

`3-sentence version`

The project studies whether a compact latent dynamics model can carry hidden causal information through time rather than only predict the next observation. I designed a synthetic environment where a hidden `armed` variable changes the future effect of reaching a beacon, trained a latent world model with rollout and probe losses, and evaluated it with counterfactual branches, latent-space probes, and failure analysis. The result is a small but serious research artifact that shows both what the model learns and where its belief-state approximation still breaks down.

`30-second spoken explanation`

This is a controlled world-model project built to test a specific research question: can a latent dynamics model learn a belief-like internal state when the world has hidden but causally important structure? I used a partially observed grid world where the agent has to remember whether it has activated a hidden switch before touching a beacon. The model learns an internal latent state, predicts future observations, and supports counterfactual rollouts, and I analyze it with hidden-state probes, PCA, rollout-error curves, and explicit failure cases. The main takeaway is that the latent clearly captures meaningful hidden structure, but calibration and harder mid-trajectory counterfactuals remain open problems.
