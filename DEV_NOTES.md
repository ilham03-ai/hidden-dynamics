# Dev Notes

These are the main things I learned while building the project.

## Why the environment stayed vector-based

I tried to keep the first version as inspectable as possible. If the model failed, I wanted to know whether it was because of memory and latent state, not because the encoder also had to solve a perception problem.

## Why I added the probe head

At first I was relying too much on reconstruction loss. That turned out to be misleading. The model could get acceptable next-step error while still being unclear about the hidden switch dependency. The `armed` and `beacon_lit` probes made debugging much easier.

## What surprised me

The counterfactual behavior improved a lot once rollout loss was part of training. Before that, the model often looked fine under teacher forcing and then drifted badly once I asked it to imagine forward without corrections.

## What still feels unresolved

The model often gives the higher score to the correct counterfactual branch before it becomes confident enough to cross a `0.5` threshold. So there is a real calibration issue here, not just a pure representation issue.

## If I revisit this

The first things I would test are:

- no probe-loss ablation
- stochastic latent variant
- image observations with the same hidden rule
- cleaner OOD split rather than only different random seeds and layouts
