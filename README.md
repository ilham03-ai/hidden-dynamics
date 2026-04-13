# Hidden Dynamics

Learning belief-like latent world representations in a partially observed synthetic environment

`Hidden Dynamics` is a compact research prototype for studying when a latent dynamics model becomes more than a next-step predictor. The environment is deliberately small, but it contains hidden causal structure that forces the model to carry behaviorally relevant information through time rather than recover everything from the current observation.

## Motivation

World models are often evaluated as predictors. That is useful, but insufficient. In partially observed settings, a useful model should also maintain a compact internal state that preserves hidden, causally relevant structure strongly enough to support imagination, counterfactual branching, and explicit state tracking. This repository is a controlled setting for testing that idea without hiding behind large-scale engineering complexity.

## Core Research Question

Can a small latent world model learn and preserve a hidden but behaviorally important variable across time well enough to:

- predict future observations,
- support multi-step open-loop rollout,
- branch into counterfactual futures under alternative action sequences,
- expose a latent state that can be interpreted and probed?

## Core Claim

A useful world model is not only a next-step predictor. It is a latent state estimator whose internal representation should preserve hidden, causally relevant structure across time well enough to support counterfactual reasoning, robust rollout, and interpretable belief-like state tracking.

## Environment

The environment is a deterministic `6 x 6` grid world with:

- one controllable agent,
- one movable crate,
- one switch,
- one beacon,
- one obstacle.

Observations are vector-valued and intentionally interpretable:

- agent position,
- crate position,
- switch position,
- beacon position,
- obstacle position,
- beacon-lit flag.

The critical hidden variable is `armed`, which is never exposed directly.

- stepping on the switch sets `armed = True`
- the beacon lights only if the agent later reaches it while `armed = True`
- two visibly similar states can therefore imply different futures depending on the unobserved trajectory history

This makes the task partially observed in a meaningful sense: the model must learn a belief-like latent state, not just a feed-forward transition rule from the current observation.

## Model

The model is a small PyTorch latent dynamics system with five components:

- an `encoder` that maps observations into latent features,
- a `posterior update` that combines the current observation with the latent prior,
- a `transition` model that predicts the next latent state from latent state plus action,
- a `decoder` that predicts the next observation,
- a `state head` that probes whether the latent carries `armed` and `beacon_lit`.

Architecturally, the project is intentionally modest. The point is not to win on scale. The point is to make the relation between hidden causal structure, latent memory, and downstream behavior easy to inspect.

## Training Objective

The training loss combines several roles:

- `next-observation reconstruction`: teaches local predictive accuracy
- `open-loop rollout loss`: forces the latent dynamics to remain useful when the model must imagine multiple future steps without corrective observations
- `latent consistency loss`: encourages the one-step prior to stay close to the posterior latent inferred after seeing the true next observation
- `armed` probe loss: pressures the latent to preserve hidden causal state
- `beacon_lit` probe loss: pressures the latent to track delayed visible consequences

Conceptually, the objective is trying to shape the latent into a compact belief state, not just a reconstruction bottleneck.

## Experiments

Each experiment is tied to a scientific question rather than a generic benchmark.

1. `One-step prediction`
   Tests whether the model can track local dynamics when it is allowed to filter through a sequence of observations and actions.

2. `Multi-step rollout`
   Tests whether the latent dynamics remain coherent when the model must imagine forward open-loop from a single starting observation.

3. `Counterfactual reasoning`
   Tests whether the same starting world state leads to different predicted futures under different action sequences, especially when only one branch satisfies the hidden switch precondition.

4. `Latent-space analysis`
   Tests whether the posterior latent organizes hidden state, visible outcome state, and trajectory phase in a coherent way rather than collapsing everything into a purely myopic predictor.

5. `Noise robustness`
   Tests whether the model’s predictive performance degrades gracefully when observations are corrupted with small Gaussian noise.

6. `Hidden-state probe behavior`
   Tests whether the latent prior explicitly preserves hidden causal structure strongly enough to recover `armed` and delayed visible structure such as `beacon_lit`.

## Results

Latest end-to-end run:

- one-step test MSE: `0.0985`
- noisy one-step test MSE: `0.0986`
- 10-step open-loop rollout MSE: `0.0986`
- hidden `armed` probe balanced accuracy: `0.762`
- `beacon_lit` probe balanced accuracy: `0.938`
- linear probe accuracy on posterior latents for `armed`: `0.829`
- counterfactual beacon ordering accuracy on held-out initial states: `0.667`
- calibrated counterfactual final-beacon accuracy: `0.771`
- calibrated counterfactual pair accuracy: `0.542`

Interpretation:

- The latent does carry meaningful hidden-state information. Both the explicit state head and the post-hoc linear probe recover `armed` well above chance.
- The latent space is structured enough that PCA shows separation by hidden armed state and a measurable trajectory-phase gradient.
- The model can solve the hand-crafted counterfactual demo exactly and often assigns higher beacon probability to the switch-first branch on held-out layouts.
- Raw counterfactual decisions at a fixed `0.5` threshold are conservative. The model distinguishes branches more reliably in score than in thresholded binary outcome, which is why calibrated counterfactual accuracy is materially stronger than raw accuracy.
- Rollout quality is good at short horizons and degrades gradually over longer open-loop prediction windows.

Representative artifacts:

- [results/figures/counterfactual_rollout.png](/Users/utente/pp1/hidden-dynamics/results/figures/counterfactual_rollout.png)
- [results/figures/rollout_example.png](/Users/utente/pp1/hidden-dynamics/results/figures/rollout_example.png)
- [results/figures/rollout_horizon_error.png](/Users/utente/pp1/hidden-dynamics/results/figures/rollout_horizon_error.png)
- [results/figures/latent_pca_armed.png](/Users/utente/pp1/hidden-dynamics/results/figures/latent_pca_armed.png)
- [results/figures/latent_pca_beacon_lit.png](/Users/utente/pp1/hidden-dynamics/results/figures/latent_pca_beacon_lit.png)
- [results/figures/latent_pca_timestep.png](/Users/utente/pp1/hidden-dynamics/results/figures/latent_pca_timestep.png)
- [results/figures/rollout_failure_cases.png](/Users/utente/pp1/hidden-dynamics/results/figures/rollout_failure_cases.png)
- [results/figures/counterfactual_failure_case.png](/Users/utente/pp1/hidden-dynamics/results/figures/counterfactual_failure_case.png)
- [results/checkpoints/evaluation_metrics.json](/Users/utente/pp1/hidden-dynamics/results/checkpoints/evaluation_metrics.json)
- [results/checkpoints/latent_analysis_summary.json](/Users/utente/pp1/hidden-dynamics/results/checkpoints/latent_analysis_summary.json)
- [results/checkpoints/failure_analysis.json](/Users/utente/pp1/hidden-dynamics/results/checkpoints/failure_analysis.json)

## Failure Modes

The current model fails in structured ways rather than at random:

- long-horizon rollout error grows steadily with prediction horizon
- filtered mid-trajectory counterfactuals are substantially harder than counterfactuals from clean initial states
- the most common counterfactual failure is missing the delayed switch prerequisite on the longer branch
- raw beacon probabilities are under-calibrated, so branch separation appears in score before it appears as a clean binary decision

These failure modes are explicit in the saved artifacts and JSON summaries rather than hidden behind aggregate averages.

## Limitations

- The environment is intentionally small and deterministic, so the project studies representation quality more than scale.
- Observations are vector-based, which improves interpretability but leaves out the perceptual challenge of image observations.
- The current counterfactual benchmark still exposes calibration weakness: the model often knows which branch is better before it is confident enough to cross a hard binary threshold.
- The strongest hidden-state evidence currently comes from probes and latent analysis; filtered counterfactual rollout from arbitrary mid-trajectory beliefs remains the weakest part of the prototype.

## Reproducibility

The repository is designed to run locally with minimal dependencies and deterministic seeds.

- datasets are generated reproducibly and stored in `data/`
- checkpoints and metrics are written to `results/checkpoints/`
- figures are written to `results/figures/`
- scripts resolve paths from the project root, so they can be launched from outside the repo directory

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py all
```

Useful entry points:

```bash
python main.py generate
python main.py train
python main.py evaluate
python main.py rollout
python main.py analyze
```

## Repository Layout

```text
hidden-dynamics/
  README.md
  PROJECT_SUMMARY.md
  config.py
  main.py
  generate_data.py
  train.py
  evaluate.py
  rollout.py
  analyze_latent.py
  models/
  world/
  utils/
  data/
  results/
```

## Future Directions

- compare latent dynamics with and without hidden-state probe supervision
- replace vector observations with rendered image observations while keeping the same hidden rule
- add stochastic transitions and evaluate whether uncertainty estimates become useful for belief tracking
- study counterfactual calibration explicitly, rather than only thresholded accuracy
- move from descriptive counterfactual rollout to planning in latent space

## Why This Prototype Matters

This project is small on purpose. Its value is not scale; it is clarity. It isolates a concrete question about latent state, hidden causal structure, and belief-like world modeling, then provides enough probes, counterfactuals, and failure analysis to make the answer inspectable rather than rhetorical.
