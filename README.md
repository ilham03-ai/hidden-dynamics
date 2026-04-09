# world-model-lab-v2

`world-model-lab-v2` is a compact research prototype for latent world modeling in a partially observed synthetic environment. The project learns a latent state from observation-action sequences, predicts future states, compares counterfactual futures, analyzes the learned latent geometry, and uses a learned uncertainty signal to drive targeted exploration.

## Motivation

A useful world model should do more than interpolate one-step transitions. It should carry hidden state forward through time, support imagination under alternative action sequences, organize experience coherently in latent space, and know when it is uncertain. This repository is a small but complete experimental setting for studying those properties end to end.

## Central Research Question

Can a lightweight latent dynamics model learn a meaningful internal representation of a synthetic world that is rich enough to:

- predict future outcomes from sequential interaction data,
- imagine alternative futures under different action sequences,
- separate hidden structure in latent space,
- estimate its own prediction uncertainty well enough to guide exploratory behavior?

## Synthetic World

The environment is a `7 x 7` grid with:

- one controllable agent,
- two control pads: `A` and `B`,
- one beacon terminal,
- two static obstacles,
- a hidden episode-level world mode.

The action space is:

- `up`
- `down`
- `left`
- `right`
- `interact`

Observations are vector-based and interpretable. Each observation contains:

- agent position,
- pad `A` position,
- pad `B` position,
- terminal position,
- two obstacle positions,
- a visible interaction signal,
- the visible beacon-lit flag.

## Hidden Dynamics

Each episode samples a hidden mode, `alpha` or `beta`.

- In `alpha`, pad `A` is the correct pad.
- In `beta`, pad `B` is the correct pad.

Interacting with the correct pad activates a hidden charge timer for a few future steps. That charge is never directly observed. The beacon terminal lights only if the agent later reaches the terminal and interacts while the hidden charge is still active.

This means two visually identical observations can imply different futures depending on the unobserved interaction history. Temporal experience is necessary to infer the true latent world state.

## Model

The world model is a small PyTorch latent dynamics system:

- `encoder`: maps observation vectors into latent states,
- `posterior update`: fuses the current observation with the model prior,
- `transition model`: predicts the next latent state from latent state plus action,
- `decoder`: reconstructs the next observation,
- `uncertainty head`: predicts expected transition error from the latent state.

The model is intentionally simple: fully connected networks throughout, with a modular design that can later be swapped to image encoders without changing the overall interface.

## Training Objective

Training combines four losses:

- one-step observation prediction loss,
- open-loop rollout loss from the initial observation,
- latent consistency loss between prior and posterior latent states,
- uncertainty regression loss, where the model predicts its own future squared prediction error.

This mix matters. One-step fitting alone is not enough for long-horizon imagination, and uncertainty is treated as a learned predictive diagnostic rather than a decorative extra head.

## Experiments

The repository implements:

1. One-step prediction
   Predict the next observation from the current observation and action.

2. Multi-step rollout
   Predict several future observations open-loop from a single initial observation.

3. Counterfactual reasoning
   Compare predicted futures from the same initial world under different action sequences.

4. Latent-space analysis
   Project latent states with PCA and t-SNE and color them by hidden mode or hidden charge state.

5. Generalization
   Evaluate on held-out random layouts and hidden modes.

6. Noise robustness
   Compare clean and noisy evaluation using saved noisy variants of the same datasets.

7. Uncertainty-aware exploration
   Compare a random policy to a policy that chooses short action sequences with maximal predicted uncertainty.

## Repository Structure

```text
world-model-lab-v2/
  README.md
  requirements.txt
  config.py
  main.py
  generate_data.py
  train.py
  evaluate.py
  rollout.py
  analyze_latent.py
  explore.py
  world/
    environment.py
    rules.py
    rendering.py
  models/
    encoder.py
    transition.py
    decoder.py
    uncertainty_head.py
    world_model.py
  utils/
    seed.py
    metrics.py
    plotting.py
    data_utils.py
  data/
  results/
  notebooks/
```

## Example Results

From the current verified run:

- clean one-step MSE: `0.0419`
- noisy one-step MSE: `0.0423`
- 10-step rollout MSE: `0.0416`
- uncertainty/error correlation: `0.2409`

Counterfactual search found a held-out scenario where:

- the true correct path activates the beacon,
- the true wrong path does not,
- the model assigns a much higher final beacon score to the correct path (`0.3503`) than to the wrong path (`0.0779`),
- the counterfactual gap is `0.2724`.

In the uncertainty exploration experiment:

- random policy mean informative interactions: `0.10`
- uncertainty-seeking policy mean informative interactions: `0.35`
- uncertainty-seeking policy also experiences higher predicted uncertainty and higher actual model error, which is the intended probing behavior.

Saved outputs:

- loss curves: [results/figures/loss_curves.png](/Users/utente/pp1/world-model-lab-v2/results/figures/loss_curves.png)
- rollout example: [results/figures/rollout_example.png](/Users/utente/pp1/world-model-lab-v2/results/figures/rollout_example.png)
- counterfactual rollout: [results/figures/counterfactual_rollout.png](/Users/utente/pp1/world-model-lab-v2/results/figures/counterfactual_rollout.png)
- uncertainty over time: [results/figures/uncertainty_over_time.png](/Users/utente/pp1/world-model-lab-v2/results/figures/uncertainty_over_time.png)
- exploration comparison: [results/figures/exploration_comparison.png](/Users/utente/pp1/world-model-lab-v2/results/figures/exploration_comparison.png)
- latent PCA by mode: [results/figures/latent_pca_mode.png](/Users/utente/pp1/world-model-lab-v2/results/figures/latent_pca_mode.png)
- latent t-SNE by mode: [results/figures/latent_tsne_mode.png](/Users/utente/pp1/world-model-lab-v2/results/figures/latent_tsne_mode.png)
- evaluation metrics: [results/checkpoints/evaluation_metrics.json](/Users/utente/pp1/world-model-lab-v2/results/checkpoints/evaluation_metrics.json)

## Running The Project

Create a local virtual environment, install the minimal dependencies, and run the full pipeline:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py all
```

Individual stages are also available:

```bash
python generate_data.py
python train.py
python evaluate.py
python rollout.py
python analyze_latent.py
python explore.py
```

## Why The Project Is Interesting

This is not just a toy next-step predictor. The world contains a hidden interaction-dependent state that matters causally but is not visible in the observation. The model therefore has to build an internal belief-like representation, not just a reactive mapping from current observation to next observation.

The latent-space plots and counterfactual rollouts are central to the project: they are the main evidence that the model is learning something structural about the world rather than only fitting supervised targets.

## Limitations

- The observation space is still vector-based; an image-based version would be a stronger visual test.
- The uncertainty head predicts expected model error, not full Bayesian epistemic uncertainty.
- The counterfactual decoder remains conservative: it clearly separates correct and wrong futures, but it underestimates the terminal beacon probability in some successful cases.
- The exploration policy maximizes predicted uncertainty, which is informative but not identical to optimal information gain.

## Future Directions

- Replace vector observations with rendered image observations and a CNN encoder.
- Add stochastic dynamics or multiple hidden modes with different time constants.
- Compare uncertainty-seeking exploration with uncertainty-reduction or disagreement-based exploration.
- Add planning in latent space for goal-directed control.
- Test whether latent state linear probes recover hidden mode and hidden charge more cleanly across longer horizons.
- Compare teacher-forced-only training against explicit rollout training.

