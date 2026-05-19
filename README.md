# Hidden Dynamics

`Hidden Dynamics` is a small PyTorch project about world models and memory.

I wanted a setting where I could tell, pretty clearly, whether a model was actually carrying hidden information forward in time or just doing decent one-step prediction. So I built a tiny grid world where the observation does not tell you everything you need to know about the future.

## The setup

The world is a deterministic `6 x 6` grid with:

- one agent
- one crate
- one switch
- one beacon
- one obstacle

The hidden variable is `armed`.

- stepping on the switch sets `armed = True`
- the beacon lights only if the agent reaches it after the switch has already been touched
- `armed` never appears directly in the observation

That is the whole trick. Two states can look identical and still imply different futures because of what happened earlier in the episode.

I kept the observations vector-based on purpose. For this project I cared more about inspecting the learned state than about adding a perception problem on top.

## Model

The model is straightforward:

- observation encoder
- posterior update
- action-conditioned latent transition
- observation decoder
- probe head for `armed` and `beacon_lit`

The architecture is not the point here. The point is to have something small enough that rollout behavior, latent probes, and counterfactual examples are easy to inspect.

## What turned out to matter

Two things mattered more than I expected:

1. The rollout loss.
   With only teacher-forced next-step prediction, the model looked better on paper than it did in open-loop rollouts.

2. The hidden-state probes.
   Low reconstruction error by itself did not tell me whether the latent had actually learned the switch dependency. The probe head and the post-hoc linear probe made that much easier to check.

## What I evaluated

I mainly cared about:

- one-step prediction
- open-loop rollout
- counterfactual branching from the same start state
- whether `armed` can be recovered from the latent
- what breaks when observation noise is added

I also saved failure cases instead of only the nicest rollouts.

## Current results

Latest run:

- one-step MSE: `0.0985`
- noisy one-step MSE: `0.0986`
- 10-step rollout MSE: `0.0986`
- `armed` probe balanced accuracy: `0.762`
- `beacon_lit` probe balanced accuracy: `0.938`
- linear probe accuracy on posterior latents for `armed`: `0.829`
- counterfactual branch ordering accuracy: `0.667`
- calibrated counterfactual final-beacon accuracy: `0.771`

My interpretation of that:

- the model is learning some real hidden-state structure
- the latent is useful enough for short-horizon rollout and simple counterfactual comparison
- the model is still conservative about beacon activation, especially in harder counterfactual cases
- score separation appears earlier than clean binary decisions, which is why calibration still matters here

## Where it is still weak

- filtered mid-trajectory counterfactuals are harder than clean rollouts from the start of an episode
- long-horizon rollout drift is still there
- the latent is informative, but not especially disentangled
- this is still a small vector-state environment, not an image-based benchmark

## Useful files

- [`world/environment.py`](world/environment.py)
- [`models/world_model.py`](models/world_model.py)
- [`train.py`](train.py)
- [`evaluate.py`](evaluate.py)
- [`rollout.py`](rollout.py)
- [`analyze_latent.py`](analyze_latent.py)
- [`DEV_NOTES.md`](DEV_NOTES.md)

If you want to inspect outputs first, these are the most useful ones:

- [`results/figures/counterfactual_rollout.png`](results/figures/counterfactual_rollout.png)
- [`results/figures/rollout_example.png`](results/figures/rollout_example.png)
- [`results/figures/rollout_horizon_error.png`](results/figures/rollout_horizon_error.png)
- [`results/figures/counterfactual_failure_case.png`](results/figures/counterfactual_failure_case.png)
- [`results/checkpoints/evaluation_metrics.json`](results/checkpoints/evaluation_metrics.json)
- [`results/checkpoints/failure_analysis.json`](results/checkpoints/failure_analysis.json)

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py all
```

Or run stages separately:

```bash
python main.py generate
python main.py train
python main.py evaluate
python main.py rollout
python main.py analyze
```

## Layout

```text
hidden-dynamics/
  README.md
  PROJECT_SUMMARY.md
  DEV_NOTES.md
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

## If I kept working on it

- compare training with and without the hidden-state probe losses
- move to image observations while keeping the same hidden rule
- use a proper OOD split instead of only new random layouts
- look at calibration directly rather than only thresholded accuracy
- try a stochastic latent model instead of a deterministic one
