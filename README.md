# Classifying cortical neurons from morphology with a graph neural network

Can a neuron's 3D shape alone tell you whether it is a **pyramidal cell** or an **interneuron**?
This project treats each reconstructed neuron as a graph (sample points as nodes, neurites as edges)
and trains a GraphSAGE network to classify it.

Data come from [NeuroMorpho.Org](https://neuromorpho.org) (mouse neocortex, frontal regions).

## Method

| Step | What happens |
|---|---|
| Selection | Mouse neocortex neurons whose brain region contains "frontal"; balanced random sample of up to 200 pyramidal cells and 200 interneurons (seed 42). Cell types matching both classes are dropped. |
| Graph | One node per SWC sample point, undirected edge per parent–child link. Neurons with more than 10,000 nodes are excluded. |
| Node features | Centred x, y, z · radius · SWC structure type · degree · hop distance from root (0–1) · branch-point flag |
| Model | 3 × GraphSAGE (64 units, ReLU) → global mean pool → dropout 0.3 → linear (2 classes). Adam, lr 5e-3, weight decay 5e-4, 60 epochs. |
| Evaluation | Stratified 5-fold CV. Within each training fold, 15% is held out as validation to choose the epoch; the test fold is scored once. Feature normalisation is fitted on training data only. |

## Results

**First run (Google Colab, T4 GPU, June 2026), original notebook:** 400 SWC files downloaded, 346 graphs
after the node cap (189 pyramidal, 157 interneuron).

| Fold | 1 | 2 | 3 | 4 | 5 | Mean ± SD |
|---|---|---|---|---|---|---|
| Accuracy | 0.971 | 0.986 | 0.957 | 0.957 | 0.971 | **0.968 ± 0.011** |

> ⚠️ **Treat this number as optimistic.** In that run, each fold reported the *best test accuracy over all
> 60 epochs*, so the test fold was also used to choose the epoch, and features were normalised using all
> neurons including the test fold. The current code fixes both (see Evaluation above).
> Re-running with `python -m neuron_gnn cv` gives the unbiased estimate; that result has not been recorded here yet.

## Known confounds (worth testing, not yet tested)

1. **SWC structure type may leak the label.** SWC type 4 marks *apical dendrite*, a defining feature of
   pyramidal cells, so the model could learn "has type-4 nodes" rather than shape. Ablation:
   `--drop-features swc_type`.
2. **Lab / archive effects.** Pyramidal cells and interneurons may come from different labs with different
   reconstruction protocols, so the model could be classifying the lab. Control:
   `--group-by-archive` keeps every archive inside one fold.
3. **Selection by metadata label.** Labels are NeuroMorpho's `cell_type` strings, not independently verified.

If accuracy stays high with both controls on, that is much stronger evidence that morphology carries the class signal.

## Usage

```bash
pip install -r requirements.txt

python -m neuron_gnn fetch                       # metadata, subset.csv and SWC files into data/
python -m neuron_gnn cv                          # 5-fold CV → results/cv_results.json
python -m neuron_gnn cv --drop-features swc_type --group-by-archive --out results/controls
python -m neuron_gnn train                       # final model → results/neuron_model.pt
python -m neuron_gnn predict data/swc/*.swc --threshold 0.75
```

A GPU is optional; on CPU the 5-fold CV takes several minutes.

## Repository layout

```
neuron_gnn/
  data.py    NeuroMorpho metadata, subset selection, SWC download
  swc.py     SWC parser and node features
  model.py   GraphSAGE classifier
  train.py   training, cross-validation, checkpoint
  cli.py     fetch / cv / train / predict
tests/       offline tests on synthetic neurons (pytest)
```

## References

- Ascoli GA, Donohue DE, Halavi M (2007). NeuroMorpho.Org: a central resource for neuronal morphologies. *J Neurosci* 27:9247–9251.
- Hamilton WL, Ying R, Leskovec J (2017). Inductive representation learning on large graphs. *NeurIPS* 30.
- Fey M, Lenssen JE (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.

Neuron data are not redistributed here; cite NeuroMorpho.Org and the original contributing labs if you use them.

## About

Daniel Davies, BSc Neuroscience, University of Sussex. Built as a personal project in June 2026;
code reorganised and evaluation corrected in November 2026 with help from an AI assistant (Claude).
