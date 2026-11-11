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

> ⚠️ **That number is optimistic.** In that run each fold reported the *best test accuracy over all
> 60 epochs*, so the test fold also chose the epoch, and features were normalised using all neurons
> including the test fold. Both are fixed in the current code; corrected results are below.

**Corrected evaluation with confound controls (Google Colab, 11 November 2026):** 340 graphs
(183 pyramidal, 157 interneuron) from 49 NeuroMorpho archives; 5-fold CV, mean ± SD across folds.

| Run | Command flags | Accuracy | Balanced accuracy |
|---|---|---|---|
| Corrected baseline | *(none)* | 0.929 ± 0.045 | 0.928 ± 0.046 |
| SWC structure type removed | `--drop-features swc_type` | 0.912 ± 0.025 | 0.913 ± 0.024 |
| Archives (labs) kept within one fold | `--group-by-archive` | 0.854 ± 0.113 | 0.828 ± 0.139 |
| Both controls | `--drop-features swc_type --group-by-archive` | 0.739 ± 0.212 | 0.785 ± 0.145 |

### Interpretation

- **Lab identity is the main confound.** Testing on labs the model never saw drops balanced accuracy from
  0.93 to 0.83, and fold-to-fold spread triples. Within the same labs, part of what the model learns is
  probably each lab's reconstruction style rather than cell type.
- **The apical-dendrite label helps mainly on unseen labs.** Removing SWC type barely matters when labs are
  shared across folds (0.913 vs 0.928), but costs a further ~0.04 balanced accuracy on unseen labs.
- **The fairest estimate of "shape alone, new lab" is ≈ 0.79 balanced accuracy**, clearly above chance
  (0.5) but with wide uncertainty.
- **Caveat:** most archives contribute only one cell type, so some lab-grouped test folds contain a
  single class; per-fold numbers there are noisy, and balanced accuracy is the better summary.

## Remaining limitations

1. **Labs and classes are entangled.** A cleaner test needs archives that reconstructed both cell types.
2. **Labels come from metadata.** Classes are NeuroMorpho's `cell_type` strings, not independently verified.
3. **Small sample.** 340 neurons and 5 folds give wide confidence intervals; repeated CV with several seeds
   would tighten the estimates.

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
