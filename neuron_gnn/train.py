"""Training, cross-validation and checkpointing.

Evaluation design
-----------------
* Outer loop: stratified k-fold (or grouped by NeuroMorpho archive, so no lab
  contributes neurons to both training and test folds).
* Inner split: a validation set carved out of each training fold picks the
  epoch. The test fold is scored once, at that epoch, and never used for
  model selection.
* Feature normalisation is fitted on training data only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

from .model import NeuronSAGE
from .swc import FEATURE_NAMES


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_columns(drop: list[str]) -> list[int]:
    unknown = set(drop) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature(s): {sorted(unknown)}; choose from {FEATURE_NAMES}")
    return [i for i, f in enumerate(FEATURE_NAMES) if f not in drop]


class Normaliser:
    """Z-score node features using statistics from training graphs only."""

    def __init__(self, cols: list[int]):
        self.cols = cols
        self.mean = self.std = None

    def fit(self, graphs):
        x = torch.cat([g.x[:, self.cols] for g in graphs])
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        return self

    def apply(self, graphs):
        out = []
        for g in graphs:
            g = copy.copy(g)
            g.x = (g.x[:, self.cols] - self.mean) / self.std
            out.append(g)
        return out

    def state(self) -> dict:
        return {"cols": self.cols, "mean": self.mean, "std": self.std}


def _predict(model, graphs, device):
    model.eval()
    y_true, y_pred, prob = [], [], []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=64):
            b = b.to(device)
            p = F.softmax(model(b.x, b.edge_index, b.batch), dim=1)
            prob += p[:, 1].cpu().tolist()
            y_pred += p.argmax(1).cpu().tolist()
            y_true += b.y.cpu().tolist()
    return np.array(y_true), np.array(y_pred), np.array(prob)


def fit(train, val, cfg, device):
    """Train on `train`; return the model state at the epoch with the best `val` accuracy."""
    model = NeuronSAGE(train[0].x.shape[1], hidden=cfg["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True)
    best_acc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            F.cross_entropy(model(b.x, b.edge_index, b.batch), b.y).backward()
            opt.step()
        y, p, _ = _predict(model, val, device)
        acc = float((y == p).mean())
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_epoch, best_acc


def cross_validate(graphs, groups, cfg, out_dir: Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    labels = np.array([int(g.y) for g in graphs])
    cols = feature_columns(cfg["drop_features"])

    if cfg["group_by_archive"]:
        splitter = StratifiedGroupKFold(n_splits=cfg["folds"], shuffle=True, random_state=cfg["seed"])
        splits = splitter.split(np.zeros(len(graphs)), labels, groups)
    else:
        splitter = StratifiedKFold(n_splits=cfg["folds"], shuffle=True, random_state=cfg["seed"])
        splits = splitter.split(np.zeros(len(graphs)), labels)

    folds, all_true, all_pred = [], [], []
    for k, (tr_idx, te_idx) in enumerate(splits, 1):
        tr_idx, va_idx = train_test_split(
            tr_idx, test_size=cfg["val_fraction"], stratify=labels[tr_idx], random_state=cfg["seed"]
        )
        norm = Normaliser(cols).fit([graphs[i] for i in tr_idx])
        tr, va, te = (norm.apply([graphs[i] for i in idx]) for idx in (tr_idx, va_idx, te_idx))
        model, epoch, val_acc = fit(tr, va, cfg, device)
        y, p, _ = _predict(model, te, device)
        res = {
            "fold": k, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "chosen_epoch": epoch, "val_acc": val_acc,
            "test_acc": float((y == p).mean()),
            "test_balanced_acc": float(balanced_accuracy_score(y, p)),
            "test_f1_interneuron": float(f1_score(y, p, zero_division=0)),
        }
        folds.append(res)
        all_true += y.tolist()
        all_pred += p.tolist()
        print(f"fold {k}: test acc {res['test_acc']:.3f} (balanced {res['test_balanced_acc']:.3f}, "
              f"epoch {epoch} chosen on validation)", flush=True)

    accs = [f["test_acc"] for f in folds]
    bals = [f["test_balanced_acc"] for f in folds]
    summary = {
        "config": cfg,
        "features_used": [FEATURE_NAMES[i] for i in cols],
        "n_graphs": len(graphs),
        "class_counts": {"pyramidal": int((labels == 0).sum()), "interneuron": int((labels == 1).sum())},
        "test_acc_mean": float(np.mean(accs)), "test_acc_sd": float(np.std(accs)),
        "balanced_acc_mean": float(np.mean(bals)), "balanced_acc_sd": float(np.std(bals)),
        "pooled_confusion_matrix": confusion_matrix(all_true, all_pred, labels=[0, 1]).tolist(),
        "folds": folds,
    }
    print(f"\n{cfg['folds']}-fold CV test accuracy: {summary['test_acc_mean']:.3f} ± {summary['test_acc_sd']:.3f} "
          f"| balanced: {summary['balanced_acc_mean']:.3f} ± {summary['balanced_acc_sd']:.3f}")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cv_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def train_final(graphs, cfg, ckpt_path: Path) -> None:
    """Train on all data (with a validation split for epoch choice) and save a checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    labels = np.array([int(g.y) for g in graphs])
    idx = np.arange(len(graphs))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg["val_fraction"], stratify=labels,
                                      random_state=cfg["seed"])
    norm = Normaliser(feature_columns(cfg["drop_features"])).fit([graphs[i] for i in tr_idx])
    tr, va = norm.apply([graphs[i] for i in tr_idx]), norm.apply([graphs[i] for i in va_idx])
    model, epoch, val_acc = fit(tr, va, cfg, device)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "in_dim": tr[0].x.shape[1],
                "hidden": cfg["hidden"], "norm": norm.state(), "config": cfg}, ckpt_path)
    print(f"saved {ckpt_path} (epoch {epoch}, validation accuracy {val_acc:.3f})")
