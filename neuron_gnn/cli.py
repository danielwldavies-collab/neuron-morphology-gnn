"""Command-line entry point.

    python -m neuron_gnn fetch     # metadata + balanced subset + SWC files
    python -m neuron_gnn cv        # cross-validated accuracy
    python -m neuron_gnn train     # train on everything, save checkpoint
    python -m neuron_gnn predict path/to/*.swc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .data import CLASSES, download_swc, fetch_metadata, select_subset
from .model import NeuronSAGE
from .swc import swc_to_graph


def load_graphs(data_dir: Path, max_nodes: int):
    subset = pd.read_csv(data_dir / "subset.csv")
    graphs, groups = [], []
    for row in subset.itertuples():
        path = data_dir / "swc" / f"{row.neuron_name}.CNG.swc"
        if not path.exists():
            continue
        g = swc_to_graph(path, int(row.label), max_nodes=max_nodes)
        if g is not None:
            graphs.append(g)
            groups.append(str(row.archive))
    n_pyr = sum(int(g.y) == 0 for g in graphs)
    print(f"graphs: {len(graphs)} | pyramidal {n_pyr} | interneuron {len(graphs) - n_pyr} "
          f"| archives {len(set(groups))}")
    return graphs, groups


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neuron_gnn")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download metadata, choose subset, download SWC files")
    f.add_argument("--region", default="frontal")
    f.add_argument("--n-per-class", type=int, default=200)
    f.add_argument("--seed", type=int, default=42)

    for name in ("cv", "train"):
        p = sub.add_parser(name)
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--lr", type=float, default=5e-3)
        p.add_argument("--weight-decay", type=float, default=5e-4)
        p.add_argument("--hidden", type=int, default=64)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--val-fraction", type=float, default=0.15)
        p.add_argument("--max-nodes", type=int, default=10_000)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--drop-features", nargs="*", default=[],
                       help="e.g. --drop-features swc_type  (see FEATURE_NAMES)")
        p.add_argument("--group-by-archive", action="store_true",
                       help="keep each NeuroMorpho archive (lab) within a single fold")
        p.add_argument("--out", type=Path, default=Path("results"))

    pr = sub.add_parser("predict")
    pr.add_argument("files", nargs="+", type=Path)
    pr.add_argument("--checkpoint", type=Path, default=Path("results/neuron_model.pt"))
    pr.add_argument("--threshold", type=float, default=0.75,
                    help="report 'uncertain' below this confidence")

    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        meta = fetch_metadata(a.data_dir / "neuron_metadata.csv")
        subset = select_subset(meta, a.region, a.n_per_class, a.seed)
        subset.to_csv(a.data_dir / "subset.csv", index=False)
        print(subset["label"].map(dict(enumerate(CLASSES))).value_counts().to_string())
        download_swc(subset, a.data_dir / "swc")
        return

    if a.cmd in ("cv", "train"):
        from .train import cross_validate, train_final
        cfg = {k: getattr(a, k) for k in ("folds", "epochs", "lr", "weight_decay", "hidden", "batch_size",
                                         "val_fraction", "max_nodes", "seed", "drop_features",
                                         "group_by_archive")}
        graphs, groups = load_graphs(a.data_dir, a.max_nodes)
        if a.cmd == "cv":
            cross_validate(graphs, groups, cfg, a.out)
        else:
            train_final(graphs, cfg, a.out / "neuron_model.pt")
        return

    if a.cmd == "predict":
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model = NeuronSAGE(ck["in_dim"], hidden=ck["hidden"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        norm = ck["norm"]
        for path in a.files:
            g = swc_to_graph(path, max_nodes=ck["config"]["max_nodes"])
            if g is None:
                print(f"{path.name:<40} skipped (node count out of range)")
                continue
            x = (g.x[:, norm["cols"]] - norm["mean"]) / norm["std"]
            with torch.no_grad():
                prob = F.softmax(model(x, g.edge_index, torch.zeros(len(x), dtype=torch.long)), dim=1)[0]
            conf, idx = float(prob.max()), int(prob.argmax())
            label = CLASSES[idx] if conf >= a.threshold else "uncertain"
            print(f"{path.name:<40} {label:<12} {conf:.3f}")


if __name__ == "__main__":
    main()
