"""Offline tests: synthetic SWC files, no network access needed."""
import numpy as np
import pandas as pd
import pytest

from neuron_gnn.cli import main
from neuron_gnn.swc import FEATURE_NAMES, node_features, parse_swc, swc_to_graph


def write_swc(path, n_branches, length, apical, rng):
    """A soma with `n_branches` straight dendrites; optionally one apical (type 4) dendrite."""
    lines = ["# synthetic", "1 1 0 0 0 5 -1"]
    nid = 1
    for b in range(n_branches + int(apical)):
        kind = 4 if (apical and b == n_branches) else 3
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        parent = 1
        for step in range(1, length + 1):
            nid += 1
            x, y, z = direction * step * 2
            lines.append(f"{nid} {kind} {x:.2f} {y:.2f} {z:.2f} 0.5 {parent}")
            parent = nid
    path.write_text("\n".join(lines) + "\n")


def test_parser_handles_parent_listed_after_child(tmp_path):
    f = tmp_path / "out_of_order.swc"
    f.write_text("1 1 0 0 0 1 -1\n2 3 1 0 0 1 3\n3 3 2 0 0 1 1\nbad line\n")
    swc = parse_swc(f)
    assert len(swc.xyz) == 3
    assert swc.edges.shape == (2, 4)      # two links, both directions


def test_features_shape_and_ranges(tmp_path):
    f = tmp_path / "n.swc"
    write_swc(f, n_branches=3, length=4, apical=False, rng=np.random.default_rng(0))
    x = node_features(parse_swc(f))
    assert x.shape == (13, len(FEATURE_NAMES))
    dist = x[:, FEATURE_NAMES.index("path_dist")]
    assert dist.min() == 0 and dist.max() <= 1
    assert x[0, FEATURE_NAMES.index("is_branch_point")] == 1   # soma has 3 children


def test_node_count_filter(tmp_path):
    f = tmp_path / "tiny.swc"
    f.write_text("1 1 0 0 0 1 -1\n2 3 1 0 0 1 1\n")
    assert swc_to_graph(f, 0) is None


@pytest.fixture
def toy_dataset(tmp_path):
    rng = np.random.default_rng(1)
    (tmp_path / "swc").mkdir()
    rows = []
    for i in range(40):
        label = i % 2
        name = f"cell{i}"
        write_swc(tmp_path / "swc" / f"{name}.CNG.swc", n_branches=2 if label == 0 else 6,
                  length=8 if label == 0 else 3, apical=(label == 0), rng=rng)
        rows.append({"neuron_name": name, "archive": f"lab{i % 4}", "label": label})
    pd.DataFrame(rows).to_csv(tmp_path / "subset.csv", index=False)
    return tmp_path


def test_cv_train_predict_end_to_end(toy_dataset, capsys):
    common = ["--data-dir", str(toy_dataset)]
    opts = ["--folds", "2", "--epochs", "3", "--out", str(toy_dataset / "results")]
    main(common + ["cv"] + opts)
    assert (toy_dataset / "results" / "cv_results.json").exists()
    main(common + ["cv"] + opts + ["--group-by-archive", "--drop-features", "swc_type"])
    main(common + ["train"] + opts)
    main(common + ["predict", str(toy_dataset / "swc" / "cell0.CNG.swc"),
                   "--checkpoint", str(toy_dataset / "results" / "neuron_model.pt")])
    assert "cell0.CNG.swc" in capsys.readouterr().out


def test_select_subset_is_balanced_and_drops_ambiguous():
    from neuron_gnn.data import select_subset
    meta = pd.DataFrame({
        "neuron_name": [f"n{i}" for i in range(9)],
        "archive": ["a"] * 9,
        "brain_region": ["neocortex, frontal"] * 8 + ["neocortex, occipital"],
        "cell_type": ["pyramidal"] * 4 + ["interneuron, basket"] * 3 + ["pyramidal, interneuron", "pyramidal"],
    })
    sel = select_subset(meta, "frontal", n_per_class=3, seed=0)
    assert sel["label"].tolist().count(0) == 3 and sel["label"].tolist().count(1) == 3
    assert "n7" not in set(sel["neuron_name"]) and "n8" not in set(sel["neuron_name"])
