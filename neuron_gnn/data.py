"""Fetch neuron metadata and SWC files from NeuroMorpho.Org.

NeuroMorpho.Org: Ascoli, Donohue & Halavi (2007), J Neurosci 27:9247-9251.
Only metadata is requested from the API; SWC files are downloaded for the
selected subset. Downloaded data are not redistributed in this repository.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://neuromorpho.org/api/neuron/select"
SWC_URL = "https://neuromorpho.org/dableFiles/{archive}/CNG%20version/{name}.CNG.swc"
HEADERS = {"User-Agent": "Mozilla/5.0 (neuron-morphology-gnn)"}

CLASSES = ["pyramidal", "interneuron"]   # label 0, label 1


def fetch_metadata(dest: Path, query: str = "species:mouse,brain_region:neocortex") -> pd.DataFrame:
    """Download metadata for every neuron matching `query` (cached to `dest`)."""
    if dest.exists():
        return pd.read_csv(dest, low_memory=False)
    import requests

    rows, page = [], 0
    while True:
        r = requests.get(API, params={"q": query, "size": 500, "page": page}, timeout=60)
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("_embedded", {}).get("neuronResources", [])
        if not batch:
            break
        for n in batch:
            rows.append({
                "neuron_name": n.get("neuron_name"),
                "archive": n.get("archive"),
                "brain_region": _join(n.get("brain_region")),
                "cell_type": _join(n.get("cell_type")),
            })
        page += 1
        total = payload.get("page", {}).get("totalPages", page)
        print(f"  metadata page {page}/{total}", flush=True)
        if page >= total:
            break
    df = pd.DataFrame(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


def select_subset(
    meta: pd.DataFrame, region: str = "frontal", n_per_class: int = 200, seed: int = 42
) -> pd.DataFrame:
    """Balanced sample of pyramidal cells and interneurons from one region.

    Rows whose cell_type mentions both classes are dropped as ambiguous.
    """
    m = meta[meta["brain_region"].str.contains(region, case=False, na=False)]
    is_pyr = m["cell_type"].str.contains("pyramidal", case=False, na=False)
    is_int = m["cell_type"].str.contains("interneuron", case=False, na=False)
    pyr = m[is_pyr & ~is_int]
    inter = m[is_int & ~is_pyr]
    pyr = pyr.sample(min(n_per_class, len(pyr)), random_state=seed)
    inter = inter.sample(min(n_per_class, len(inter)), random_state=seed)
    sel = pd.concat([pyr, inter])[["neuron_name", "archive", "brain_region", "cell_type"]].copy()
    sel["label"] = [0] * len(pyr) + [1] * len(inter)
    return sel.reset_index(drop=True)


def download_swc(subset: pd.DataFrame, swc_dir: Path) -> None:
    """Download the standardised (CNG) SWC file for each neuron in `subset`."""
    swc_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for _, row in subset.iterrows():
        dest = swc_dir / f"{row.neuron_name}.CNG.swc"
        if dest.exists():
            ok += 1
            continue
        url = SWC_URL.format(archive=str(row.archive).lower(), name=row.neuron_name)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
                dest.write_bytes(r.read())
            ok += 1
        except Exception:
            fail += 1
        if (ok + fail) % 50 == 0:
            time.sleep(0.5)   # be polite to the server
    print(f"SWC download: {ok} ok, {fail} failed")


def _join(v) -> str:
    return ", ".join(v) if isinstance(v, list) else str(v or "")
