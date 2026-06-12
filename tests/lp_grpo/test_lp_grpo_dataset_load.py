"""Test _load_lp_grpo_p0_from_dataset by creating a fake parquet and exercising
the loading logic in isolation (without instantiating the full RayPPOTrainer).
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


# Replicate the loading logic from RayPPOTrainer._load_lp_grpo_p0_from_dataset
# without needing the full trainer.
def load_p0(dataset, eps_p=0.05):
    """Mirror of RayPPOTrainer._load_lp_grpo_p0_from_dataset."""
    df = dataset
    if "p_zero" not in df.column_names:
        return {"p_0_map": {}, "n_loaded": 0, "n_total": len(df), "missing": True}

    p_zero_list = list(df["p_zero"])
    extra_list = list(df["extra_info"]) if "extra_info" in df.column_names else [None] * len(df)

    n_total = len(p_zero_list)
    n_loaded = 0
    seen_indices = set()
    n_collisions = 0
    p_0_map = {}
    for p0, extra in zip(p_zero_list, extra_list):
        if p0 is None:
            continue
        idx = (extra or {}).get("index", 0) if isinstance(extra, dict) else 0
        if idx in seen_indices:
            n_collisions += 1
        else:
            seen_indices.add(idx)
        p_0_map[idx] = min(max(float(p0), eps_p), 1.0 - eps_p)
        n_loaded += 1
    return {"p_0_map": p_0_map, "n_loaded": n_loaded, "n_total": n_total,
            "n_collisions": n_collisions, "n_unique": len(seen_indices), "missing": False}


def test_synthetic_parquet():
    """Create a 10-prompt fake parquet, load p_zero, verify."""
    rows = []
    for i in range(10):
        rows.append({
            "prompt": [{"role": "user", "content": f"Question {i}"}],
            "p_zero": float(i) / 10.0,            # 0.0, 0.1, ..., 0.9
            "extra_info": {"index": i, "source": "fake"},
        })
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fake.parquet")
        df.to_parquet(path)
        ds = Dataset.from_parquet(path)
        result = load_p0(ds)

    assert not result["missing"], "p_zero column should exist"
    assert result["n_loaded"] == 10, f"expected 10 loaded, got {result['n_loaded']}"
    assert result["n_unique"] == 10, f"expected 10 unique, got {result['n_unique']}"
    assert result["n_collisions"] == 0, f"expected 0 collisions, got {result['n_collisions']}"
    # Verify clipping: 0.0 -> 0.05, 0.9 -> 0.9 (within bounds)
    assert result["p_0_map"][0] == 0.05, f"index 0: {result['p_0_map'][0]}"
    assert abs(result["p_0_map"][5] - 0.5) < 1e-6
    assert result["p_0_map"][9] == 0.9
    print(f"[PASS] test_synthetic_parquet: loaded {result['n_loaded']} prompts, "
          f"p_0 values: {sorted([(k, round(v, 3)) for k, v in result['p_0_map'].items()])}")


def test_collision_detection():
    """Two prompts share same index -> collision detected."""
    rows = [
        {"prompt": "A", "p_zero": 0.3, "extra_info": {"index": 1}},
        {"prompt": "B", "p_zero": 0.7, "extra_info": {"index": 1}},  # COLLISION
        {"prompt": "C", "p_zero": 0.5, "extra_info": {"index": 2}},
    ]
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "collide.parquet")
        df.to_parquet(path)
        ds = Dataset.from_parquet(path)
        result = load_p0(ds)
    assert result["n_collisions"] == 1, f"expected 1 collision, got {result['n_collisions']}"
    assert result["n_unique"] == 2
    # The 2nd entry should overwrite the 1st (last write wins)
    assert abs(result["p_0_map"][1] - 0.7) < 1e-6, "second p_0 for index 1 should win"
    print(f"[PASS] test_collision_detection (collisions={result['n_collisions']})")


def test_missing_column():
    """Dataset without p_zero column -> empty p_0_map, missing=True."""
    rows = [{"prompt": f"Q{i}", "extra_info": {"index": i}} for i in range(3)]
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "no_p0.parquet")
        df.to_parquet(path)
        ds = Dataset.from_parquet(path)
        result = load_p0(ds)
    assert result["missing"], "should detect missing column"
    print(f"[PASS] test_missing_column")


def test_none_p_zero_values():
    """Some rows have p_zero=None -> skipped silently."""
    rows = [
        {"prompt": "A", "p_zero": 0.3, "extra_info": {"index": 1}},
        {"prompt": "B", "p_zero": None, "extra_info": {"index": 2}},
        {"prompt": "C", "p_zero": 0.7, "extra_info": {"index": 3}},
    ]
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "with_none.parquet")
        df.to_parquet(path)
        ds = Dataset.from_parquet(path)
        result = load_p0(ds)
    assert result["n_loaded"] == 2, f"expected 2 loaded (None skipped), got {result['n_loaded']}"
    print(f"[PASS] test_none_p_zero_values (loaded {result['n_loaded']}/{result['n_total']})")


def test_clipping_bounds():
    """p_zero=0.0 and p_zero=1.0 should be clipped to [eps_p, 1-eps_p]."""
    rows = [
        {"prompt": "A", "p_zero": 0.0, "extra_info": {"index": 1}},
        {"prompt": "B", "p_zero": 1.0, "extra_info": {"index": 2}},
        {"prompt": "C", "p_zero": 0.5, "extra_info": {"index": 3}},
    ]
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clip.parquet")
        df.to_parquet(path)
        ds = Dataset.from_parquet(path)
        result = load_p0(ds, eps_p=0.05)
    assert result["p_0_map"][1] == 0.05, f"0.0 should clip to 0.05, got {result['p_0_map'][1]}"
    assert abs(result["p_0_map"][2] - 0.95) < 1e-6, f"1.0 should clip to 0.95"
    assert abs(result["p_0_map"][3] - 0.5) < 1e-6
    print(f"[PASS] test_clipping_bounds (0.0->0.05, 1.0->0.95, 0.5->0.5)")


if __name__ == "__main__":
    test_synthetic_parquet()
    test_collision_detection()
    test_missing_column()
    test_none_p_zero_values()
    test_clipping_bounds()
    print("\n=== ALL DATASET TESTS PASSED ===")
