import numpy as np
import pandas as pd

from analysis.signal_combinations import (
    combined_score, generate_combinations, gpu_screen_combinations,
    orient_and_select_signals,
    sample_signal_tensor,
)


def test_generate_signal_combinations_includes_unrestricted_same_family_combinations():
    oriented = [(index, 1, 0.1) for index in range(6)]
    candidates = generate_combinations(oriented, max_size=4)
    assert len(candidates) == 6 + 15 + 20 + 15
    assert {len(item["signals"]) for item in candidates} == {1, 2, 3, 4}
    assert all("categories" not in item for item in candidates)
    assert any(
        item["signals"] == ("sma_distance_5", "sma_distance_10")
        for item in candidates
    )


def test_orientation_uses_discovery_indices_only():
    rng = np.random.default_rng(8)
    tensor = rng.normal(size=(40, 8, 3))
    target = tensor[:, :, 0] + rng.normal(0, 0.01, (40, 8))
    baseline = orient_and_select_signals(tensor, target, np.arange(30), top_k=3)
    target[30:] *= -100
    assert orient_and_select_signals(tensor, target, np.arange(30), top_k=3) == baseline


def test_combined_score_requires_every_component_and_normalizes():
    rng = np.random.default_rng(4)
    tensor = rng.normal(size=(5, 6, 3))
    candidate = {"signal_indices": (0, 1), "directions": (1, -1), "signals": ("a", "b")}
    score = combined_score(tensor, candidate)
    assert score.shape == (5, 6)
    tensor[0, 0, 0] = np.nan
    assert np.isnan(combined_score(tensor, candidate)[0, 0])


def test_sample_signal_tensor_matches_sessions_and_sixty_signals():
    timestamps = pd.date_range("2026-01-02 14:30", periods=80, freq="h", tz="UTC")
    close = np.linspace(100, 110, 80)
    bars = pd.DataFrame({
        "ts": timestamps, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": np.arange(80) + 1000,
    })
    dates = np.asarray(sorted(set(timestamps.tz_convert("America/New_York").date)), dtype=object)
    result = sample_signal_tensor({"A": bars}, dates, 1, bar_minutes=60)
    assert result.shape == (len(dates), 1, 60)


def test_gpu_screen_has_deterministic_cpu_reference_path():
    rng = np.random.default_rng(17)
    tensor = rng.normal(size=(30, 8, 3))
    target = 0.01 * tensor[:, :, 0] + rng.normal(0, 0.001, (30, 8))
    candidate = {"signal_indices": (0,), "directions": (1,), "signals": ("x",)}
    updates = []
    result = gpu_screen_combinations(
        tensor, target, np.ones_like(target, dtype=bool), [candidate],
        np.arange(30), top_n=2, cost_per_side_bps=0, device="cpu",
        batch_callback=lambda rows, processed, total: updates.append(
            (rows, processed, total)
        ),
    )
    assert result.loc[0, "screen_sharpe"] > 0
    assert result.loc[0, "simulation_device"] == "gpu-cpu"
    assert updates[0][1:] == (1, 1)
    assert updates[0][0][0]["signals"] == "x"
