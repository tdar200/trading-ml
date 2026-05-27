"""
ModelStore — persist + look up trained models keyed by training-cutoff date.

Pickle format (via joblib):
- Filename: ``{label}_{cutoff_date:%Y-%m-%d}.joblib`` under ``cache_dir``.
- Payload: ``{"model": LGBMClassifier, "feature_names": list[str]}``.

Usage:
    store = ModelStore(cache_dir=Path("models/cache"), label="default")
    store.save(pd.Timestamp("2024-04-01"), trained_model, feature_names=[...])
    pair = store.load_for(pd.Timestamp("2024-05-15"))   # → (model, feature_names) or None

``load_for`` picks the most-recent stored model whose cutoff_date is
STRICTLY before the query date. This enforces walk-forward: a query at
date D never sees a model trained on data including D.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


_CUTOFF_RE = re.compile(r"^(?P<label>.+)_(?P<date>\d{4}-\d{2}-\d{2})\.joblib$")


@dataclass
class ModelStore:
    """Persist + retrieve dated model artifacts. One ``label`` per
    (hyperparameter combo) namespace within a shared ``cache_dir``."""
    cache_dir: Path
    label: str
    # Set by load_for to the path of the artifact actually loaded
    # (useful for diagnostics; not load-bearing).
    last_loaded_path: Path | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _filename(self, cutoff_date: pd.Timestamp) -> Path:
        return self.cache_dir / f"{self.label}_{cutoff_date.strftime('%Y-%m-%d')}.joblib"

    def save(
        self,
        cutoff_date: pd.Timestamp,
        model: Any,
        *,
        feature_names: list[str],
    ) -> Path:
        path = self._filename(cutoff_date)
        joblib.dump({"model": model, "feature_names": list(feature_names)}, path)
        return path

    def list_cutoffs(self) -> list[pd.Timestamp]:
        out: list[pd.Timestamp] = []
        for path in self.cache_dir.glob(f"{self.label}_*.joblib"):
            m = _CUTOFF_RE.match(path.name)
            if not m:
                continue
            if m.group("label") != self.label:
                continue
            out.append(pd.Timestamp(m.group("date")))
        return sorted(out)

    def load_for(self, query_date: pd.Timestamp) -> tuple[Any, list[str]] | None:
        """Return (model, feature_names) for the most-recent cutoff strictly before
        query_date. None if no qualifying artifact exists."""
        cutoffs = self.list_cutoffs()
        eligible = [c for c in cutoffs if c < query_date]
        if not eligible:
            self.last_loaded_path = None
            return None
        chosen = max(eligible)
        path = self._filename(chosen)
        payload = joblib.load(path)
        self.last_loaded_path = path
        return payload["model"], list(payload["feature_names"])
