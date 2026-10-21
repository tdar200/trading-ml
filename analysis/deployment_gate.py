"""Fail-closed historical and forward-paper qualification for intraday strategies."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from analysis.metrics import max_drawdown, sharpe

PAPER_COLUMNS = (
    "session_date", "candidate_sha256", "predicted_gross_bps", "realized_gross_bps",
    "modeled_cost_bps", "realized_cost_bps", "net_return", "orders_submitted",
    "orders_filled", "broker_reconciled", "broker_evidence_path",
    "broker_evidence_sha256", "risk_breach", "previous_row_sha256", "row_sha256",
)

_PAPER_HASH_COLUMNS = PAPER_COLUMNS[:-1]


@dataclass(frozen=True)
class DeploymentThresholds:
    min_evaluation_sessions: int = 252
    min_portfolio_events: int = 200
    min_sharpe: float = 1.25
    maximum_drawdown: float = -0.10
    maximum_pbo: float = 0.20
    minimum_dsr_confidence: float = 0.95
    maximum_concentration: float = 0.25
    minimum_positive_window_fraction: float = 0.75
    min_paper_sessions: int = 126
    min_paper_events: int = 100
    min_paper_sharpe: float = 1.0
    min_fill_rate: float = 0.95
    max_cost_overrun_fraction: float = 0.40


def candidate_fingerprint(candidate: dict[str, Any]) -> str:
    normalized = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in candidate.items()
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paper_row_fingerprint(row: dict[str, Any]) -> str:
    """Canonical hash for one row in the tamper-evident paper journal."""
    normalized = {}
    for key in _PAPER_HASH_COLUMNS:
        value = row[key]
        if key == "session_date":
            timestamp = pd.Timestamp(value)
            timestamp = (
                timestamp.tz_localize("UTC") if timestamp.tzinfo is None
                else timestamp.tz_convert("UTC")
            )
            value = timestamp.isoformat()
        elif isinstance(value, np.generic):
            value = value.item()
        normalized[key] = value
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def historical_qualification(
    summary: dict[str, Any], costs: pd.DataFrame,
    thresholds: DeploymentThresholds = DeploymentThresholds(),
) -> dict[str, Any]:
    base = summary["evaluation_base_cost"]
    pbo_value = summary.get("probability_backtest_overfit", {}).get("probability_overfit")
    dsr_value = summary.get("deflated_sharpe", {}).get("p_value")
    stress_25 = costs.loc[np.isclose(costs["cost_scenario_bps_per_side"], 2.5)]
    stress_5 = costs.loc[np.isclose(costs["cost_scenario_bps_per_side"], 5.0)]
    windows = int(summary.get("portfolio_window_count", 4))
    gates = {
        "evaluation_sessions": int(summary["evaluation_sessions"]) >= thresholds.min_evaluation_sessions,
        "portfolio_events": int(base["portfolio_events"]) >= thresholds.min_portfolio_events,
        "net_sharpe": float(base["sharpe"]) >= thresholds.min_sharpe,
        "maximum_drawdown": float(base["max_drawdown"]) >= thresholds.maximum_drawdown,
        "positive_at_2_5_bps": bool(len(stress_25) and stress_25.iloc[0]["total_return"] > 0),
        "nonnegative_at_5_bps_with_activity": bool(
            len(stress_5) and stress_5.iloc[0]["total_return"] >= 0
            and stress_5.iloc[0]["portfolio_events"] >= 20
        ),
        "pbo": pbo_value is not None and np.isfinite(pbo_value) and float(pbo_value) <= thresholds.maximum_pbo,
        "deflated_sharpe": dsr_value is not None and np.isfinite(dsr_value) and float(dsr_value) >= thresholds.minimum_dsr_confidence,
        "concentration": float(summary["largest_absolute_contribution_share"]) <= thresholds.maximum_concentration,
        "positive_windows": int(summary["discovery_positive_windows"]) / windows >= thresholds.minimum_positive_window_fraction,
        "sip_nbbo": bool(summary.get("sip_quote_ready", False)),
        "point_in_time_membership": bool(summary.get("point_in_time_membership", False)),
        "event_calendar": bool(summary.get("event_calendar_ready", False)),
        "historical_interval_unexposed": bool(summary.get("historical_interval_unexposed", False)),
    }
    observed = {
        "evaluation_sessions": int(summary["evaluation_sessions"]),
        "portfolio_events": int(base["portfolio_events"]),
        "net_sharpe": float(base["sharpe"]),
        "maximum_drawdown": float(base["max_drawdown"]),
        "positive_at_2_5_bps": float(stress_25.iloc[0]["total_return"]) if len(stress_25) else None,
        "nonnegative_at_5_bps_with_activity": {
            "total_return": float(stress_5.iloc[0]["total_return"]),
            "portfolio_events": int(stress_5.iloc[0]["portfolio_events"]),
        } if len(stress_5) else None,
        "pbo": float(pbo_value) if pbo_value is not None and np.isfinite(pbo_value) else None,
        "deflated_sharpe": float(dsr_value) if dsr_value is not None and np.isfinite(dsr_value) else None,
        "concentration": float(summary["largest_absolute_contribution_share"]),
        "positive_windows": int(summary["discovery_positive_windows"]) / windows,
        "sip_nbbo": bool(summary.get("sip_quote_ready", False)),
        "point_in_time_membership": bool(summary.get("point_in_time_membership", False)),
        "event_calendar": bool(summary.get("event_calendar_ready", False)),
        "historical_interval_unexposed": bool(summary.get("historical_interval_unexposed", False)),
    }
    required = {
        "evaluation_sessions": f">={thresholds.min_evaluation_sessions}",
        "portfolio_events": f">={thresholds.min_portfolio_events}",
        "net_sharpe": f">={thresholds.min_sharpe}",
        "maximum_drawdown": f">={thresholds.maximum_drawdown}",
        "positive_at_2_5_bps": ">0 total return",
        "nonnegative_at_5_bps_with_activity": ">=0 return and >=20 events",
        "pbo": f"<={thresholds.maximum_pbo}",
        "deflated_sharpe": f">={thresholds.minimum_dsr_confidence} confidence",
        "concentration": f"<={thresholds.maximum_concentration}",
        "positive_windows": f">={thresholds.minimum_positive_window_fraction}",
        "sip_nbbo": True,
        "point_in_time_membership": True,
        "event_calendar": True,
        "historical_interval_unexposed": True,
    }
    return {
        "qualified": all(gates.values()), "gates": gates,
        "observed": observed, "required": required,
        "blockers": [name for name, passed in gates.items() if not passed],
    }


def validate_paper_log(frame: pd.DataFrame) -> pd.DataFrame:
    if missing := set(PAPER_COLUMNS) - set(frame.columns):
        raise ValueError(f"paper log missing columns: {sorted(missing)}")
    result = frame[list(PAPER_COLUMNS)].copy()
    result["session_date"] = pd.to_datetime(result["session_date"], utc=True)
    if result["session_date"].duplicated().any() or not result["session_date"].is_monotonic_increasing:
        raise ValueError("paper sessions must be unique and chronological")
    numeric = [
        "predicted_gross_bps", "realized_gross_bps", "modeled_cost_bps",
        "realized_cost_bps", "net_return",
    ]
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("paper numeric fields must be finite")
    for column in ("broker_reconciled", "risk_breach"):
        if result[column].dtype != bool:
            normalized = result[column].astype(str).str.lower().str.strip()
            if not normalized.isin({"true", "false", "1", "0"}).all():
                raise ValueError(f"paper {column} must be boolean")
            result[column] = normalized.isin({"true", "1"})
    for column in ("orders_submitted", "orders_filled"):
        values = pd.to_numeric(result[column], errors="raise")
        if (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"paper {column} must contain non-negative integers")
        result[column] = values.astype(int)
    if (result["orders_filled"] > result["orders_submitted"]).any():
        raise ValueError("paper orders_filled cannot exceed orders_submitted")
    hashes = result["broker_evidence_sha256"].astype(str)
    if not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("paper broker_evidence_sha256 must contain SHA-256 digests")
    for value in result["broker_evidence_path"].astype(str):
        evidence_path = PurePosixPath(value)
        if evidence_path.is_absolute() or ".." in evidence_path.parts or not value:
            raise ValueError("paper broker_evidence_path must be a safe relative path")
    expected_previous = "GENESIS"
    for index, row in result.iterrows():
        if str(row["previous_row_sha256"]) != expected_previous:
            raise ValueError(f"paper hash chain broken at row {index}")
        expected = paper_row_fingerprint(row.to_dict())
        if str(row["row_sha256"]) != expected:
            raise ValueError(f"paper row hash mismatch at row {index}")
        expected_previous = expected
    return result


def paper_qualification(
    frame: pd.DataFrame, thresholds: DeploymentThresholds = DeploymentThresholds(),
    *, expected_fingerprint: str | None = None, evidence_root: Path | None = None,
) -> dict[str, Any]:
    paper = validate_paper_log(frame)
    equity = pd.Series(np.concatenate([[1.0], np.cumprod(1 + paper["net_return"].to_numpy())]))
    traded = paper[paper["orders_filled"] > 0]
    predicted_edge = float(
        ((traded["predicted_gross_bps"] - traded["modeled_cost_bps"]).clip(lower=0)
         * traded["orders_filled"]).sum()
    )
    cost_overrun = float(
        ((traded["realized_cost_bps"] - traded["modeled_cost_bps"]).clip(lower=0)
         * traded["orders_filled"]).sum()
    )
    paper_sharpe = float(sharpe(paper["net_return"]))
    submitted = int(paper["orders_submitted"].sum())
    filled = int(paper["orders_filled"].sum())
    fill_rate = filled / submitted if submitted else 0.0
    evidence_verified = (
        evidence_root is not None and paper_evidence_verified(paper, evidence_root)
    )
    gates = {
        "frozen_candidate": paper["candidate_sha256"].nunique() == 1 and (
            expected_fingerprint is None or paper["candidate_sha256"].iloc[0] == expected_fingerprint
        ),
        "paper_sessions": len(paper) >= thresholds.min_paper_sessions,
        "paper_events": filled >= thresholds.min_paper_events,
        "positive_net_return": float(equity.iloc[-1] - 1) > 0,
        "paper_sharpe": np.isfinite(paper_sharpe) and paper_sharpe >= thresholds.min_paper_sharpe,
        "paper_drawdown": float(max_drawdown(equity)) >= thresholds.maximum_drawdown,
        "fill_rate": fill_rate >= thresholds.min_fill_rate,
        "cost_overrun": predicted_edge > 0 and cost_overrun / predicted_edge <= thresholds.max_cost_overrun_fraction,
        "broker_evidence": bool(paper["broker_reconciled"].all()) and evidence_verified,
        "zero_risk_breaches": not bool(paper["risk_breach"].any()),
    }
    observed = {
        "frozen_candidate": paper["candidate_sha256"].iloc[0] if len(paper) else None,
        "paper_sessions": len(paper), "paper_events": filled,
        "positive_net_return": float(equity.iloc[-1] - 1), "paper_sharpe": paper_sharpe,
        "paper_drawdown": float(max_drawdown(equity)), "fill_rate": fill_rate,
        "cost_overrun": cost_overrun / predicted_edge if predicted_edge > 0 else None,
        "broker_evidence": bool(paper["broker_reconciled"].all()) and evidence_verified,
        "zero_risk_breaches": int(paper["risk_breach"].sum()),
    }
    required = {
        "frozen_candidate": expected_fingerprint or "one unchanged candidate",
        "paper_sessions": f">={thresholds.min_paper_sessions}",
        "paper_events": f">={thresholds.min_paper_events}", "positive_net_return": ">0",
        "paper_sharpe": f">={thresholds.min_paper_sharpe}",
        "paper_drawdown": f">={thresholds.maximum_drawdown}",
        "fill_rate": f">={thresholds.min_fill_rate}",
        "cost_overrun": f"<={thresholds.max_cost_overrun_fraction}",
        "broker_evidence": True, "zero_risk_breaches": 0,
    }
    return {
        "qualified": all(gates.values()), "gates": gates,
        "sessions": len(paper), "events": filled,
        "total_return": float(equity.iloc[-1] - 1), "sharpe": paper_sharpe,
        "max_drawdown": float(max_drawdown(equity)),
        "observed": observed, "required": required,
        "blockers": [name for name, passed in gates.items() if not passed],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paper_evidence_verified(paper: pd.DataFrame, evidence_root: Path) -> bool:
    root = evidence_root.resolve()
    for _, row in paper.iterrows():
        path = (root / str(row["broker_evidence_path"])).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return False
        if file_sha256(path) != str(row["broker_evidence_sha256"]):
            return False
    return True


def verify_deployment_approval(run_dir: Path) -> dict[str, Any]:
    """Fail closed unless a qualification report still matches every input artifact."""
    report_path = run_dir / "deployment_qualification.json"
    if not report_path.exists():
        raise RuntimeError("deployment qualification report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("deployment_approved", False):
        raise RuntimeError("strategy has not passed deployment qualification")
    frozen_path = run_dir / "frozen_candidate.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if candidate_fingerprint(frozen["candidate"]) != frozen["sha256"]:
        raise RuntimeError("frozen candidate fingerprint mismatch")
    if report.get("candidate_sha256") != frozen["sha256"]:
        raise RuntimeError("qualification report belongs to a different candidate")
    artifacts = report.get("artifact_sha256", {})
    required_artifacts = {
        "frozen_candidate.json", "evaluation_cost_stress.csv", "paper_observations.csv",
    }
    if set(artifacts) != required_artifacts:
        raise RuntimeError("qualification report does not bind every required artifact")
    for name, expected in artifacts.items():
        path = run_dir / name
        if not path.exists() or file_sha256(path) != expected:
            raise RuntimeError(f"qualified artifact changed or is missing: {name}")
    paper = validate_paper_log(pd.read_csv(run_dir / "paper_observations.csv"))
    if not paper_evidence_verified(paper, run_dir):
        raise RuntimeError("qualified broker evidence changed or is missing")
    if not report.get("historical", {}).get("qualified") or not report.get("paper", {}).get("qualified"):
        raise RuntimeError("qualification report contains failed gates")
    return report
