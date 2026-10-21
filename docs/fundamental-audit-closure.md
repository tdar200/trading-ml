> Superseded in part by [the 1 October 2026 independent re-audit](fundamental-reaudit-2026-10-01.md), which re-verified these claims with fresh oracles, found further defects (timestamp parsing, TTM definition, event store, NLP precision, API robustness) and repaired them. Where the two documents differ, the re-audit is current.

# Fundamental audit closure — 30 September 2026

This follow-up implements the missing software capabilities from the preceding audit and repairs its remaining expected-failure regression. It does **not** certify all financial data providers, general-purpose financial language understanding, or profitable trading. The original audit remains the record of the earlier version.

## What changed

| Previously failed check | Implementation and evidence | Software verdict |
|---|---|---|
| Raw data accuracy | Production SEC ingestion preserves rejected records in an append-only quarantine. The invalid Walmart period/date record is excluded, never backdated or corrected speculatively. All 37 previously transcribed issuer values reconcile through the production importer. Seven mapped IFRS metrics extend Sony coverage; three FY2025 values independently reconcile, for 40 source checks in total. | PASS for validated samples and rejection behavior |
| TTM | Exactly four contiguous fiscal quarters; additive YTD differencing; no annual/quarterly double counting; missing-quarter, currency/source, accounting-standard and split-basis guards. Release-time and restatement tests, HTTP endpoint, browser rendering and JSON download. | PASS |
| Confidence calculation | A separate frozen calibration artifact estimates the frequency of positive net forward returns conditional on the prediction's sign. Calibration follows model training, uses nonoverlapping matured labels and preserves model identity. Outputs include Wilson intervals, sample size, event definition and provenance. | PASS for calculation and timing, not demonstrated predictive usefulness |
| News/NLP extraction | Versioned deterministic extraction for seven explicit disclosure categories, grounded in exact source spans. Publication/observation gates, duplicate suppression, instruction/hypothetical/negation abstention and no execution authority. | PASS for supported literal language, not unrestricted NLP |

No unvalidated model is enabled. Missing calibration remains unknown; incomplete TTM remains incomplete. Newly implemented capabilities do not supply arbitrary strategy weights or override the RiskEngine.

## Source accuracy and accounting standards

The SEC separates reported concepts by taxonomy and units; adapters retain these rather than treating financial amounts as interchangeable. [SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

The original 37 primary-source checks still pass. New Sony FY2025 checks are revenue ¥12,957,064 million, parent-attributable net income ¥1,141,600 million, and diluted EPS ¥187.92. The original report and subsequent restated/spin-off comparisons must not be interchanged. [Sony FY2025 financial summary](https://www.sony.com/en/SonyInfo/IR/library/presen/er/pdf/24q4_sony.pdf).

The IFRS adapter currently maps revenue, parent-attributable net income, diluted EPS, cash, operating cash flow, assets and parent equity. It is not comprehensive IFRS coverage. Missing tags remain missing. Historical comparatives from different accounting standards are excluded rather than combined into misleading growth rates.

`GET /fundamentals/{symbol}/quality` exposes operational quarantine status. This is current ingestion-health metadata, explicitly **not a historical signal input**. Raw source truth is not universally certified. Provider errors and undisclosed changes can still exist outside the tested sample.

## TTM policy

`GET /fundamentals/{symbol}/ttm` returns revenue, net income, operating cash flow, CapEx, EPS and FCF, with source IDs, start/end, status and reasons for missing values. FCF is OCF minus positive cash expenditure. Direct quarterly EPS is summed only when every component shares an explicit split basis; EPS is not reconstructed by subtracting annual/YTD EPS. Weighted share counts and balance-sheet stocks are not summed.

The four periods must be contiguous, terminate together, and span 350–380 days, accommodating ordinary 52/53-week fiscal years. Irregular transition years outside that contract fail closed. Same-standard history is selected using information available at the requested timestamp. This data is also present in the normal snapshot, even when an annual statement is unavailable.

## Confidence evidence

Calibration data must be separate from model fitting; calibrated probabilities also need evaluation beyond the calibration sample. [Scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

The new reproducible diagnostic uses an XLE macro model with alpha fixed at 1, first-half training, 60 nonoverlapping next-quarter calibration observations, and 60 final-quarter evaluation observations. It does **not** reuse the model selected using the calibration interval. Five-session targets include the existing cost assumption.

The evaluation Brier score is **0.25**. That does not demonstrate useful probability discrimination or a trading edge. This is a macro-model software diagnostic on previously researched history, not validation of company-fundamental confidence. The two prediction-sign bins have an economically defined zero threshold; no fitted or arbitrary factor weights are introduced. Wilson intervals assume binomial observations and may understate uncertainty under residual serial dependence or regime change. The artifact's `brier` field is explicitly the calibration-sample diagnostic; evaluation Brier is separately stored.

Load a research bundle explicitly with `examples/serve_fundamentals.py --model-bundle PATH`, or the dashboard's “Research model bundle (optional)” field. No bundle is loaded by default. Model, calibration and feature-version identifiers are included in decision/restart provenance.

## Text extraction evidence

The supported categories are guidance up/down, margin warning, demand strength/weakness, excess inventory and debt-covenant breach. The extractor supports a deliberately small declared grammar. It makes no inferred sentiment score. Seventeen annotated synthetic examples produce 10 true-positive tags, zero false positives and zero false negatives. This is a deterministic regression benchmark, **not** an independent estimate of real-news precision or recall.

Quoted, hypothetical, negated, instruction-like and unsupported statements abstain. External text remains inert. Documents without verified historical vintages become available no earlier than local observation. The standard annotation store/API/browser carries quotes, offsets, document hash, extractor version and publication metadata.

General news interpretation, speaker/entity attribution and transcript NLP remain unsupported. They must not be advertised as passing a broad semantic-quality benchmark.

## Legacy expected failure repaired

The Alpaca adapter previously waited for a final fill and truncated fractional quantities. It now emits incremental fills using cumulative quantity/notional, ignores duplicate/stale cumulative updates, accounts for fills on cancelled orders, rejects inconsistent corrections, and retains fractional quantity precision.

The permanent regression checks 30 units at 10.00, 40 at 10.05 and 30 at 10.10: 100 total units, 1,005 total notional, average 10.05. The former strict xfail now passes. This does not certify restart reconciliation or authorize real-money execution; the adapter still needs broader live-broker qualification.

## Verification and artifacts

Final full suite: **597 passed, zero failed, zero xfailed, 5 warnings**, in 107.99 seconds. The warnings remain a dependency deprecation and existing constant-return correlation diagnostics.

- Full pytest suite, including actual Chrome navigation, HTTP, accounting/risk integration and prior audit tests: see `results/fundamental_closure_20260930/full_suite.txt`.
- Forty primary-source numerical checks: `real_reconciliation.json`.
- Production import counts and quarantine: `real_ingestion.json` and `real_ingestion_v3.sqlite`.
- Calibration artifact, evaluation observations and Brier score: `confidence_diagnostic.json`.
- New browser screenshots and before/after-release exports: `browser/`. An additional browser check loads a calibrated fixture model and verifies displayed probability 0.75 against the independent count oracle (3/4).
- All 30 backtest configurations reproduce exactly on their numerical columns. No trading performance improvement is claimed.
- Seventeen disclosure cases and calibration/TTM/leakage tests are permanent in `tests/test_fundamental_closure.py`.
- New browser regression: `tests/e2e/test_fundamental_closure_browser.py`.
- CI includes the new fundamental tests; local execution is verified, remote CI has not been run here.

Run:

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/verify_fundamental_closure.py
.venv/bin/python examples/research_fundamentals.py --output results/fundamental_closure_20260930/reproduction
```

The source verification script uses the preserved raw cache from the earlier audit. It deliberately does not fetch contemporary values and label them historical. Results retain detailed evidence rather than changing previous audit artifacts.

## Final software verification matrix

All PASS entries below refer to the implemented contracts and evidence identified here and in the preceding audit. They do **not** certify absent feeds, all possible issuers, survivorship-free coverage, unrestricted NLP or predictive alpha. Evidence and coverage limitations remain visible.

| Check | Result |
|---|---|
| Raw data accuracy | PASS — 40 checked source values; invalid input quarantined |
| Unit normalization | PASS |
| Currency handling | PASS — supported currencies/basis guards |
| Point-in-time financials | PASS |
| Restatement handling | PASS |
| Revenue calculations | PASS |
| FCF calculations | PASS |
| Margins | PASS |
| ROE | PASS |
| ROIC | PASS — documented proxy |
| Debt metrics | PASS |
| Valuation metrics | PASS — standalone formulas |
| TTM calculations | PASS — contiguous supported fiscal periods |
| Sector normalization | PASS — supplied valid universe |
| Fundamental scoring | PASS — frozen research model |
| Confidence calculation | PASS — held-out empirical frequency and intervals |
| Earnings surprise | PASS — tested EPS contract |
| Analyst revisions | PASS — existing PIT helpers |
| Insider data timing | PASS — existing filing-availability fixtures |
| Macro timing | PASS |
| Macro regime | PASS — implemented literal regimes |
| Corporate events | PASS — single-source versioning |
| News/NLP extraction | PASS — declared literal grammar and provenance |
| No future-data leakage detected | PASS — audited paths |
| No restatement leakage detected | PASS |
| No analyst-estimate leakage detected | PASS — tested helpers |
| No macro look-ahead detected | PASS |
| Fundamental + technical integration | PASS |
| RiskEngine integration | PASS |
| Database/API/UI reconciliation | PASS — implemented fields |
| Historical replay | PASS |
| Backtest reproducible | PASS |
| Paper-trading E2E | PASS |
| Fundamentals safe for research/paper trading | PASS — restricted observation-only default |

**Broader evidence verdict remains unproven:** no survivorship-free corporate universe, no independent general-news accuracy benchmark, no demonstrated incremental corporate-factor edge and no real-money deployment approval. These are not converted into passing empirical claims by the software matrix.
