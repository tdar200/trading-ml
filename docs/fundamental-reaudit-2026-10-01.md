# Independent re-audit of the fundamental-analysis layer, 1 October 2026

**Scope.** The point-in-time (PIT) fundamental layer (`fundamentals/`), the paper engine path that consumes it (`livesim/small_account.py`, `strategies/small_account.py`), the dashboard page, the research code that uses it (`fundamentals/research.py`), and the written research claims about it.

**Method.** The 30 September audit and closure documents were treated as claims to reproduce, not as evidence. Seven independent tracks (PIT, real data, formulas, macro/events, decision/risk, API/UI, backtest/leakage) each wrote a fresh oracle that imports no production calculation helper and none of the earlier oracle, then attacked the system through its public entry points. Confirmed defects were reproduced as strict-xfail tests, then repaired and the tests promoted to permanent regressions.

**Headline.**

- **Software correctness: PASS for the implemented, tested scope, after repairs.** 61 defects were found and 58 repaired. None was CRITICAL: no future information reached a historical decision. 10 were HIGH, 24 MEDIUM, 27 LOW. Three remain open on purpose (below).
- **Evidence of a trading edge: none.** Several headline research claims do not survive recomputation and are withdrawn (section 11).
- **Not production-ready for real money.** Passing software tests say nothing about predictive value.

## 1. Architecture audited

```
SEC companyfacts + submissions  ->  providers.sec_facts  ->  Fact.normalized  ->  store.ingest_validated
ALFRED vintage CSV              ->  providers.alfred_snapshot  ----------------->  (append-only SQLite: observations,
JSON export                     ->  providers.ingest_json_records ------------->    raw_documents, corporate_events,
document text                   ->  catalysts.extract_disclosures ------------->    ingestion_rejections, decisions)
                                                    |
                         store.as_of(T): available_at <= T, latest known version per (metric, period, source)
                                                    |
        FeatureEngine.snapshot / ttm  +  MacroRegimeService  +  EventCatalystEngine  ->  FundamentalScoreEngine (frozen ridge)
                                                    |
                    CombinedSignalEngine.decide  (+ technical signal)  ->  PaperEngine -> RiskEngine.approve -> sized paper order
                                                    |
                        decisions journal  ->  FundamentalAPI / HTTP  ->  Streamlit page
```

Wired end to end: annual statement features, TTM, quarter changes, macro regime, event/annotation state, frozen-model score, combined decision, risk gate, journal, API, page.
Standalone helpers (tested, not served by the API or page): valuation multiples, sector normalisation, FX conversion, split adjustment, estimate revisions, insider activity, sector adapters, own-history percentile.

## 2. Data sources

| Source | Used for | Cross-check performed |
|---|---|---|
| SEC EDGAR companyfacts + submissions | statements, acceptance time | 17 issuers (8 audited earlier, 9 new: AAPL, PFE, F, KO, NEE, AMZN, PLUG, NKE, GSK). Every stored fact re-derived from raw JSON (35,683 + the new issuers): value, period, accession, unit, form, `available_at` all exact. 338 headline values read from the primary 10-K/10-Q/20-F documents: 0 FAIL, 14 GAP (unmapped tags, coverage not error). |
| ALFRED vintage snapshots | macro (CPI, UNRATE, FEDFUNDS) | 26 vintages x 3 series compared cell by cell with the raw CSV; 10,000+ (series, period, boundary) values. |
| Dated JSON export | any provider-neutral record | unit/currency/schema fuzz (1,800 seeds). |
| Price frames | research only | not a fundamental source. |

No licensed estimates, insider (Form 4), short-interest, or news feed is populated; those paths are tested with synthetic fixtures only.

## 3. Point-in-time audit (most important section)

| Data type | Verified | Result |
|---|---|---|
| Financial statements | Visibility flips at the exact publication instant (07:00:00 vs 06:59:59.999999999) through `store.as_of`, `snapshot`, `FundamentalAPI.get`, real HTTP and the browser; offsets Z, +01:00, -04:00, +05:30; four DST days; SQL string comparison attacked with mixed precision and years 1677 to 9999. 320 random multi-year companies (restatements, withdrawals, late/early filing, duplicates, shuffled ingest) against an independent oracle at 5 probes each, including publication +/- 1 microsecond: 0 mismatches. | PASS |
| Restatements | EPS 1.20 to 1.05 flips at the exact instant; both rows kept; 1,042 real restated keys flip exactly at the later filing. | PASS |
| Late filers | Previous annual and TTM period used until the filing instant. | PASS |
| Future-data canary | Value 123456789123 in future filings, vintages, events, estimates and articles never appears in any snapshot, TTM, history, decision, watchlist or UI download. Full database vs physically truncated database: byte-identical canonical JSON including `data_snapshot_id`, 320 seeds, over HTTP, in the Streamlit page and on the shipped real database. | PASS |
| Earnings events | Consensus published at or after the event is excluded (`NO_PRE_EVENT_CONSENSUS`); event availability inclusive at 14:00:00 and invisible at 13:59:59.999999. | PASS |
| Analyst estimates | Fixed forecast period; revisions after T invisible; 30 day boundary exact to the second. No production caller sources forward EPS, so no non-PIT consensus can reach a decision. | PASS (helper only) |
| Insider data | A trade on D filed on D+2 is invisible until the filing second. Requires `metadata.transaction_id` for several trades to coexist. | PASS (synthetic) |
| Macro | Availability = vintage day + 1 at 00:00 UTC. A 08:30 ET release is never visible early (scanned every 5 minutes); visible up to about 11.5 h late (conservative). Real revised CPI: 68 of 74 periods revised; the original value shows until the revising vintage is public. Using latest-revised data instead would flip the inflation label on 4.3% of ETF days and the labour label on 25.8%. | PASS (conservative) |
| News | `available_at = max(published, observed)` unless `vintage_verified`; a retrospective article cannot influence an earlier decision. | PASS |
| Daily-bar paper engine | Bar timestamp 04:00 UTC (before the open) decides; fills at the open. Filings between 04:00 and 13:30 UTC are seen the next bar (conservative, never early). | PASS |
| Concurrency / restart | Writer committing mid-decision is invisible to that decision (mutation-tested: removing the read transaction fails the tests); SIGKILL loses only uncommitted rows; reopened database reproduces decision and `data_snapshot_id` under three hash seeds. | PASS |
| Every "latest" selection in the package | 12 sites reviewed (guard test, ten at audit time plus two added by the repairs); all operate on rows already filtered by `available_at <= T`. | PASS |

**Real host-dependent defect found and fixed (QA-PIT-08, HIGH latent).** `utc()` used pandas free-form parsing, so a timezone abbreviation was resolved through the host's local zone: on a London host `Sat, 10 May 2025 07:00:00 GMT` became 06:00Z, making a fact available one hour early. This machine reports BST, so it was live. `utc()` now accepts only ISO-8601 with `Z` or a numeric offset (or an aware datetime); `04/05/2025`, `24-03-04` and `... GMT/BST/EST` are rejected.

## 4. Formula verification

Thirteen golden companies (the supplied Company A, high/low growth, loss-making, high debt, cash-rich, distressed, diluted, improving, deteriorating, bank-like, zero-denominator, sparse) were pushed through the public path (Facts in "GBP millions" to store to snapshot) and compared with a Decimal/Fraction oracle: **all 62 features, flags and valuation ratios match, maximum relative error 1.8e-13.** 400 random seeds compared 24,242 features with 0 mismatches. Removing each of 29 inputs in turn never produced 0, infinity or NaN.

| Metric (Company A) | Expected | Actual | Difference | Result |
|---|---:|---:|---:|---|
| Revenue growth | 20% | 20% | < 1e-12 | PASS |
| Gross / operating / net margin | 50% / 20% / 15% | same | < 1e-12 | PASS |
| FCF, FCF margin | 20m, 16.6667% | same | 0 | PASS |
| Current ratio, debt/equity | 2.0, 0.2222 | same | < 1e-12 | PASS |
| Market cap, P/E | 200m, 11.1111 | same | < 1e-12 | PASS |
| Price/FCF, FCF yield | 10, 10% | same | 0 | PASS |
| ROE (average equity) | 18/85 | 0.2117647 | < 1e-12 | PASS |
| ROIC (NOPAT / average invested capital) | 19.2/82.5 | 0.2327 | < 1e-12 | PASS |
| Dilution 100m to 120m | 20% | 20% | < 1e-12 | PASS |

TTM (phases 86 to 89) was checked against a fiscal-quarter-index oracle on 300 seeds and against the printed documents: before the repair 22 of 24 official TTM values matched (one coverage gap and one real defect, both now handled); after the repair TTM equals the printed FY total at every year end of the 8 audited companies and the 5 sampled new ones.

**Conventions confirmed, not defects:** growth needs a positive base (EPS from -1 to +1 is unknown; +1 to -1 is -200%); ratios with a non-positive denominator are unknown; `quick_ratio` also deducts prepaids; ROIC subtracts all cash; sector ratio gating is a literal label match (now case-insensitive).

**Not implemented (reported as absent, not faked):** PEG, ROIC/margin momentum, a 0 to 100 composite score, factor weights summing to one, sector-weighted dimension scores, HIGH_DEBT / DISTRESS / ESTIMATE_CUTS flags, median/quartile statistics.

## 5. Score verification

The score is a frozen ridge prediction of the 5-session net return, not a 0 to 100 quality score. Verified: hand arithmetic (0.07, 0.8125); 200 random models against an exact solve; 60 ridge fits against an exact-Fraction solution using training-only statistics; 100 monotonicity seeds; missing, NaN or infinite features give no score; non-finite output fails closed; strict `>` threshold behaves at 0.069999, 0.07, 0.070001 and at neighbouring ulps; completeness is reported separately from the score. Confidence is the held-out sign frequency with a Wilson interval; its mechanics match an independent count, but the shipped artifact is degenerate (bins [[0,0],[60,30]], every probability 0.5, Brier 0.25), so it carries no predictive information.

## 6. Macro verification

26 real vintages x 3 series match the raw CSV exactly; 400 random timestamps match an oracle built only from the CSV files (status, regime, six features, exact lineage); the state flips at each of 26 vintage-availability instants to the microsecond. Hand values: CPI yoy 1/30, 3-month change 1/330. Staleness boundary exact (100 days 23:59:59 available, 101 days stale). Absent by design: yield curve, PMI, a composite macro score.

## 7. Catalyst verification

EPS surprise (negative, zero, tiny consensus), decay 2^(-age/half-life) at days 0, 1, 5, 20 for four half-lives, age measured from `available_at`, wildcard events, corrections in either insertion order: all match an exact oracle. Not implemented: revenue surprise, typed M&A/CEO/capital-raise/buyback/dividend extraction, cross-source de-duplication (five feeds give five entries, none aggregated).

**Text extraction precision.** The earlier "17 examples, 0 false positives" was a regression check only. A 118-snippet corpus (66 gold spans, 58 negatives of which 49 were deliberately hard: denials, hypotheticals, other-company quotes, prior-period references, abbreviation/decimal splits) was annotated before the extractor ran.

| | Precision | Recall |
|---|---:|---:|
| Before repair | 0.557 | 0.742 |
| After repair | 0.939 | 0.697 |

The cue lists were tuned after seeing this corpus, so the second row is not an independent estimate; real-filing precision is unknown. Recall is low by design (paraphrase, passive voice and named subjects are not extracted). Catalysts have zero decision authority: decisions are identical with a +899% surprise, a -600% miss, or guidance annotations present (only `data_snapshot_id` differs).

## 8. Combined strategy verification

1,600 combinations (5 policies x shadow x 5 models x 4 fundamental states x 4 macro states x technical signal) match an independent specification for permit, label, explanation and score. A score of 1,000,000 cannot override: kill switch, `STOP_TRADING`, live mode, stale data (90.001 s), clock skew, spread/slippage, duplicate signal, minimum notional, daily/weekly loss, drawdown, database failure. Exact boundaries still trade. A GBP 100 deterministic scenario (score 0.8125, Donchian breakout, macro available, +10% earnings surprise) sized the first entry at 0.114073 units at 105.042, matching an independent Decimal oracle to 1e-12; three target exits ended at 102.9909, matching an independent ledger. An exception inside `decide()` leaves cash, positions, journal and curve unchanged and the next bar trades normally.

## 9. Database / API / UI reconciliation

A 188-row table (oracle, raw SQLite, HTTP JSON, UI) for revenue, EPS, FCF, ROIC, debt/equity, growth, margins, TTM, macro, catalyst and score across ten companies (non-calendar fiscal years, loss-maker, GBP-millions reporter, 6.8e11 USD-billions company, negative equity, restatement, late filer) had **0 mismatches**. All endpoints were exercised at four timestamps (valid JSON, no NaN tokens, UTC offsets, `no-store`), with SQL/path/XSS payloads in six parameters, 20-thread concurrency, and missing/corrupt/locked/empty/truncated database files (503 "no trade", never stale numbers). In real Chrome: displayed card strings, decoded chart data (revenue [80,90,100] becomes [80,90,98 restated,120] after the filing; EPS 1.20 to 1.05 at 07:00:00), fiscal-year x-axis for a September and a June fiscal year end, and download JSON equal to the API.

Not provided by the API or page (not defects): FCF/ROIC/margin/valuation history charts, valuation multiples, coverage by symbol/sector/year, a "FY2024" label (the axis shows the period end date).

## 10. Leakage audit

| Check | Result |
|---|---|
| Look-ahead (financials, estimates, macro, events, news) | none found |
| Restatement leakage | none; the earlier version is shown until the later filing is public |
| Macro revision leakage | none in the store; research uses vintage state per decision timestamp (verified on 1,464 real bar timestamps) |
| Analyst-estimate leakage | none; helper is not wired into any decision |
| ML label leakage (fundamentals research) | none: scaler fitted on training rows only, labels matured before the fit clock, alpha chosen on validation, diagnostic window untouched; truncation and x7 future-price canaries leave features, labels, the model id and the journal byte-identical |
| Stage 5 meta features | **caveat (QA-BT-02, open):** features use `close[date]` and the backtest fills at that same close, so decision information is not strictly earlier than the fill. Not a future-bar leak; not fixed because it would invalidate the stored models |
| Survivorship / universe | **not survivorship-free:** fixed present-day ETF list, one-company fundamental research database, 8 mega-cap survivors in the audit database, point-in-time membership optional. A present-day stock list of recent winners is used by the scalping finder. No delisting-return handling (series that end are closed at the last close; a price of exactly 0 raises) |
| Index constituents / sector history | not implemented; the normaliser trusts a caller-supplied universe |

## 11. Backtest reproduction and research claims

The research was re-run from scratch on a copy of the database: all 1,120 numeric cells in the comparison, incremental and IC tables are **bit-identical** to the three stored runs; an independent Decimal oracle reproduces all 21 metric columns across 30 configurations to 1e-12; a separate rule simulator matches the real `PaperEngine` on 150 random price paths (gaps, DST, halts, every exit type). Of 111 documented claims, 97 recalculate, 4 fail, 3 are invalid, 7 are informational.

**Claims withdrawn or qualified (README now says so):**

- **Stage 5 "meta clearly beats baseline OOS, +29.1% median window Sharpe".** The "walk-forward" windows are nested and expanding, so the median is not out-of-sample. On the full 1,463 sessions the best meta combination has Sharpe 0.9916 against 0.9867 (+0.5%); 1 of 24 combinations beats the baseline; total return 68.8% against 153.7%; meta wins 1 of 5 disjoint sub-periods.
- **Deflated Sharpe z = 23.96 / 25.14 / 23.28.** These combine an annualised Sharpe with a per-day standard error. Units-consistent z is about 2.47 (Stage 3), 2.10 and 1.91 (Stage 5). With the best window's own sample size, z = 1.11, p = 0.87, below the 0.95 gate.
- **Macro filter.** Not better than random trade skipping (24-seed placebo, all p > 0.2). Inert for XLE (score above zero every day) and blocks every XLP and XLV trade; joint and regime variants equal the technical-only system in 35 of 36 runs.
- **Selection.** The "leading XLE candidate" is the best of 144 with none qualifying; all 144 diagnostic outcomes had already been inspected.
- **Document wording (fixed).** "7.98% compounded over 30 days" is the per-calendar-day rate for a 10x; a 7.98% total would reach GBP 107.98.

## 12. Ablation verification

Technical, technical + fundamental (observation), technical + macro, joint, regime: reproduced. 12 paired tests: minimum bootstrap p = 0.103, all Holm-adjusted p = 1.0, largest increment GBP 1.94. At 1x/2x/3x costs the macro verdict flips for XLK and XLV and the increment grows with cost (a trade-removal effect). Threshold shifts of 0.1% change trade counts (XLF 11 to 1). Corporate fundamental ablations (value/quality/growth/revisions/catalysts) are **not estimable**: the research database holds one company plus macro; there is no point-in-time universe.

## 13. Defects

Severity counts: CRITICAL 0, HIGH 10, MEDIUM 24, LOW 27. Status: 58 fixed, 3 open by design.

| ID | Sev | Defect | Fix |
|---|---|---|---|
| PIT-08 / REAL-06 / UI-06 / UI-07 | HIGH | Free-form timestamps: host-timezone abbreviation parsing, `dd/mm` read as `mm/dd`, 2-digit years | `utc()` strict ISO-8601 with `Z`/offset; `iso_date()` for periods |
| REAL-01 | HIGH | Same metric changed definition between versions (WMT NetIncomeLoss vs ProfitLoss; mixed revenue tags): TTM 2.5% above the 10-K | TTM from published totals; mixed tag definitions withheld (`MIXED_TAG_DEFINITIONS`) |
| REAL-02 | HIGH | TTM at a fiscal year end missed the 10-K total (XOM -1.35%, KO -1.8%, AMZN FY2012 -21%) | TTM = FY + current YTD - prior YTD; equals the reported FY at year end |
| REAL-04 | HIGH | `INSERT OR IGNORE` silently dropped rows with null `period_start`/`currency` while reporting success | `ON CONFLICT(id) DO NOTHING`; nulls stored as "not applicable"; constraint violations raise |
| CALC-03 | HIGH | One split reported by two vendors applied twice (unwired helper) | de-duplicated by effective instant; conflicting ratios raise |
| DEC-01 to 04 | HIGH | Extractor tagged denied, hypothetical, other-company and prior-period statements; sentence splitter detached negations at `Inc.`/decimals | broadened cues, year-anchor rule, abbreviation/decimal-aware splitter, third-party cues |
| BT-07 | HIGH | Stage 5 headline claim not reproducible (section 11) | README retraction |
| PIT-02 / MAC-01 | MEDIUM | Equal-timestamp event versions resolved by hash order (older revision could win) | numeric-aware revision order; tie flagged `conflicting_versions` |
| PIT-07 / UI-02 | MEDIUM | Overflow produced NaN features and a dropped HTTP connection | finite guard; unserialisable payload becomes a JSON error |
| REAL-03 | MEDIUM | NEE revenue tag unmapped; snapshot a year behind but served AVAILABLE | tag mapped; snapshot is STALE when later annual flows are public |
| REAL-05 / MAC-08 / DEC-06 | MEDIUM | Malformed input raised TypeError/KeyError/OverflowError; one bad row poisoned a company | controlled `ValueError`; optional per-row quarantine (`rejects`) |
| REAL-07 | MEDIUM | Two FX sources for one date: silently the first | conflict raises |
| CALC-01 | MEDIUM | Winsorisation ineffective for 100 or fewer names | also clip at 5 robust sigma (median and MAD) |
| CALC-04 | MEDIUM | `valuation_features` accepted bool/inf, leaked TypeError | same input contract as `accounting_features` |
| CALC-05 | MEDIUM | Own-history percentile mixed quarterly, YTD, annual rows | compare like durations |
| CALC-06 | MEDIUM | TTM had no freshness gate (JPM TTM ended 2014 beside a current annual block) | `age_days`; STALE past 550 days |
| MAC-02 / DEC-11 | MEDIUM | `add_event` validated nothing; one bad or wildcard record broke every later decision | non-mapping payload rejected at write; malformed earnings flagged at read |
| MAC-03 | MEDIUM | Two insiders' same-day purchases superseded each other | `metadata.transaction_id` keeps them apart (amendments reuse the id) |
| MAC-05 / DEC-10 | MEDIUM | Re-fetching an unchanged document reset catalyst age and decay | first public copy governs |
| DEC-05 | MEDIUM | `channel` ignored: first-person "we" in analyst/social text attributed to the company | "we" patterns only for official channels |
| UI-01 / PIT-04 | MEDIUM | Blank `as_of=` silently meant "now" over HTTP | `keep_blank_values`; blank is a 400 |
| UI-03 | MEDIUM | One ambiguous symbol made the whole watchlist a 400 | per-symbol isolation, `AMBIGUOUS_SOURCE` row |
| UI-11 | MEDIUM | Stale snapshots were scored and ranked | score null, `STALE_FUNDAMENTALS` |
| BT-01 / BT-03 | MEDIUM | Misworded 10x rate; garbled delisting date meant "member forever" | document corrected; unparseable `end` raises |
| BT-02 | MEDIUM | Stage 5 features use the decision-date close (same as fill) | **open**, docstring corrected |
| PIT-01 | LOW | `INSERT OR REPLACE` bypasses the append-only trigger | **open**: SQLite cannot block it without breaking idempotent re-ingest; protect the file |
| BT-04 | LOW | 10-day time exit slips one session across US spring DST (2 of 190 reported exits) | **open**: the GPU market sweep compares canonical replay to its kernel at 1e-6 and shares this engine; fix both together on a new results root |
| PIT-03 / MAC-07 | LOW | NaN staleness thresholds disabled the gate | validated finite and positive |
| PIT-05 | LOW | Re-importing a re-formatted JSON file duplicated rows | file hash excluded from the identity |
| PIT-06 | LOW | Raw JSON bytes varied per process (unsorted set) | sorted |
| REAL-08 / MAC-06 | LOW | Transient bad 200 body cached forever | validated before caching or archiving |
| REAL-09 | LOW | JSON `true` became 1.0 | rejected as non-numeric |
| CALC-02 / MAC-04 | LOW | Float-noise ties raised FALLING_MARGINS / RISING | 1e-12 tolerance |
| DEC-07 to 09 | LOW | Unknown annotation keys persisted; hard-risk reason masked by the gate; veto explanation contradictory | key allowlist; reason kept; explanation fixed |
| UI-04, 05, 08, 09, 10, 12 | LOW | `/health` OK on corrupt DB; `::1` unusable; unvalidated symbol echoed; `-0.00` shown; silent empty chart; markdown echoed in errors | all fixed |
| BT-05 / BT-06 | LOW | Journal rounded to 10 decimals; calendar-day purge for a session horizon | exact JSON; 35-day purge |

Feature version is now `fundamentals-v4-audited`. Rebuild normalised data into a **new** database from archived raw files; do not reuse an old journal under a changed configuration.

## 14. Regression tests added

`tests/qa_reaudit/` (75 files, about 19,700 lines): 45 test files, 5,988 passing, 4 intentional strict-xfails (BT-02, BT-04, BT-07, PIT-01), 7 skipped parametrisations, plus 67 real-Chrome tests (64 new, 3 from the earlier audit; one of them once failed under heavy parallel load and passes alone). Prefixes: `pit` (1,111), `real` (861), `calc` (2,130), `mac` (829), `dec` (440), `ui` (226 + 64 browser), `bt` (381), `lead` (22). The original 126 fundamental tests and the 534 other non-GPU repository tests still pass.

Run: `.venv/bin/python -m pytest -p no:cacheprovider -q -o addopts="" tests/qa_reaudit`. Files that need cached SEC/ALFRED data or price history under `results/` (git-ignored) run locally; `.github/workflows/fundamental-audit.yml` runs the self-contained subset (added locally, not yet executed on GitHub).

Evidence: `results/qa_reaudit_20261001/` (reconciliation CSVs, 188-row UI table, claim table, robustness and placebo outputs, screenshots).

## 15. Remaining limitations

- **No predictive edge is established.** Corporate-factor ablations are not estimable; one company in the research database.
- **Coverage.** SEC tag gaps leave ROIC and net debt/EBITDA unknown for most real companies (capex `PaymentsToAcquireProductiveAssets`, debt tags, continuing-operations cash flow). EPS TTM, `eps_yoy`, `diluted_shares_yoy` and DILUTION are always unknown for SEC data because the adapter writes `split_basis` while the engine requires an explicit, verified `share_basis` (a deliberate fail-safe), so TTM status is never AVAILABLE on real data.
- **Provider lag.** companyfacts trailed submissions for 5 of 17 cached issuers; nothing flags "filing known, facts not ingested".
- **Not implemented:** yield curve and macro composite, revenue surprise, typed corporate-event extraction, cross-source event de-duplication, short interest, Form 4 ingestion, a live estimate archive, point-in-time index membership, delisting returns, spin-offs, a 0 to 100 composite score.
- **Operational.** `observed_at` is not used by `snapshot`/`decide`: a live decision re-run after a backfill differs and only `data_snapshot_id` shows it. `vintage_verified=True` is a caller assertion that backdates availability. A reused source label collapses documents to the latest. Host header is not validated on the loopback API.
- **Paper engine.** Daily-bar loss limits cannot veto an entry at default settings (day and week baselines are flat at the entry bar); `PaperEngine` has no currency, so "GBP 100" is a unit.
- **Left for the owner.** `test_QA_MAC_03_two_insiders_same_trade_date_both_counted` in `tests/qa_reaudit/test_mac_estimates_insiders.py` still carries its strict-xfail decorator although the defect is fixed (the test now passes, so the suite reports one strict-XPASS failure). Delete that single decorator.

## Final verification matrix

PASS means the implemented, tested scope passed after repairs. FAIL includes missing functionality, not only wrong arithmetic. Software correctness is separate from evidence of an edge.

| Requirement | Verdict | Scope / qualification |
|---|---|---|
| Raw data accuracy | PASS | 338 primary-document values, 0 FAIL, 14 coverage GAPs; companyfacts can lag filings |
| Unit normalisation | PASS | thousands/millions/billions in four currencies; ambiguous text rejected |
| Currency handling | PASS | four currencies synthetic, USD/JPY/GBP real; FX is a helper, conflicts now fail |
| Point-in-time financials | PASS | exact to the nanosecond; 320-seed differential; truncated-database identity |
| Restatement handling | PASS | latest known version per time; history kept |
| Revenue calculations | PASS | |
| FCF calculations | PASS | OCF minus positive CapEx exactly once |
| Margins | PASS | |
| ROE | PASS | average equity |
| ROIC | PASS | documented NOPAT proxy; unknown for most real filers (tag coverage) |
| Debt metrics | PASS | missing stays missing; real coverage poor |
| Valuation metrics | PASS | library only, not served by API or page |
| TTM calculations | PASS | published-totals form; EPS never available on SEC data |
| Sector normalisation | PASS | helper on a caller-supplied universe |
| Fundamental scoring | PASS | frozen ridge; no 0 to 100 composite exists |
| Confidence calculation | PASS | mechanics only; shipped artifact is degenerate |
| Earnings surprise | PASS | EPS only |
| Analyst revisions | PASS | helper; no feed |
| Insider data timing | PASS | synthetic; needs `transaction_id` |
| Macro timing | PASS | conservative vintage day + 1 |
| Macro regime | PASS | literal sign regimes only |
| Corporate events | PASS | single source; no cross-source de-duplication |
| News/NLP extraction | PASS | narrow literal grammar; precision 0.94, recall 0.70 on an adversarial corpus tuned after viewing it |
| No future-data leakage detected | PASS | audited paths |
| No restatement leakage detected | PASS | |
| No analyst-estimate leakage detected | PASS | helper not wired |
| No macro look-ahead detected | PASS | vintage data |
| Fundamental + technical integration | PASS | |
| RiskEngine integration | PASS | hard limits win at score 1,000,000 |
| Database/API/UI reconciliation | PASS | 188 rows, 0 mismatches |
| Historical replay | PASS | |
| Backtest reproducible | PASS | bit-identical |
| Paper-trading E2E | PASS | deterministic fixture |
| Fundamentals safe for research/paper trading | PASS | observation-only default, no approved model |
| Real-money readiness | **FAIL** | no edge established; survivorship-biased universe; broker reconciliation not qualified |
| Strategy has a genuine predictive edge | **FAIL** | claims withdrawn (section 11) |
