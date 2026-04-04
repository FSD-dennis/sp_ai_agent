# S&P 500 Regime Detection AI Agent

An autonomous quantitative research agent that ingests daily S&P 500 and sector ETF data, engineers regime features, runs HMM and clustering-based regime segmentation, evaluates results, and produces structured daily research reports — all orchestrated by a LangGraph + GPT-4o agent.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Agent                       │
│                                                         │
│  START ─► data ─► features ─► model ─► eval ─► reason  │
│                                                ↓   ↑    │
│                                         (retry?) ──┘    │
│                                                ↓        │
│                                        viz ─► report    │
│                                                ↓        │
│                                               END       │
└─────────────────────────────────────────────────────────┘
```

| Node | Module | What it does |
|------|--------|-------------|
| **data** | `src/data/` | Fetches prices + volume from Yahoo Finance, cleans & computes returns |
| **features** | `src/features/` | Builds volatility, dispersion, correlation, macro, liquidity features |
| **model** | `src/models/` | Fits HMM + KMeans/GMM on train set, predicts full sample |
| **eval** | `src/models/evaluation.py` | Per-regime stats, transition matrix, silhouette, model comparison |
| **reasoning** | `src/agent/nodes.py` | GPT-4o interprets results and writes executive summary (or deterministic fallback) |
| **viz** | `src/visualization/` | Generates 5 plot types as date-stamped PNGs |
| **report** | `src/reporting/` | Renders Jinja2 markdown daily research note |

---

## Prerequisites

- **Python 3.12+**
- **OpenAI API key** (for GPT-4o reasoning node; optional — the agent falls back to deterministic summaries without it)

---

## Setup

```bash
# 1. Clone the repository
cd sp_ai_agent

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key (optional)
```

---

## Running

### Full pipeline (historical backtest + report)

```bash
python scripts/run_pipeline.py
```

With overrides:

```bash
python scripts/run_pipeline.py --seed 123 --start-date 2015-01-01 --end-date 2025-12-31
```

### Daily incremental update

```bash
python scripts/run_daily.py
```

This clears cached data, fetches the latest bar, re-runs the full model, and generates a new daily report.

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

All parameters live in [`config/settings.yaml`](config/settings.yaml):

| Section | Key parameters |
|---------|---------------|
| **data** | `tickers` (SPY + 11 sector ETFs), `macro_tickers` (TLT, SHY), `start_date`, `end_date`, `cache_dir` |
| **features** | `volatility_windows`, `ewma_span`, `correlation_window`, `volume_shock_threshold` |
| **preprocessing** | `ffill_limit` (5), `max_missing_pct` (0.20) |
| **models** | `n_regimes` (3), HMM `n_iter`/`covariance_type`, clustering `method` (both/kmeans/gmm) |
| **split** | `train_ratio` (0.70) — strict chronological, no shuffling |
| **agent** | `model_name` (gpt-4o), `temperature`, `max_llm_calls` (3) |
| **outputs** | `models_dir`, `plots_dir`, `reports_dir` |

---

## Project Structure

```
sp_ai_agent/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.yaml              # Central configuration
├── src/
│   ├── data/
│   │   ├── fetcher.py             # Yahoo Finance retrieval + parquet cache
│   │   └── preprocessor.py        # Forward-fill, return computation
│   ├── features/
│   │   ├── builder.py             # Orchestrates all feature pipelines
│   │   ├── volatility.py          # Rolling vol (5d/21d/63d), EWMA
│   │   ├── dispersion.py          # Cross-sector return dispersion
│   │   ├── correlation.py         # Rolling avg correlation, breakdown flags
│   │   ├── macro.py               # VIX proxy, yield curve proxy
│   │   └── liquidity.py           # Volume shocks, Amihud illiquidity
│   ├── models/
│   │   ├── hmm_model.py           # Gaussian HMM (hmmlearn)
│   │   ├── clustering_model.py    # KMeans + GMM (scikit-learn)
│   │   ├── splitter.py            # Chronological + walk-forward splits
│   │   └── evaluation.py          # Regime stats, transition matrix, comparison
│   ├── agent/
│   │   ├── graph.py               # LangGraph StateGraph wiring
│   │   ├── nodes.py               # Node functions (data→features→model→eval→viz→report)
│   │   ├── state.py               # TypedDict agent state schema
│   │   └── tools.py               # @tool-decorated wrappers for agent reasoning
│   ├── reporting/
│   │   ├── generator.py           # Jinja2-based markdown report renderer
│   │   └── template.md.j2         # Daily research note template
│   ├── visualization/
│   │   └── plots.py               # 5 plot types, saved as date-stamped PNGs
│   └── utils/
│       ├── config.py              # YAML + env var loader
│       ├── logger.py              # Structured logging
│       └── seed.py                # Reproducibility (numpy, random, os)
├── outputs/
│   ├── data/                      # Cached parquet files
│   ├── models/                    # Saved model artifacts (joblib)
│   ├── plots/                     # Generated PNG visualizations
│   └── reports/                   # Daily markdown research notes
├── scripts/
│   ├── run_pipeline.py            # Full pipeline CLI entry point
│   └── run_daily.py               # Incremental daily update
└── tests/
    ├── test_fetcher.py            # Data fetching (mocked)
    ├── test_features.py           # Feature engineering (synthetic data)
    ├── test_models.py             # HMM + clustering on synthetic regimes
    └── test_splitter.py           # Chronological split correctness
```

---

## Module Explanations

### `src/data/` — Data Ingestion
- **fetcher.py**: Downloads adjusted-close prices and volume from Yahoo Finance via `yfinance`. Implements parquet caching to avoid redundant API calls and a 0.5s sleep between ticker fetches to respect rate limits.
- **preprocessor.py**: Forward-fills gaps (max 5 days), drops columns with >20% missing data, and computes simple/log returns using purely backward-looking transforms (no look-ahead).

### `src/features/` — Feature Engineering
- **volatility.py**: Rolling standard deviation (5d, 21d, 63d windows, annualized) and EWMA volatility as a lightweight GARCH proxy.
- **dispersion.py**: Cross-sectional standard deviation of sector daily returns and its rolling z-score — captures "risk-off" events when sectors diverge.
- **correlation.py**: Average pairwise rolling correlation across sectors and binary "correlation breakdown" flags when correlation drops >2σ below its rolling mean.
- **macro.py**: Realized volatility as a VIX proxy; TLT/SHY price ratio as a yield-curve slope proxy.
- **liquidity.py**: Volume z-score shocks and Amihud illiquidity ratio (|return|/volume).
- **builder.py**: Calls all sub-modules, aligns indices, and drops NaN warm-up rows to produce a clean feature matrix.

### `src/models/` — Regime Detection
- **splitter.py**: Strict chronological train/test split (70/30 default) with assertions that `train.index.max() < test.index.min()`. Also provides expanding-window walk-forward splits for robustness studies.
- **hmm_model.py**: Fits a `GaussianHMM` from `hmmlearn` with `StandardScaler` fit on train only. Supports save/load via `joblib`.
- **clustering_model.py**: Fits `KMeans` and `GaussianMixture` from scikit-learn with the same train-only scaler discipline.
- **evaluation.py**: Computes per-regime annualized return, volatility, Sharpe ratio, max drawdown, average duration, and % time. Also produces empirical transition matrices, stability metrics, silhouette scores, and side-by-side HMM vs. clustering comparison.

### `src/agent/` — LangGraph Orchestration
- **state.py**: `AgentState` TypedDict with all fields the pipeline progressively populates (data, features, model outputs, evaluation, plots, report).
- **nodes.py**: Six domain nodes (`data_node`, `feature_node`, `model_node`, `eval_node`, `viz_node`, `report_node`) plus a `reasoning_node` that calls GPT-4o to interpret results. Falls back to a deterministic summary if no API key is set.
- **graph.py**: Builds a `StateGraph` with edges: START → data → features → model → eval → reasoning → viz → report → END. Conditional edge from reasoning allows retrying the model node with different parameters.
- **tools.py**: `@tool`-decorated wrappers enabling the reasoning node to introspect available pipeline functions.

### `src/reporting/` — Report Generation
- **template.md.j2**: Jinja2 template for daily research notes — executive summary, current regime, stats tables, transition matrix, model comparison, plot references, and caveats.
- **generator.py**: Renders the template with pipeline results and saves as a date-stamped markdown file.

### `src/visualization/` — Plotting
- **plots.py**: Five plot types: regime timeline (price chart with coloured bands), feature correlation heatmap, per-regime performance bars, transition probability heatmap, and HMM vs. clustering comparison. All saved as date-stamped PNGs with `plt.close()`.

### `src/utils/` — Utilities
- **config.py**: Loads YAML configuration, resolves paths, and injects environment variables (OpenAI key).
- **logger.py**: Configures structured console logging with timestamps.
- **seed.py**: Sets `numpy`, `random`, and `PYTHONHASHSEED` for full reproducibility.

---

## Outputs

After a pipeline run, the `outputs/` directory contains:

| Directory | Contents |
|-----------|----------|
| `outputs/data/` | `prices.parquet`, `volume.parquet` — cached raw data |
| `outputs/models/` | `hmm_model.joblib`, `kmeans_model.joblib`, `gmm_model.joblib` |
| `outputs/plots/` | `regime_timeline_YYYY-MM-DD.png`, `feature_heatmap_YYYY-MM-DD.png`, etc. |
| `outputs/reports/` | `YYYY-MM-DD.md` — daily research note |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **StandardScaler fit on train only** | Prevents test-set leakage |
| **Forward-fill max 5 days** | Handles weekends/holidays without fabricating data |
| **3 regimes default** | Maps to bull / bear / transition; configurable in YAML |
| **EWMA over GARCH** | Avoids `arch` dependency; lightweight proxy sufficient for regime features |
| **Parquet caching** | Avoids re-fetching from Yahoo Finance on every run |
| **Max 3 LLM calls** | Controls OpenAI API cost; enforced in graph structure |
| **Deterministic fallback** | Agent works without an API key — produces rule-based summaries |

---

## Limitations & Future Work

- **Regime labels are arbitrary** — the model does not inherently know "bull" vs "bear"; interpretation is statistical.
- **HMM assumes Gaussian emissions** — real financial returns have fat tails and skew.
- **Clustering ignores temporal order** — only feature-space proximity is used.
- **Walk-forward evaluation** is implemented in `splitter.py` but not surfaced in the default pipeline (single 70/30 split). Add a `--walk-forward` CLI flag for production use.
- **No real-time streaming** — the daily update script is designed for cron jobs, not live market data.
- **Sector ETF coverage** — adding international ETFs or crypto would expand regime detection scope.

---

## License

Internal research use. See your organisation's policies.
