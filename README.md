# YES BANK Stock Analysis — EDA & Advanced Machine Learning

An end-to-end educational data-science project that studies the supplied monthly YES BANK OHLC history, then evaluates whether the available historical information can support reliable multi-horizon stock-price forecasts.

The project is deliberately split into two stages:

1. **Exploratory Data Analysis (EDA):** validate and understand the dataset, describe historical price behaviour, and investigate distributions, relationships, trends, and volatility.
2. **Advanced Machine Learning:** construct leakage-aware forecasting targets and features, compare machine-learning models with simple and statistical benchmarks, evaluate forecasts chronologically, and examine uncertainty and a small illustrative decision strategy.

> **Important:** This is an academic/research exercise, not investment advice or a live trading system. The source CSV ends in **November 2020**. Outputs are historical demonstrations and must not be interpreted as current YES BANK forecasts.

---

## Table of contents

- [Project objectives](#project-objectives)
- [Dataset](#dataset)
- [Repository contents](#repository-contents)
- [Stage 1 — EDA](#stage-1--exploratory-data-analysis-eda)
- [Stage 2 — Advanced ML](#stage-2--advanced-machine-learning)
- [Selected experimental results](#selected-experimental-results)
- [How to run](#how-to-run)
- [Reproducibility and validation](#reproducibility-and-validation)
- [Limitations and responsible interpretation](#limitations-and-responsible-interpretation)
- [Possible future improvements](#possible-future-improvements)
- [Author and contribution](#author-and-contribution)

---

## Project objectives

### Business / analytical objective

Use the historical monthly Open, High, Low, and Close values to:

- establish the quality and structure of the supplied data;
- describe long-term price phases and monthly return behaviour;
- analyse price ranges, moving averages, correlations, and volatility;
- engineer information that would have been available at a forecast origin;
- evaluate 1-, 3-, and 6-month-ahead closing-price forecasts;
- compare machine-learning forecasts against transparent naïve and statistical benchmarks;
- quantify forecast errors and inspect whether model behaviour changes across periods or volatility regimes.

The project does **not** assume that an accurate historical fit implies a profitable or deployable trading strategy.

## Dataset

**File:** `data_YesBank_StockPrices.csv`

| Field | Description | Use |
|---|---|---|
| `Date` | Monthly observation label (for example, `Jul-05`) | Time index |
| `Open` | Opening price for the month | EDA and model feature |
| `High` | Highest observed price during the month | EDA and model feature |
| `Low` | Lowest observed price during the month | EDA and model feature |
| `Close` | Closing price for the month | Main variable of interest and forecast basis |

### Dataset facts

| Property | Value |
|---|---:|
| Observations | 185 monthly rows |
| Source columns | 5 |
| Period | July 2005 – November 2020 |
| Missing values in original file | 0 |
| Duplicate rows in original file | 0 |
| Duplicate monthly dates | 0 |
| OHLC ordering checks | Passed |

The CSV does not contain an explicit currency metadata field. The notebooks use rupee (`₹`) labels in line with the assignment context; confirm the source currency before reusing the work externally.

---

## Repository contents

```text
YES_BANK_Stock_Analysis_Project/
├── README.md
├── YES_BANK_EDA_Notebook.ipynb
├── YES_BANK_Advanced_ML_Notebook.ipynb
├── data_YesBank_StockPrices.csv
├── YES_BANK_advanced_model_bundle.joblib
├── YES_BANK_feature_schema.json
├── MODEL_CARD.md
├── results_summary.json
└── requirements.txt
```

### Main deliverables

- **`YES_BANK_EDA_Notebook.ipynb`** — complete EDA notebook, including data checks, feature engineering, 22 chart sections, interpretations, business recommendations, and conclusion.
- **`YES_BANK_Advanced_ML_Notebook.ipynb`** — advanced multi-horizon ML study with walk-forward evaluation, benchmarks, diagnostics, strategy sensitivity, and model persistence.
- **`YES_BANK_advanced_model_bundle.joblib`** — serialized fitted model bundle produced by the ML workflow.
- **`YES_BANK_feature_schema.json`** — feature contract and model metadata used by the saved bundle.
- **`MODEL_CARD.md`** — summary of intended use, model evaluation, risks, and limitations.
- **`results_summary.json`** — machine-readable summary of principal experimental results.

---

## Stage 1 — Exploratory Data Analysis (EDA)

Notebook: **`YES_BANK_EDA_Notebook.ipynb`**

### 1. Data understanding and validation

The notebook:

- loads the source CSV and checks the expected columns;
- inspects sample records, shape, data types, summary statistics, and unique values;
- checks missing values, duplicate rows, and duplicate dates;
- parses `Date` as a datetime value and sorts rows chronologically;
- validates positive prices and the expected OHLC relationship (`Low` should not exceed `Open` or `Close`, and `High` should not be below either).

The raw source contains 185 rows and 5 columns, with no missing values or duplicate rows. The OHLC consistency checks pass for the supplied data.

### 2. Feature engineering

The EDA notebook adds descriptive features such as:

- calendar features: `Year`, `Month`, and `Month_Name`;
- absolute trading range: `High - Low`;
- opening-to-closing price change and percentage change;
- month-over-month closing return and absolute return;
- percentage trading range;
- 6-month and 12-month moving averages;
- 12-month rolling return volatility.

Initial return and rolling-window values are naturally undefined until the required historical observations exist; they are not treated as raw-data errors.

### 3. UBM visualization framework

The notebook contains **22 chart sections**, organised as required by the UBM approach:

| Analysis group | Chart sections | Focus |
|---|---:|---|
| **U — Univariate** | 6 | Closing-price and OHLC distributions, returns, trading range, rolling volatility |
| **B — Bivariate** | 12 | Price over time, annual comparisons, OHLC relationships, monthly patterns, moving averages, returns and volatility |
| **M — Multivariate** | 4 | OHLC correlation, pair plot, year-by-month price heatmap, engineered-feature correlation |

Each chart section includes visualization code and written interpretation/business context in the notebook.

### 4. Main EDA themes

- The closing-price distribution is right-skewed.
- The average closing price reaches its highest annual level in 2017 in the supplied sample.
- The OHLC variables are very strongly positively correlated; this is expected for measurements of the same monthly price candle, but it can create redundancy in regression features.
- Monthly returns include both large positive and negative movements.
- Rolling volatility changes over time, so the series should not be assumed to have constant risk.
- The 2020 observations cover January through November only, not a complete calendar year.

The EDA notebook is descriptive. It does not train or validate a forecasting model.

---

## Stage 2 — Advanced Machine Learning

Notebook: **`YES_BANK_Advanced_ML_Notebook.ipynb`**

### 1. Forecasting formulation

The ML notebook predicts future closing-price changes at **1-, 3-, and 6-month horizons**. It models the future log return from the forecast origin:

\[
 r_{t,h} = \log(Close_{t+h}) - \log(Close_t)
\]

The predicted log return is converted back into a price estimate using the closing price known at the forecast origin.

### 2. Leakage-aware feature construction

Features are constructed only from information available at or before the forecast origin. The notebook includes lagged return and price behaviour, rolling/momentum summaries, calendar information, and OHLC-derived features. Preprocessing that needs fitting is contained within the relevant modelling pipeline.

### 3. Validation design

The workflow is time-aware rather than randomly shuffled:

- the data are split chronologically into a development period and a final holdout;
- expanding-window time-series cross-validation is used within development data;
- the cross-validation gap is horizon-aware;
- holdout forecasts are generated walk-forward;
- a training label is only used after its target month has become observable at that forecast origin.

This prevents future target information from entering earlier training folds.

### 4. Models and benchmarks

Machine-learning candidates:

- Ridge Regression;
- Elastic Net;
- Random Forest;
- Extra Trees;
- HistGradientBoosting.

Comparative benchmarks:

- persistence / naïve forecast;
- expanding historical drift;
- seasonal naïve forecast;
- ARIMA(1,1,0) statistical benchmark.

The notebook reports price-level errors in rupees alongside return-based and directional metrics, including MAE, RMSE, MAPE/sMAPE where appropriate, log-return MAE, directional accuracy, and R².

### 5. Additional experiments and diagnostics

The advanced notebook includes **30 chart/diagnostic outputs**, including:

- time-series trends, log returns, realized volatility, and drawdowns;
- return seasonality and autocorrelation checks;
- OHLC and engineered-feature correlation diagnostics;
- feature distributions and feature-ablation comparisons;
- cross-validation and final holdout model scorecards;
- actual-versus-predicted paths and predicted-versus-actual plots;
- direction-classification confusion matrix and probability calibration;
- empirical residual bands, residual distributions, Q-Q diagnostics, and residual autocorrelation;
- permutation feature importance;
- error comparison by volatility regime and target year;
- an illustrative long/flat strategy, drawdown, and transaction-cost sensitivity;
- a comparison of forecast loss against a baseline.

The notebook also saves the fitted model bundle, reloads it, and checks the reusable prediction interface.

---

## Selected experimental results

The results below are from the supplied historical CSV and the executed notebook. Values are rounded for readability; the notebook and `results_summary.json` contain the recorded values.

### Final chronological holdout — price MAE

| Forecast horizon | Selected ML model | ML MAE | Best benchmark | Benchmark MAE |
|---:|---|---:|---|---:|
| 1 month | Extra Trees | ₹24.30 | Persistence | ₹23.80 |
| 3 months | Elastic Net | ₹47.42 | Persistence | ₹43.23 |
| 6 months | Elastic Net | ₹92.20 | Persistence | ₹72.70 |

**Key finding:** None of the selected ML models beats the best benchmark by holdout price MAE. The project reports this rather than presenting model complexity as evidence of forecasting skill.

ARIMA(1,1,0) benchmark MAE values were approximately ₹24.47 (1 month), ₹45.07 (3 months), and ₹73.34 (6 months).

### Other diagnostic findings

- **Feature ablation:** In the tested 1-month Ridge cross-validation experiment, last-return-only features achieved a log-return MAE of 0.0866. The full engineered feature set scored 0.1034. More features did not improve this experiment.
- **Direction classification:** Holdout accuracy was 58.33%, below the 63.89% majority-class benchmark; ROC AUC was 0.4314. This does not demonstrate useful directional discrimination.
- **Empirical forecast-band coverage:** Holdout coverage was 63.9%, 50.0%, and 22.6% for the 1-, 3-, and 6-month horizons. The empirical bands under-cover the holdout and should not be treated as calibrated 90% prediction intervals.
- **Illustrative strategy test:** At an assumed 0.10% one-way transaction cost, the long/flat test had a cumulative return of -64.7% and maximum drawdown of -74.8%. Buy-and-hold over the same holdout had a cumulative return of -95.2%. Both lost substantial value; this small historical test does not establish a trading edge.

The ML results are evidence about this dataset and this particular evaluation design only. They are not guarantees of future performance.

---

## How to run

### Requirements

- Python 3.10 or later is recommended.
- JupyterLab or Jupyter Notebook, or VS Code with the Jupyter extension.
- Install the pinned dependencies in `requirements.txt`.

### Setup

1. Download or clone this project folder.
2. Keep both notebooks and `data_YesBank_StockPrices.csv` together in the project root.
3. Create and activate a virtual environment (recommended).
4. Install packages:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Start JupyterLab:

   ```bash
   jupyter lab
   ```

6. Run `YES_BANK_EDA_Notebook.ipynb` from the first cell to the last.
7. Run `YES_BANK_Advanced_ML_Notebook.ipynb` from the first cell to the last.

The notebooks search for the supplied CSV in their own folder or documented common paths. For a portable local setup, place the CSV beside both notebooks.

### Re-running the model

The ML notebook re-creates the experiments and model bundle. The included `.joblib` artifact was produced by the executed workflow. For reproducibility, use the pinned package versions and do not load serialized joblib files from untrusted sources.

---

## Reproducibility and validation

The distributed notebook files were inspected for saved execution error outputs:

| Notebook | Total cells | Code cells | Chart/diagnostic outputs | Saved execution errors |
|---|---:|---:|---:|---:|
| EDA | 182 | 35 | 22 chart sections | 0 |
| Advanced ML | 123 | 54 | 30 | 0 |

The ML notebook also records deterministic settings where supported and performs a model-bundle reload sanity check. Exact results may vary if package versions, data, or code are changed.

---

## Limitations and responsible interpretation

1. **Small sample:** 185 monthly observations are limited for flexible machine-learning methods and statistical testing.
2. **Old data:** The file ends in November 2020 and cannot support current 2026 market forecasts.
3. **OHLC-only inputs:** The source contains no volume, corporate actions, dividends, benchmark-index returns, fundamentals, macroeconomic variables, company announcements, or news.
4. **Regime sensitivity:** The test period is short and may reflect unusual market conditions; results can vary across different holdout windows.
5. **Weak predictive evidence:** The selected ML models do not outperform the strongest simple benchmark on the final holdout MAE.
6. **Uncalibrated forecast bands:** The empirical residual bands under-cover the final holdout, especially at longer horizons.
7. **Strategy simplifications:** The illustrative strategy does not fully model taxes, bid-ask spread, slippage, liquidity constraints, market impact, corporate actions, or realistic order execution.
8. **No causal claims:** Correlations, feature importance, and statistical tests are descriptive/predictive diagnostics, not proof of economic causation.
9. **Currency metadata:** The CSV does not explicitly state its currency. Confirm the source metadata before external publication.

**Do not use the included artifacts to place trades or make investment decisions.** They are intended for coursework, reproducible experimentation, and learning time-series model evaluation.

---

## Possible future improvements

- Obtain updated, reliable, corporate-action-adjusted price data with a clear source and license.
- Add daily or weekly observations if consistent history is available.
- Add volume, NIFTY/BANKNIFTY and sector returns, volatility indices, fundamentals, and macroeconomic variables with point-in-time alignment.
- Expand the evaluation to multiple rolling out-of-time test windows.
- Compare against stronger but still transparent statistical baselines and test whether incremental features deliver stable improvements.
- Recalibrate prediction intervals and report proper interval scores and coverage by horizon/regime.
- Add realistic transaction costs, slippage, turnover limits, and execution assumptions before any strategy research is interpreted.
- Require repeatable improvement over naïve baselines before considering deployment.

---

## Author and contribution

- **Author / Team:** Add your name and team member details.
- **Institution / Course:** Add your institution and course details.
- **GitHub repository:** Add the repository URL after publishing the project.
- **Project type:** Academic EDA and machine-learning study.

If this repository is made public, verify that the supplied dataset may be redistributed and include its original source/attribution if required.

---

## Acknowledgement

This project was developed around the user-provided YES BANK monthly stock-price CSV and the supplied EDA/ML submission templates. Findings and numerical results in this README are derived from the included notebooks and their recorded outputs.
