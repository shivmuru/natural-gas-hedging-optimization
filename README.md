# Natural Gas Winter Hedging Optimization

**Team 28** · Shivani Murugapiran, Jake Deleray, James Wu, Yingshan Zhang, Jiayue Du

On October 1, a U.S. gas utility has to decide **what share of each winter month's expected demand (November to February) to lock in at NYMEX futures prices**. The rest of its demand is bought at the Henry Hub spot price. Homes and businesses must be heated whatever gas costs, so the utility can't simply buy less when prices spike. The goal is a hedge that **keeps the expected winter cost low without leaving the utility badly exposed in a high-price winter**.

`gurobi.py` builds and solves this as a scenario-based linear program in Gurobi, for two service territories with very different sizes: **Texas** (the largest U.S. gas consumer) and **Vermont** (the smallest in the contiguous U.S.).

---

## Quick start

```bash
pip install -r requirements.txt          # gurobipy ships with a free size-limited license, enough for this model
python gurobi.py                         # Texas and Vermont: decide on Oct 2 2023, backtest against actual winter 2023-24
```

On its first run, the script downloads two things and caches them in `data/`:
- NOAA CPC daily population-weighted heating degree days for each year from 1997 to 2024 (`data/hdd/`)
- The EIA state consumption workbooks for TX and VT (`NG_CONS_SUM_DCU_STX_M.xls`, `NG_CONS_SUM_DCU_SVT_M.xls`)

**Prefer Jupyter?** Open `gurobi.ipynb` from the repo root, change the *Settings* cell (it holds the same options as the command-line flags below), and choose *Run All*. The notebook walks through each part of the model (prices, weather, demand, GPR, the Gurobi LP, reporting) and shows the result tables and charts inline.

Common options:

```bash
python gurobi.py --futures 3.10 3.55 3.90 3.70     # Oct 1 2026 CME settlements, Nov Dec Jan Feb ($/MMBtu)
python gurobi.py --decision-year 2025 --futures NOV DEC JAN FEB   # backtest winter 2025-26 (Oct 1 2025 settlements)
python gurobi.py --decision-year 2022               # backtest winter 2022-23 (futures taken from the EIA data)
python gurobi.py --states VT --theta 3 --cmax-factor 0.97
python gurobi.py --lower 0.2 0.2 0.2 0.2 --upper 0.8 0.8 0.8 0.8   # policy hedge band
python gurobi.py --no-cmax                          # drop the high-price cost constraint
python gurobi.py --company "TX=CENTERPOINT ENERGY ENTEX"
python gurobi.py --hdd-csv my_hdd.csv --no-download # offline: columns state,date,hdd
python gurobi.py --help
```

> **Important: the futures strip.** EIA's NYMEX series ends on April 5, 2024, so the decision-date prices F_m for October 1, 2026 aren't in the data. Pass them with `--futures`. If you leave it out, the script uses the latest October strip in the file (Oct 2, 2023) and prints a warning.

---

## Backtesting a past winter

`--decision-year` (or `decision_year` in the notebook) sets the October 1 on which the hedge is decided. When that winter is already over, the model is solved using **only the information available at the time**, and the resulting hedge is then scored against what actually happened.

- **No look-ahead.** The scenarios, the demand regression and the high-price set H use only winters that ended before the decision. For a 2022 decision, that means 1997-98 through 2021-22. The GPR indicator uses September of the decision year, and the utility's share uses the 5 years of EIA-176 data before it.
- **Realized cost.** Each strategy is costed as Σ_m [h_m D_m F_m + (D_actual,m − h_m D_m) · P_actual,m]:
  - P_actual is the winter's actual monthly Henry Hub average.
  - D_actual is the utility's share of actual EIA residential + commercial consumption. If EIA hasn't published it yet, the regression at actual HDD is used instead.
- **Benchmarks.** The model's hedge is compared with fixed 0/25/50/75/100% hedges and a *perfect-hindsight* hedge (U_m in months where spot ended above futures, L_m otherwise).
- **Outputs.**
  - `backtest_months.csv`: futures, actual spot, forecast vs. actual demand, and the dollar gain or loss from each month's hedge.
  - `backtest_strategies.csv`: realized cost and savings vs. no hedge for each strategy.
  - `backtest_costs.png`: a chart of the strategies' realized costs.

**Winter 2025-26.** The actual Henry Hub monthly averages in the data are Nov $3.79, Dec $4.26, Jan $7.72 and Feb $3.62. The NYMEX settlements from October 1, 2025 are not in the EIA file, so a 2025 backtest needs them passed with `--futures`. For decision years 1998 to 2023, the futures come from the data automatically.

---

## Model

### Sets
| Set | Description |
|---|---|
| M = {Nov, Dec, Jan, Feb} | delivery months, indexed by *m* |
| S = {1997-98, …, 2023-24} | 27 historical winters (scenarios), indexed by *s* |
| H ⊂ S | high-price winters (top quartile, which is 7 winters) |
| R = {TX, VT} | service territories; the model is solved separately for each |

### Parameters
| Symbol | Meaning | Units | Source / construction |
|---|---|---|---|
| F_m | NYMEX futures for month *m* on the decision date | $/MMBtu | CME (`--futures`) |
| F_sm | NYMEX futures for month *m* on the first trading day of October of winter *s* | $/MMBtu | EIA `NG_PRI_FUT_S1_D` (contracts 1–4 = Nov–Feb) |
| P_sm | monthly average of daily Henry Hub spot prices | $/MMBtu | EIA `RNGWHHDd` |
| P̃_sm | F_m · P_sm / F_sm, which applies each past winter's spot "surprise" to today's strip | $/MMBtu | derived |
| HDD_sm | sum of daily population-weighted HDD in month *m* of winter *s* | °F-days | NOAA CPC `StatesCONUS.Heating.txt` |
| D_m | forecast utility demand at normal HDD (the 27-winter mean) | MMBtu | regression |
| D_sm | utility demand at winter *s*'s actual HDD | MMBtu | regression |
| π_s | scenario probability: 1/27, or H × θ then renormalized when g = 1 | – | assumption + GPR |
| g | 1 if the GPR for September of the decision year is at or above the 90th percentile of monthly GPR since 1985 | binary | Caldara & Iacoviello GPR |
| θ | weight multiplier on H when g = 1 (default 2) | – | assumption |
| L_m, U_m | policy bounds on the hedge share (default 0 and 1) | – | assumption |
| C_max | cap on the probability-weighted average cost over H | $ | policy (default = H-average cost of a 50% hedge) |
| c_s(h) | total Nov–Feb procurement cost in scenario *s* for hedge h (defined below) | $ | cost expression, computed from the parameters above and h |

**Scenario cost c_s(h).** For each winter s ∈ S, the cost of hedge h is

$$
c_s(h) = \sum_{m\in M}\Big[h_m D_m F_m + (D_{sm} - h_m D_m)\,\tilde P_{sm}\Big]
$$

This is the hedged volume bought at the futures price plus the rest of that winter's demand bought at the rescaled spot price. If a winter turns out mild and D_sm < h_m D_m, the spot term goes negative: the surplus hedged gas is resold at that month's spot price.

c_s is **not a decision variable**. It is fully determined by the hedge shares h_m, so it is written as a cost expression rather than with its own equality constraint. It is listed with the parameters because everything in it except h_m is fixed data. Unlike a true parameter, though, its value changes with h, so it is written c_s(h). For a fixed h, c_s is linear in h, and so the model is a linear program.

**Demand.** For each month, an OLS fit `StateDemand_sm = a_m + b_m · HDD_sm` is estimated on state residential + commercial consumption. EIA reports it in MMcf, which the script converts at 1,036 MMBtu/MMcf. The fit is then scaled to one utility by its share of state R+C deliveries in EIA-176, averaged over 2020–2024. The default utilities are **Atmos Energy** (TX) and **Vermont Gas Systems** (VT; it is effectively the only gas utility in the state). Pairing each winter's weather with that same winter's prices keeps the "cold winter, high price" correlation in the scenarios.

### Decision variables
- h_m ∈ [L_m, U_m], m ∈ M: the share of forecast demand D_m hedged with futures for month *m*. These four values are the only decisions.

### Formulation

$$
\min_{h}\ \sum_{s\in S} \pi_s\, c_s(h)
$$

subject to

$$
L_m \le h_m \le U_m \quad \forall m\in M \qquad \text{(hedge policy band; } U_m\le 1 \Rightarrow \text{hedged} \le \text{forecast)}
$$

$$
\frac{\sum_{s\in H}\pi_s\, c_s(h)}{\sum_{s\in H}\pi_s} \le C_{max} \qquad \text{(high-price winter cost limit)}
$$

*Implementation:* `gurobi.py` builds this model in collected form, with the four h_m as its only variables and one linear constraint:

min Σ_m (F_m − P̄_m) D_m h_m + K  s.t.  L_m ≤ h_m ≤ U_m,  Σ_m (F_m − P̄^H_m) D_m h_m + K^H ≤ C_max

Here π̃_s = π_s / Σ_{s∈H} π_s (probability of winter s given a high-price winter), P̄_m = Σ_{s∈S} π_s P̃_sm and P̄^H_m = Σ_{s∈H} π̃_s P̃_sm are the expected rescaled spot prices, and K, K^H are the expected no-hedge costs over all winters and over H. Expanding c_s(h) and collecting terms in h gives exactly this form. K does not change the optimal hedge; it is set as the objective constant so the reported objective is the expected cost. Each winter's cost is computed from the optimal h after solving (`cost_s_optimal` in `parameters_scenario.csv`).

### Geopolitical indicator
The monthly Geopolitical Risk (GPR) index of Caldara & Iacoviello is in `data/data_gpr_export.xls`. The code uses it in four steps:

1. **Threshold.** It takes every monthly GPR value from January 1985 onward and computes the 90th percentile, which is **148.1**.
2. **Signal.** It reads the GPR for September of the decision year, the last full month before the October 1 decision. If that value is at or above the threshold, **g = 1**; otherwise g = 0. September 2026 is **155.9** (g = 1). September 2025 is **126.7**, below the threshold (g = 0), so the 2025 backtest uses equal weights.
3. **Reweighting.** If g = 1, each high-price winter in H has its weight multiplied by θ (default 2), and all weights are renormalized to sum to 1. With 7 of 27 winters in H and θ = 2, the H winters go from 26% of the total probability to 41%. If g = 0, every winter keeps weight 1/|S|.
4. **Effect on the model.** g only changes the probabilities π_s. It does not change prices, demand or C_max's definition. Because π_s appears in both the objective (expected cost) and the C_max constraint (H-average cost), a higher weight on expensive winters makes hedging look more valuable on average. That is why θ ≥ 3 pushes the model to hedge more even without the cost limit.

The data, parameters and constraints are otherwise the same whether g is 0 or 1. Use `--force-g 0` or `--force-g 1` (or `force_g` in the notebook) to override the rule, and `--theta` to change the multiplier.

---

## Outputs

Everything is printed to the console and written to `outputs/<STATE>_<decision year>/` (e.g. `outputs/TX_2026/`):

| File | Contents |
|---|---|
| `optimal_hedge.csv` | h_m and hedged volume (MMBtu) |
| `strategy_comparison.csv` | 0/25/50/75/100% and optimal hedges: expected cost, H-average cost, worst winter, std. dev., $/MMBtu |
| `sensitivity_expected_cost.csv` / `sensitivity_hedge_shares.csv` | matrix of θ ∈ {1, 1.5, 2, 3, 4} × C_max factor ∈ {0.90 … 1.10, no limit} |
| `parameters_month.csv`, `parameters_scenario.csv` | every model parameter (F, P, P̃, D, HDD, π, H membership, optimal c_s) |
| `strategy_frontier.png`, `sensitivity_heatmap.png` | expected cost vs. high-price exposure, and a heatmap of the sensitivity grid |
| `backtest_*.csv`, `backtest_costs.png` | only for past winters: realized cost of each strategy (see *Backtesting a past winter*) |

The console also reports the regression coefficients and R² for each month, along with the **shadow price of C_max**: how many dollars of expected cost each extra dollar of allowed high-price cost saves.

---

## Data

| # | File | Description | Source |
|---|---|---|---|
| 1 | `data/RNGWHHDd.xls` | Henry Hub daily spot price, Jan 1997 to Oct 2026 | [EIA](https://www.eia.gov/dnav/ng/hist/rngwhhdd.htm) |
| 2 | `NG_CONS_SUM_DCU_STX_M.xls`, `NG_CONS_SUM_DCU_SVT_M.xls` (downloaded on first run) | Monthly residential and commercial gas consumption for Texas and Vermont, used for the demand regression. The repo also contains the New York file (`…SNY_M.xls`), which is used only if you run with `--states NY`. | EIA: [Texas](https://www.eia.gov/dnav/ng/ng_cons_sum_dcu_STX_m.htm), [Vermont](https://www.eia.gov/dnav/ng/ng_cons_sum_dcu_SVT_m.htm) |
| 3 | `data/eia176_company_deliveries.xlsx` | EIA-176 annual company-level deliveries by state, 1997–2024 (Mcf) | [EIA-176 query](https://www.eia.gov/naturalgas/ngqs/) |
| 4 | `data/hdd/` (downloaded) | Daily population-weighted HDD by state | [NOAA CPC](https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/) |
| 5 | `data/NG_PRI_FUT_S1_D.xls` | NYMEX futures, contracts 1–4, daily, Dec 1993 to Apr 5 2024 | [EIA](https://www.eia.gov/dnav/ng/ng_pri_fut_s1_d.htm) |
| 6 | `data/data_gpr_export.xls` | Geopolitical Risk Index, monthly | [Caldara & Iacoviello](https://www.matteoiacoviello.com/gpr.htm) |

---

## Modeling assumptions and notes

- **Price taker.** The utility's purchases don't move futures or spot prices.
- **Static hedge.** All four hedge shares are set on October 1 and never rebalanced.
- **Henry Hub as the local price.** Basis differentials, transaction costs and margin calls are ignored.
- **Linear demand in HDD.** The utility serves a fixed share of state R+C demand.
- **Monthly averaging.** Daily spot prices are averaged into monthly means.
- **GPR as a proxy.** GPR stands in for gas-market risk and enters only through the binary indicator g.
- **Definition of H (a deviation from Milestone 3).** H is ranked on the *rescaled* winter spot P̃ by default, i.e. the winters where spot came in furthest above the October futures. The costs c_s are computed with P̃, so ranking H on raw spot (the 2000s, when all prices were high) picks winters that weren't actually expensive *relative to the hedge*. That makes C_max never bind. `--high-basis raw` restores the Milestone 3 definition.
- **Expect corner solutions.** The objective is linear, so without the C_max constraint the optimum sits at a bound. Historically, Henry Hub settled about 2–11% below the October futures on average (Nov 0.98, Dec 0.93, Jan/Feb 0.89), so pure expected-cost minimization leans toward hedging nothing. The C_max constraint and θ are what produce interior, risk-aware hedges, which is why the sensitivity matrix is the main result.
