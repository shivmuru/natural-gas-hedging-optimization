"""
Natural Gas Winter Hedging Optimization (Team 28)
=================================================

On October 1 a U.S. gas utility decides what share h_m of each winter month's
forecast demand (Nov, Dec, Jan, Feb) to lock in at NYMEX futures prices. The
rest is bought at the Henry Hub spot price. The model is a two-stage scenario
LP solved with Gurobi:

    min   sum_s pi_s * c_s
    s.t.  c_s = sum_m [ h_m D_m F_m + (D_sm - h_m D_m) * Ptilde_sm ]   for all s in S
          L_m <= h_m <= U_m                                           for all m in M
          sum_{s in H} pi_s c_s / sum_{s in H} pi_s <= Cmax

Sets
    M = {Nov, Dec, Jan, Feb}          delivery months
    S = {1997-98, ..., 2023-24}       historical winters (scenarios)
    H ⊂ S                             top-quartile winters by average (rescaled) Henry Hub spot
    R = {TX, VT}                      service territories (one model per state)

Parameters (all built from the files in ./data and NOAA CPC degree days)
    F_m        futures price for month m on the decision date (Oct 1, 2026)
    F_sm       futures price for month m on the first trading day of October of winter s
    P_sm       monthly average Henry Hub spot price in month m of winter s
    Ptilde_sm  = F_m * P_sm / F_sm (historical spot surprise applied to today's strip)
    D_m        forecast utility demand at normal HDD (regression)
    D_sm       utility demand at winter s's actual HDD (regression)
    pi_s       scenario probability; high-price winters get weight * theta if g = 1
    g          geopolitical indicator: 1 if Sept-2026 GPR >= 90th pct of monthly GPR since 1985
    Cmax       limit on probability-weighted average cost over the high-price winters H

Run:
    python gurobi.py                      # TX and VT, baseline settings
    python gurobi.py --states VT --theta 3 --cmax-factor 0.97
    python gurobi.py --futures 3.10 3.55 3.90 3.70   # Oct 1 2026 CME settlements
    python gurobi.py --help
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover
    sys.exit("gurobipy is required:  pip install gurobipy")

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

MONTHS = ["Nov", "Dec", "Jan", "Feb"]
# Calendar month and year offset (relative to the winter's start year) for each delivery month
MONTH_CAL = {"Nov": (11, 0), "Dec": (12, 0), "Jan": (1, 1), "Feb": (2, 1)}
# On the first trading day of October, NYMEX contracts 1-4 are Nov, Dec, Jan, Feb
FUTURES_COL = {"Nov": "C1", "Dec": "C2", "Jan": "C3", "Feb": "C4"}

FIRST_WINTER = 1997  # 1997-98
LAST_WINTER = 2023   # 2023-24 (last October with futures quotes in the EIA file)

HEAT_CONTENT = 1.036  # MMBtu per Mcf (≈1,036 Btu/cf), so 1 MMcf = 1,036 MMBtu

STATE_NAMES = {"TX": "Texas", "VT": "Vermont", "NY": "New York"}
DEFAULT_COMPANY = {
    "TX": "ATMOS ENERGY CORPORATION",
    "VT": "VERMONT GAS SYSTEMS INC",
    "NY": "CONSOLIDATED EDISON NEW YORK INC",
}

EIA_CONS_URL = "https://www.eia.gov/dnav/ng/xls/NG_CONS_SUM_DCU_S{st}_M.xls"
NOAA_HDD_URL = (
    "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/"
    "{year}/StatesCONUS.Heating.txt"
)


@dataclass
class Config:
    states: list[str] = field(default_factory=lambda: ["TX", "VT"])
    theta: float = 2.0                     # weight multiplier on H when g = 1
    gpr_percentile: float = 0.90           # GPR threshold percentile
    gpr_month: str = "2026-09"             # month whose GPR sets g
    force_g: int | None = None             # override g (0/1)
    high_quantile: float = 0.75            # H = winters with avg spot >= this quantile
    high_basis: str = "rescaled"           # define H on rescaled spot (default) or "raw" spot
    cmax_factor: float | None = 1.0        # Cmax = factor * H-avg cost of the 50% hedge
    cmax_abs: float | None = None          # absolute Cmax in $ (overrides factor)
    lower: list[float] = field(default_factory=lambda: [0.0] * 4)
    upper: list[float] = field(default_factory=lambda: [1.0] * 4)
    futures: list[float] | None = None     # F_m for Nov, Dec, Jan, Feb on Oct 1 2026
    share_years: list[int] = field(default_factory=lambda: list(range(2020, 2025)))
    companies: dict[str, str] = field(default_factory=dict)
    scenario_demand: str = "fitted"        # "fitted" (regression) or "actual"
    data_dir: Path = DATA_DIR
    out_dir: Path = OUT_DIR
    hdd_csv: Path | None = None
    download: bool = True
    sensitivity: bool = True
    plots: bool = True
    verbose: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (hedging-model)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def _read_eia_sheet(path: Path, sheet: str = "Data 1") -> pd.DataFrame:
    """Read an EIA dnav workbook sheet: row 2 = source keys, row 3 = labels, data after."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    keys = raw.iloc[1].tolist()
    df = raw.iloc[3:].copy()
    df.columns = ["Date"] + [str(k) for k in keys[1:]]
    df["Date"] = pd.to_datetime(df["Date"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def winter_label(y: int) -> str:
    return f"{y}-{str(y + 1)[-2:]}"


def winter_month_date(y: int, m: str) -> tuple[int, int]:
    cal, off = MONTH_CAL[m]
    return y + off, cal


# ---------------------------------------------------------------------------
# 1. Prices
# ---------------------------------------------------------------------------
def load_spot(cfg: Config) -> pd.DataFrame:
    """P_sm: monthly average of daily Henry Hub spot prices, indexed by (winter, month)."""
    df = _read_eia_sheet(cfg.data_dir / "RNGWHHDd.xls")
    df = df.rename(columns={"RNGWHHD": "spot"}).dropna(subset=["spot"])
    monthly = df.groupby([df.Date.dt.year, df.Date.dt.month])["spot"].mean()
    rows = {}
    for y in range(FIRST_WINTER, LAST_WINTER + 1):
        rows[winter_label(y)] = [monthly.loc[winter_month_date(y, m)] for m in MONTHS]
    return pd.DataFrame.from_dict(rows, orient="index", columns=MONTHS)


def load_futures(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """F_sm: Nov-Feb futures on the first trading day of October of each winter.

    Returns (historical strips, the most recent October strip in the file)."""
    df = _read_eia_sheet(cfg.data_dir / "NG_PRI_FUT_S1_D.xls", sheet="Data 2")
    df = df.rename(columns={"RNGC1": "C1", "RNGC2": "C2", "RNGC3": "C3", "RNGC4": "C4"})
    rows, dates = {}, {}
    for y in range(FIRST_WINTER, LAST_WINTER + 1):
        sub = df[(df.Date >= f"{y}-10-01")].dropna(subset=["C1", "C2", "C3", "C4"]).head(1)
        if sub.empty:
            raise ValueError(f"No October {y} futures quotes found")
        rows[winter_label(y)] = [float(sub[FUTURES_COL[m]].iloc[0]) for m in MONTHS]
        dates[winter_label(y)] = sub.Date.iloc[0].date()
    fut = pd.DataFrame.from_dict(rows, orient="index", columns=MONTHS)
    fut.attrs["quote_dates"] = dates
    return fut, fut.iloc[-1]


# ---------------------------------------------------------------------------
# 2. Weather (NOAA CPC population-weighted HDD)
# ---------------------------------------------------------------------------
def _parse_cpc_file(text: str) -> pd.DataFrame:
    lines = [ln.strip().strip('"') for ln in text.splitlines() if ln.strip()]
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Region|"))
    header = lines[start].split("|")
    recs = []
    for ln in lines[start + 1:]:
        parts = ln.split("|")
        if len(parts) != len(header):
            continue
        st = parts[0].strip()
        for d, v in zip(header[1:], parts[1:]):
            recs.append((st, d, v))
    out = pd.DataFrame(recs, columns=["state", "date", "hdd"])
    out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    out["hdd"] = pd.to_numeric(out["hdd"], errors="coerce")
    out.loc[out.hdd < 0, "hdd"] = np.nan  # missing-value flags
    return out


def load_hdd(cfg: Config, states: list[str]) -> pd.DataFrame:
    """Daily HDD (state, date, hdd) from NOAA CPC StatesCONUS.Heating.txt, cached in data/hdd/."""
    if cfg.hdd_csv is not None:
        df = pd.read_csv(cfg.hdd_csv, parse_dates=["date"])
        return df[df.state.isin(states)]
    cache = cfg.data_dir / "hdd"
    frames = []
    for year in range(FIRST_WINTER, LAST_WINTER + 2):
        f = cache / f"StatesCONUS.Heating.{year}.txt"
        if not f.exists():
            if not cfg.download:
                raise FileNotFoundError(f"{f} missing and --no-download set")
            _download(NOAA_HDD_URL.format(year=year), f)
        frames.append(_parse_cpc_file(f.read_text()))
    df = pd.concat(frames, ignore_index=True)
    return df[df.state.isin(states)]


def monthly_hdd(hdd: pd.DataFrame, st: str) -> pd.DataFrame:
    """HDD_sm: sum of daily HDD within each delivery month of each winter."""
    d = hdd[hdd.state == st]
    tot = d.groupby([d.date.dt.year, d.date.dt.month])["hdd"].sum(min_count=1)
    rows = {}
    for y in range(FIRST_WINTER, LAST_WINTER + 1):
        rows[winter_label(y)] = [tot.get(winter_month_date(y, m), np.nan) for m in MONTHS]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=MONTHS)
    if out.isna().any().any():
        missing = out[out.isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing HDD for {st} in winters {missing}")
    return out


# ---------------------------------------------------------------------------
# 3. Demand
# ---------------------------------------------------------------------------
def load_state_consumption(cfg: Config, st: str) -> pd.DataFrame:
    """Residential + commercial consumption for (winter, month), in MMBtu."""
    f = cfg.data_dir / f"NG_CONS_SUM_DCU_S{st}_M.xls"
    if not f.exists():
        if not cfg.download:
            raise FileNotFoundError(f"{f} missing and --no-download set")
        _download(EIA_CONS_URL.format(st=st), f)
    df = _read_eia_sheet(f)
    res = next(c for c in df.columns if c.startswith("N3010"))
    com = next(c for c in df.columns if c.startswith("N3020"))
    df["rc_mmcf"] = df[res] + df[com]
    tot = df.set_index([df.Date.dt.year, df.Date.dt.month])["rc_mmcf"]
    rows = {}
    for y in range(FIRST_WINTER, LAST_WINTER + 1):
        rows[winter_label(y)] = [tot.loc[winter_month_date(y, m)] * 1_000 * HEAT_CONTENT
                                 for m in MONTHS]  # MMcf -> Mcf -> MMBtu
    return pd.DataFrame.from_dict(rows, orient="index", columns=MONTHS)


def utility_share(cfg: Config, st: str) -> tuple[str, float]:
    """Utility's share of state residential + commercial deliveries (EIA-176)."""
    df = pd.read_excel(cfg.data_dir / "eia176_company_deliveries.xlsx")
    df.columns = [int(c) if isinstance(c, (int, float)) else c for c in df.columns]
    df["Company"] = df["Company"].astype(str).str.strip()
    company = cfg.companies.get(st, DEFAULT_COMPANY.get(st))
    sub = df[(df.Area == STATE_NAMES[st]) & df.Item.isin(["Residential Volume", "Commercial Volume"])]
    years = [y for y in cfg.share_years if y in sub.columns]
    by_co = sub.groupby("Company")[years].sum(min_count=1)
    total = by_co.loc["Total of All Companies"]
    if company not in by_co.index:
        # fall back to the largest LDC in the state
        company = by_co.drop("Total of All Companies").mean(axis=1).idxmax()
    share = float((by_co.loc[company] / total).mean())
    return company, share


@dataclass
class DemandFit:
    a: pd.Series
    b: pd.Series
    r2: pd.Series
    hdd_normal: pd.Series


def fit_demand(state_mmbtu: pd.DataFrame, hdd: pd.DataFrame) -> DemandFit:
    """OLS per month: StateDemand_sm = a_m + b_m * HDD_sm."""
    a, b, r2 = {}, {}, {}
    for m in MONTHS:
        x, y = hdd[m].values.astype(float), state_mmbtu[m].values.astype(float)
        bm, am = np.polyfit(x, y, 1)
        resid = y - (am + bm * x)
        a[m], b[m] = am, bm
        r2[m] = 1 - resid.var() / y.var()
    return DemandFit(pd.Series(a), pd.Series(b), pd.Series(r2), hdd.mean())


# ---------------------------------------------------------------------------
# 4. Geopolitical risk
# ---------------------------------------------------------------------------
def geopolitical_indicator(cfg: Config) -> tuple[int, float, float]:
    g = pd.read_excel(cfg.data_dir / "data_gpr_export.xls", usecols=["month", "GPR"])
    g["month"] = pd.to_datetime(g["month"])
    g = g.dropna(subset=["GPR"])
    hist = g[g.month >= "1985-01-01"]["GPR"]
    thresh = float(hist.quantile(cfg.gpr_percentile))
    cur = g[g.month.dt.strftime("%Y-%m") == cfg.gpr_month]["GPR"]
    if cur.empty:
        raise ValueError(f"No GPR value for {cfg.gpr_month}")
    value = float(cur.iloc[0])
    flag = int(value >= thresh) if cfg.force_g is None else int(cfg.force_g)
    return flag, value, thresh


# ---------------------------------------------------------------------------
# 5. Assemble model inputs for one state
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    state: str
    company: str
    share: float
    F: pd.Series            # F_m
    F_hist: pd.DataFrame    # F_sm
    P: pd.DataFrame         # P_sm
    Ptilde: pd.DataFrame    # rescaled spot
    D: pd.Series            # D_m
    Ds: pd.DataFrame        # D_sm
    HDD: pd.DataFrame
    fit: DemandFit
    H: list[str]
    pi: pd.Series
    g: int
    gpr_value: float
    gpr_thresh: float


def build_instance(cfg: Config, st: str, spot, fut_hist, F_now, hdd_daily, gpr) -> Instance:
    company, share = utility_share(cfg, st)
    state_d = load_state_consumption(cfg, st)
    HDD = monthly_hdd(hdd_daily, st)
    fit = fit_demand(state_d, HDD)

    D = share * (fit.a + fit.b * fit.hdd_normal)
    if cfg.scenario_demand == "fitted":
        Ds = share * (HDD * fit.b + fit.a)
    else:
        Ds = share * state_d
    Ds = Ds.clip(lower=0)

    Ptilde = (spot / fut_hist) * F_now

    basis = spot if cfg.high_basis == "raw" else Ptilde
    avg = basis.mean(axis=1)
    H = avg[avg >= avg.quantile(cfg.high_quantile)].index.tolist()

    g, gv, gt = gpr
    pi = pd.Series(1.0, index=spot.index)
    if g == 1:
        pi[H] *= cfg.theta
    pi /= pi.sum()

    return Instance(st, company, share, F_now, fut_hist, spot, Ptilde, D, Ds, HDD, fit, H, pi, g, gv, gt)


# ---------------------------------------------------------------------------
# 6. Cost evaluation and the Gurobi model
# ---------------------------------------------------------------------------
def scenario_costs(inst: Instance, h: pd.Series) -> pd.Series:
    """c_s for a fixed hedge vector h (used for benchmarks and Cmax calibration)."""
    hedged = (h * inst.D * inst.F).sum()
    spot_part = ((inst.Ds - h * inst.D) * inst.Ptilde).sum(axis=1)
    return hedged + spot_part


def high_avg(inst: Instance, c: pd.Series) -> float:
    w = inst.pi[inst.H]
    return float((w * c[inst.H]).sum() / w.sum())


def resolve_cmax(cfg: Config, inst: Instance) -> float | None:
    if cfg.cmax_abs is not None:
        return cfg.cmax_abs
    return cmax_from_factor(inst, cfg.cmax_factor)


def cmax_from_factor(inst: Instance, factor: float | None) -> float | None:
    """Cmax = factor x (high-price average cost of hedging 50% every month)."""
    if factor is None:
        return None
    ref = high_avg(inst, scenario_costs(inst, pd.Series(0.5, index=MONTHS)))
    return factor * ref


@dataclass
class Solution:
    status: str
    h: pd.Series | None = None
    c: pd.Series | None = None
    expected: float | None = None
    high_avg: float | None = None
    cmax: float | None = None
    cmax_dual: float | None = None


def solve(cfg: Config, inst: Instance, cmax: float | None) -> Solution:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if cfg.verbose else 0)
    env.start()
    mdl = gp.Model(f"hedge_{inst.state}", env=env)

    S = list(inst.pi.index)
    h = mdl.addVars(MONTHS, lb={m: cfg.lower[i] for i, m in enumerate(MONTHS)},
                    ub={m: cfg.upper[i] for i, m in enumerate(MONTHS)}, name="h")
    c = mdl.addVars(S, lb=-GRB.INFINITY, name="c")

    # (1) scenario cost definition
    for s in S:
        mdl.addConstr(
            c[s] == gp.quicksum(
                h[m] * inst.D[m] * inst.F[m]
                + (inst.Ds.at[s, m] - h[m] * inst.D[m]) * inst.Ptilde.at[s, m]
                for m in MONTHS),
            name=f"cost[{s}]")

    # (2) hedge-share bounds are variable bounds (L_m <= h_m <= U_m)

    # (3) high-price scenario cost limit
    risk = None
    if cmax is not None:
        wH = float(inst.pi[inst.H].sum())
        risk = mdl.addConstr(
            gp.quicksum(inst.pi[s] * c[s] for s in inst.H) <= cmax * wH, name="high_price_limit")

    mdl.setObjective(gp.quicksum(inst.pi[s] * c[s] for s in S), GRB.MINIMIZE)
    mdl.optimize()

    if mdl.Status != GRB.OPTIMAL:
        status = {GRB.INFEASIBLE: "infeasible", GRB.INF_OR_UNBD: "infeasible"}.get(
            mdl.Status, f"status_{mdl.Status}")
        mdl.dispose(); env.dispose()
        return Solution(status, cmax=cmax)

    hv = pd.Series({m: h[m].X for m in MONTHS})
    cv = pd.Series({s: c[s].X for s in S})
    dual = None
    if risk is not None:
        # $ of expected cost saved per extra $ of allowed high-price average cost
        dual = -risk.Pi * float(inst.pi[inst.H].sum())
    sol = Solution("optimal", hv, cv, mdl.ObjVal, high_avg(inst, cv), cmax, dual)
    mdl.dispose(); env.dispose()
    return sol


# ---------------------------------------------------------------------------
# 7. Reporting, benchmarks and sensitivity
# ---------------------------------------------------------------------------
def strategy_table(inst: Instance, sol: Solution) -> pd.DataFrame:
    strategies = {f"{int(k * 100)}% hedge": pd.Series(k, index=MONTHS) for k in (0, .25, .5, .75, 1)}
    if sol.h is not None:
        strategies["Optimal"] = sol.h
    total_vol = inst.D.sum()
    rows = []
    for name, h in strategies.items():
        c = scenario_costs(inst, h)
        rows.append({
            "strategy": name,
            **{f"h_{m}": round(h[m], 4) for m in MONTHS},
            "expected_cost": (inst.pi * c).sum(),
            "high_price_avg_cost": high_avg(inst, c),
            "worst_winter_cost": c.max(),
            "worst_winter": c.idxmax(),
            "std_cost": np.sqrt((inst.pi * (c - (inst.pi * c).sum()) ** 2).sum()),
            "expected_$per_MMBtu": (inst.pi * c).sum() / total_vol,
        })
    return pd.DataFrame(rows).set_index("strategy")


def sensitivity(cfg: Config, inst: Instance) -> dict[str, pd.DataFrame]:
    """Re-solve over a grid of theta (H weight, as if g = 1; theta = 1 is the g = 0 case)
    and Cmax factors."""
    thetas = [1.0, 1.5, 2.0, 3.0, 4.0]
    factors = [0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10, None]
    exp_tab, hedge_tab = {}, {}
    for th in thetas:
        pi = pd.Series(1.0, index=inst.pi.index)
        pi[inst.H] *= th
        pi /= pi.sum()
        sub = Instance(**{**inst.__dict__, "pi": pi})
        exp_row, h_row = {}, {}
        for fct in factors:
            col = "no limit" if fct is None else f"{fct:.2f}x"
            cmax = cmax_from_factor(sub, fct)
            s = solve(cfg, sub, cmax)
            exp_row[col] = s.expected if s.status == "optimal" else np.nan
            h_row[col] = "/".join(f"{v:.2f}" for v in s.h) if s.h is not None else s.status
        exp_tab[f"theta={th}"] = exp_row
        hedge_tab[f"theta={th}"] = h_row
    return {
        "sensitivity_expected_cost": pd.DataFrame.from_dict(exp_tab, orient="index"),
        "sensitivity_hedge_shares": pd.DataFrame.from_dict(hedge_tab, orient="index"),
    }


def make_plots(out: Path, inst: Instance, strat: pd.DataFrame, sens: dict | None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in strat.iterrows():
        ax.scatter(r.expected_cost / 1e6, r.high_price_avg_cost / 1e6, s=60,
                   marker="*" if name == "Optimal" else "o", zorder=3)
        ax.annotate(name, (r.expected_cost / 1e6, r.high_price_avg_cost / 1e6),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Expected winter cost ($M)")
    ax.set_ylabel("Avg cost in high-price winters H ($M)")
    ax.set_title(f"{inst.state}: cost vs. high-price exposure")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "strategy_frontier.png", dpi=150); plt.close(fig)

    if sens:
        t = sens["sensitivity_expected_cost"] / 1e6
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(t.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(t.shape[1]), t.columns, rotation=30)
        ax.set_yticks(range(t.shape[0]), t.index)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                ax.text(j, i, "infeas." if np.isnan(v) else f"{v:,.1f}", ha="center", va="center",
                        fontsize=8, color="white")
        ax.set_xlabel("Cmax (multiple of 50%-hedge high-price cost)")
        ax.set_title(f"{inst.state}: expected cost ($M) by theta and Cmax")
        fig.colorbar(im, ax=ax, shrink=.8)
        fig.tight_layout(); fig.savefig(out / "sensitivity_heatmap.png", dpi=150); plt.close(fig)


def report(cfg: Config, inst: Instance, sol: Solution) -> None:
    out = cfg.out_dir / inst.state
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}\n{STATE_NAMES[inst.state]} ({inst.state})\n{'=' * 72}")
    print(f"Utility: {inst.company}  (share of state R+C deliveries = {inst.share:.1%})")
    print("Demand regression  D_state = a_m + b_m * HDD_m")
    print(pd.DataFrame({"a_m (MMBtu)": inst.fit.a, "b_m (MMBtu/HDD)": inst.fit.b,
                        "R^2": inst.fit.r2, "normal HDD": inst.fit.hdd_normal}).round(3).to_string())
    print(f"\nForecast utility demand D_m (MMBtu):\n{inst.D.round(0).to_string()}")
    print(f"\nHigh-price winters H ({len(inst.H)}): {', '.join(inst.H)}")
    print(f"GPR {cfg.gpr_month} = {inst.gpr_value:.1f}, threshold = {inst.gpr_thresh:.1f} "
          f"-> g = {inst.g}" + (f", H weights x{cfg.theta}" if inst.g else ""))
    if sol.status != "optimal":
        print(f"\nModel status: {sol.status.upper()} (Cmax = {sol.cmax:,.0f}). "
              "Loosen --cmax-factor or the hedge bounds.")
        return
    print(f"\nOptimal hedge shares: " + ", ".join(f"{m} {sol.h[m]:.1%}" for m in MONTHS))
    print(f"Expected cost      : ${sol.expected:,.0f}")
    print(f"High-price avg cost: ${sol.high_avg:,.0f}"
          + (f"  (Cmax = ${sol.cmax:,.0f})" if sol.cmax else ""))
    if sol.cmax_dual is not None:
        print(f"Shadow price of Cmax: {sol.cmax_dual:.4f} $ expected / $ of Cmax")

    strat = strategy_table(inst, sol)
    print("\nStrategy comparison:")
    print(strat.drop(columns=[f"h_{m}" for m in MONTHS]).to_string(
        float_format=lambda v: f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"))

    # save parameters and results
    pd.DataFrame({"F_m": inst.F, "D_m": inst.D, "L_m": cfg.lower, "U_m": cfg.upper}).to_csv(
        out / "parameters_month.csv")
    pd.concat({"F_sm": inst.F_hist, "P_sm": inst.P, "Ptilde_sm": inst.Ptilde,
               "D_sm": inst.Ds, "HDD_sm": inst.HDD}, axis=1).assign(
        pi_s=inst.pi, in_H=inst.pi.index.isin(inst.H), c_s_optimal=sol.c).to_csv(
        out / "parameters_scenario.csv")
    pd.DataFrame({"h_m": sol.h, "hedged_MMBtu": sol.h * inst.D}).to_csv(out / "optimal_hedge.csv")
    strat.to_csv(out / "strategy_comparison.csv")

    sens = None
    if cfg.sensitivity:
        sens = sensitivity(cfg, inst)
        for k, v in sens.items():
            v.to_csv(out / f"{k}.csv")
        print("\nSensitivity - hedge shares Nov/Dec/Jan/Feb (rows theta, cols Cmax factor):")
        print(sens["sensitivity_hedge_shares"].to_string())
    if cfg.plots:
        make_plots(out, inst, strat, sens)
    print(f"\nResults written to {out}")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Natural gas winter hedging LP (Gurobi)")
    p.add_argument("--states", nargs="+", default=["TX", "VT"], choices=list(STATE_NAMES))
    p.add_argument("--theta", type=float, default=2.0, help="weight on H winters when g=1")
    p.add_argument("--cmax-factor", type=float, default=1.0,
                   help="Cmax as a multiple of the 50%%-hedge high-price avg cost")
    p.add_argument("--cmax", type=float, default=None, help="absolute Cmax in $ (overrides factor)")
    p.add_argument("--no-cmax", action="store_true", help="drop the high-price cost constraint")
    p.add_argument("--lower", type=float, nargs=4, default=[0, 0, 0, 0], metavar=("NOV", "DEC", "JAN", "FEB"))
    p.add_argument("--upper", type=float, nargs=4, default=[1, 1, 1, 1], metavar=("NOV", "DEC", "JAN", "FEB"))
    p.add_argument("--futures", type=float, nargs=4, default=None, metavar=("NOV", "DEC", "JAN", "FEB"),
                   help="NYMEX settlements on Oct 1 2026 ($/MMBtu); default = latest October strip in data")
    p.add_argument("--gpr-month", default="2026-09")
    p.add_argument("--force-g", type=int, choices=[0, 1], default=None)
    p.add_argument("--high-basis", choices=["raw", "rescaled"], default="rescaled",
                   help="rank winters for H by rescaled spot Ptilde (default) or raw spot")
    p.add_argument("--scenario-demand", choices=["fitted", "actual"], default="fitted")
    p.add_argument("--company", action="append", default=[], metavar="ST=NAME",
                   help='override utility, e.g. --company "TX=CENTERPOINT ENERGY ENTEX"')
    p.add_argument("--hdd-csv", type=Path, default=None,
                   help="offline HDD file with columns state,date,hdd (skips NOAA download)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--no-sensitivity", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="show Gurobi log")
    a = p.parse_args(argv)
    return Config(
        states=a.states, theta=a.theta, cmax_factor=None if a.no_cmax else a.cmax_factor,
        cmax_abs=None if a.no_cmax else a.cmax, lower=a.lower, upper=a.upper, futures=a.futures,
        gpr_month=a.gpr_month, force_g=a.force_g, high_basis=a.high_basis,
        scenario_demand=a.scenario_demand,
        companies=dict(c.split("=", 1) for c in a.company), hdd_csv=a.hdd_csv,
        data_dir=a.data_dir, out_dir=a.out_dir, download=not a.no_download,
        sensitivity=not a.no_sensitivity, plots=not a.no_plots, verbose=a.verbose)


def main(argv=None) -> None:
    cfg = parse_args(argv)
    for i in range(4):
        if not 0 <= cfg.lower[i] <= cfg.upper[i] <= 1:
            sys.exit("Hedge bounds must satisfy 0 <= L_m <= U_m <= 1")

    print("Loading prices ...")
    spot = load_spot(cfg)
    fut_hist, latest = load_futures(cfg)
    if cfg.futures is not None:
        F_now = pd.Series(cfg.futures, index=MONTHS)
    else:
        F_now = latest.copy()
        print(f"  NOTE: --futures not given; using the latest October strip in the EIA file "
              f"({fut_hist.attrs['quote_dates'][fut_hist.index[-1]]}). Pass the Oct 1 2026 CME "
              f"settlements with --futures NOV DEC JAN FEB.")
    print(f"  F_m = {F_now.round(3).to_dict()}")

    print("Loading weather ...")
    hdd = load_hdd(cfg, cfg.states)
    gpr = geopolitical_indicator(cfg)

    for st in cfg.states:
        inst = build_instance(cfg, st, spot, fut_hist, F_now, hdd, gpr)
        sol = solve(cfg, inst, resolve_cmax(cfg, inst))
        report(cfg, inst, sol)


if __name__ == "__main__":
    main()
