"""
Milestone 5 analysis: sensitivity, what-if scenarios, Monte Carlo and train/test results.

Builds on gurobi.py (same data, same model) for the October 2, 2023 decision and writes
tables (CSV) and charts (PNG) to outputs/m5/.

    python m5_analysis.py                    # Texas and Vermont
    python m5_analysis.py --states VT --sims 20000
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import gurobi as gm
from gurobi import MONTHS

# Fixed categorical order (dataviz reference palette): color follows the strategy, never its rank
COLORS = {"Model (optimal)": "#2a78d6", "20% hedge": "#eb6834", "50% hedge": "#1baf7a",
          "80% hedge": "#eda100", "No hedge": "#8a8f98"}
INK, MUTED, GRID = "#1f2328", "#5b616b", "#e3e5e8"


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def fmt_h(h) -> str:
    return "/".join(f"{v:.0%}" for v in h) if h is not None else "infeasible"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup(cfg: gm.Config):
    spot = gm.load_spot(cfg)
    fut, _ = gm.load_futures(cfg)
    F_now = gm.resolve_futures(cfg, fut)
    hdd = gm.load_hdd(cfg, cfg.states)
    gpr = gm.geopolitical_indicator(cfg)
    insts = {st: gm.build_instance(cfg, st, spot, fut, F_now, hdd, gpr) for st in cfg.states}
    return spot, fut, hdd, insts


def summarize(cfg, inst, sol, label, **extra) -> dict:
    row = {"scenario": label, "status": sol.status, "hedge Nov/Dec/Jan/Feb": fmt_h(sol.h)}
    if sol.status == "optimal":
        cons = sol.constraints if sol.constraints is not None else pd.DataFrame()
        row.update({
            "expected_cost": sol.expected, "high_price_cost": sol.high_avg, "futures_spend": sol.spend,
            "cmax": sol.cmax, "budget": sol.budget,
            "cmax_shadow": sol.cmax_dual, "budget_shadow": sol.budget_dual,
            "binding": ", ".join(i for i, r in cons.iterrows() if abs(r["slack"]) < 1e-3) or "bounds only",
        })
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# 1. Cmax sweep: the cost/risk trade-off curve
# ---------------------------------------------------------------------------
def cmax_sweep(cfg, inst) -> pd.DataFrame:
    rows = []
    for k in [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.06, 1.08, None]:
        sol = gm.solve(cfg, inst, gm.cmax_from_factor(inst, k))
        rows.append({"kappa": "no limit" if k is None else k, **summarize(cfg, inst, sol, f"kappa={k}")})
    return pd.DataFrame(rows).set_index("kappa")


# ---------------------------------------------------------------------------
# 2. Budget sweep
# ---------------------------------------------------------------------------
def budget_sweep(cfg, inst) -> pd.DataFrame:
    rows = []
    for kappa in (1.00, 0.97, 0.95):
        for b in (0.30, 0.35, 0.40, 0.45, 0.50, 0.60, None):
            sol = gm.solve(cfg, inst, gm.cmax_from_factor(inst, kappa), gm.resolve_budget(cfg, inst, b))
            rows.append({"kappa": kappa, "budget_share": "none" if b is None else b,
                         **summarize(cfg, inst, sol, "")})
    return pd.DataFrame(rows).drop(columns="scenario")


# ---------------------------------------------------------------------------
# 3. What-if scenarios. Cmax and B are policy limits set in dollars before the winter, so they
#    stay at their baseline dollar values unless the scenario is about the limits themselves.
# ---------------------------------------------------------------------------
def what_ifs(cfg, inst) -> pd.DataFrame:
    cmax0 = gm.resolve_cmax(cfg, inst)
    b0 = gm.resolve_budget(cfg, inst)
    rows = [summarize(cfg, inst, gm.solve(cfg, inst, cmax0, b0), "Baseline (20-80% band, 50% budget)")]

    m4 = replace(cfg, lower=[0.0] * 4, upper=[1.0] * 4)
    rows.append(summarize(m4, inst, gm.solve(m4, inst, cmax0, None),
                          "Milestone 4 model (0-100%, no budget)"))

    pi = pd.Series(1.0, index=inst.pi.index)
    pi[inst.H] *= cfg.theta
    alert = replace(inst, pi=pi / pi.sum())
    rows.append(summarize(cfg, alert, gm.solve(cfg, alert, cmax0, b0),
                          f"Geopolitical alert (g=1, theta={cfg.theta:g})"))

    jan = replace(cfg, lower=[0.2, 0.2, 0.4, 0.2])
    rows.append(summarize(jan, inst, gm.solve(jan, inst, cmax0, b0), "January minimum hedge 40%"))

    rows.append(summarize(cfg, inst, gm.solve(cfg, inst, cmax0, 0.40 * (inst.F * inst.D).sum()),
                          "Tighter budget (40%)"))
    rows.append(summarize(cfg, inst, gm.solve(cfg, inst, 0.97 * cmax0, b0), "Tighter risk limit (Cmax -3%)"))

    spike = replace(inst, Ds=inst.Ds * 1.2)
    rows.append(summarize(cfg, spike, gm.solve(cfg, spike, cmax0, b0),
                          "Demand 20% above history, same $ limits"))
    rows.append(summarize(cfg, spike, gm.solve(cfg, spike, gm.resolve_cmax(cfg, spike), b0),
                          "Demand 20% above history, limits rescaled"))

    recent = [s for s in inst.pi.index if int(s[:4]) >= 2008]
    rec = replace(inst, F_hist=inst.F_hist.loc[recent], P=inst.P.loc[recent],
                  Ptilde=inst.Ptilde.loc[recent], Ds=inst.Ds.loc[recent], HDD=inst.HDD.loc[recent],
                  H=[s for s in inst.H if s in recent],
                  pi=pd.Series(1 / len(recent), index=recent))
    rows.append(summarize(cfg, rec, gm.solve(cfg, rec, gm.resolve_cmax(cfg, rec), b0),
                          "Only the last 15 winters as scenarios"))
    return pd.DataFrame(rows).set_index("scenario")


# ---------------------------------------------------------------------------
# 4. Monte Carlo: simulate winters around history and score the chosen hedge
# ---------------------------------------------------------------------------
def monte_carlo(cfg, inst, h_model: pd.Series, n: int = 10_000, seed: int = 4004,
                price_sigma: float = 0.15, hdd_sigma: float = 0.08) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each simulated winter draws a historical winter s (with probability pi_s) to keep its joint
    price/weather pattern, then perturbs each month's spot-to-futures ratio by lognormal noise
    (sd price_sigma) and its HDD by normal noise (sd hdd_sigma). Demand follows the regression."""
    rng = np.random.default_rng(seed)
    S = inst.pi.index.to_numpy()
    idx = rng.choice(len(S), size=n, p=inst.pi.to_numpy())
    ratio = (inst.P / inst.F_hist).to_numpy()[idx]
    ratio = ratio * np.exp(price_sigma * rng.standard_normal(ratio.shape) - price_sigma ** 2 / 2)
    hdd = inst.HDD.to_numpy()[idx] * (1 + hdd_sigma * rng.standard_normal(ratio.shape))
    a, b = inst.fit.a[MONTHS].to_numpy(), inst.fit.b[MONTHS].to_numpy()
    D_act = np.clip(inst.share * (a + b * np.clip(hdd, 0, None)), 0, None)
    P = inst.F[MONTHS].to_numpy() * ratio
    F, D = inst.F[MONTHS].to_numpy(), inst.D[MONTHS].to_numpy()

    strategies = {"Model (optimal)": h_model.to_numpy(), "20% hedge": np.full(4, .2),
                  "50% hedge": np.full(4, .5), "80% hedge": np.full(4, .8), "No hedge": np.zeros(4)}
    costs = pd.DataFrame({k: (h * D * F + (D_act - h * D) * P).sum(axis=1)
                          for k, h in strategies.items()})
    stats = []
    for k in costs:
        c = costs[k].to_numpy()
        tail = np.sort(c)[-int(0.05 * n):]
        stats.append({"strategy": k, "mean": c.mean(), "std": c.std(), "p5": np.percentile(c, 5),
                      "median": np.median(c), "p95": np.percentile(c, 95), "cvar95": tail.mean(),
                      "prob_cheaper_than_no_hedge": (c < costs["No hedge"]).mean()})
    return costs, pd.DataFrame(stats).set_index("strategy")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_frontier(out: Path, sweeps: dict, strats: dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.2 * len(sweeps), 3.9), squeeze=False)
    for ax, (st, sw) in zip(axes[0], sweeps.items()):
        sw = sw[sw.status == "optimal"]
        ax.plot(sw.high_price_cost / 1e6, sw.expected_cost / 1e6, color=COLORS["Model (optimal)"],
                lw=2, marker="o", ms=4, label="Model, varying $C_{max}$", zorder=3)
        base = sw.loc[1.0]
        ax.scatter(base.high_price_cost / 1e6, base.expected_cost / 1e6, s=90, facecolor="white",
                   edgecolor=COLORS["Model (optimal)"], lw=2, zorder=4)
        ax.annotate("baseline\n$C_{max}$ = 1.00×", (base.high_price_cost / 1e6, base.expected_cost / 1e6),
                    textcoords="offset points", xytext=(8, 6), fontsize=7.5, color=INK)
        st_tab = strats[st]
        for name, color in (("No hedge", COLORS["No hedge"]), ("20% hedge", COLORS["20% hedge"]),
                            ("50% hedge", COLORS["50% hedge"]), ("80% hedge", COLORS["80% hedge"])):
            key = "0% hedge" if name == "No hedge" else name
            r = st_tab.loc[key]
            ax.scatter(r.high_price_avg_cost / 1e6, r.expected_cost / 1e6, s=55, marker="s",
                       color=color, edgecolor="white", lw=1.5, zorder=4, label=name)
        ax.set_xlabel("Average cost in high-price winters ($M)", fontsize=8, color=INK)
        ax.set_ylabel("Expected winter cost ($M)", fontsize=8, color=INK)
        ax.set_title(f"{gm.STATE_NAMES[st]}", fontsize=10, color=INK, loc="left")
        style(ax)
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Lower expected cost costs more in bad winters: every point on the line is optimal for one $C_{max}$",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig_frontier.png", dpi=200)
    plt.close(fig)


def plot_monte_carlo(out: Path, mcs: dict):
    import matplotlib.pyplot as plt
    order = ["No hedge", "20% hedge", "Model (optimal)", "50% hedge", "80% hedge"]
    fig, axes = plt.subplots(1, len(mcs), figsize=(5.2 * len(mcs), 3.4), squeeze=False)
    for ax, (st, stats) in zip(axes[0], mcs.items()):
        y = np.arange(len(order))[::-1]
        for yi, k in zip(y, order):
            r = stats.loc[k] / 1e6
            c = COLORS[k]
            ax.plot([r.p5, r.p95], [yi, yi], color=c, lw=2, solid_capstyle="round")
            ax.scatter(r["median"], yi, color=c, s=40, zorder=3, edgecolor="white", lw=1.5)
            ax.scatter(r.cvar95, yi, color=c, marker="|", s=160, lw=2, zorder=3)
        ax.set_yticks(y, order, fontsize=8, color=INK)
        ax.set_xlabel("Simulated winter cost ($M)", fontsize=8, color=INK)
        ax.set_title(gm.STATE_NAMES[st], fontsize=10, color=INK, loc="left")
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("10,000 simulated winters: dot = median, line = 5th to 95th percentile, tick = average of worst 5%",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig_monte_carlo.png", dpi=200)
    plt.close(fig)


def plot_train_test(out: Path, tts: dict):
    import matplotlib.pyplot as plt
    keys = ["Model (optimal)", "20% hedge", "50% hedge", "80% hedge"]
    fig, axes = plt.subplots(len(tts), 1, figsize=(8.5, 2.9 * len(tts)), squeeze=False)
    for ax, (st, res) in zip(axes[:, 0], tts.items()):
        d = res[keys].sub(res["0% hedge"], axis=0) / 1e6
        x = np.arange(len(d))
        w = 0.19
        for i, k in enumerate(keys):
            ax.bar(x + (i - 1.5) * w, d[k], width=w - 0.02, color=COLORS[k], label=k)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, d.index, fontsize=8)
        ax.set_ylabel("Cost vs. no hedge ($M)", fontsize=8, color=INK)
        ax.set_title(f"{gm.STATE_NAMES[st]}: bars below zero = hedging saved money that winter",
                     fontsize=9, color=INK, loc="left")
        style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_train_test.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Milestone 5 analysis")
    p.add_argument("--states", nargs="+", default=["TX", "VT"])
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--hdd-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=gm.OUT_DIR)
    a = p.parse_args(argv)
    cfg = gm.Config(states=a.states, decision_year=2023, hdd_csv=a.hdd_csv, out_dir=a.out_dir,
                    sensitivity=False, plots=False)
    out = cfg.out_dir / "m5"
    out.mkdir(parents=True, exist_ok=True)
    spot, fut, hdd, insts = setup(cfg)

    sweeps, strats, mcs, tts = {}, {}, {}, {}
    pd.set_option("display.width", 220)
    for st, inst in insts.items():
        d = out / st
        d.mkdir(exist_ok=True)
        base = gm.solve(cfg, inst, gm.resolve_cmax(cfg, inst))
        strats[st] = gm.strategy_table(inst, base)
        strats[st].to_csv(d / "strategy_comparison.csv")
        base.ranging.to_csv(d / "post_optimality_variables.csv")
        base.constraints.to_csv(d / "post_optimality_constraints.csv")
        print(f"\n=== {st} baseline: {fmt_h(base.h)}, expected ${base.expected:,.0f}, "
              f"high-price ${base.high_avg:,.0f}, spend ${base.spend:,.0f} of ${base.budget:,.0f}")
        print(base.ranging.to_string())
        print(base.constraints.to_string())
        print(strats[st].drop(columns=[f"h_{m}" for m in MONTHS]).to_string())

        sweeps[st] = cmax_sweep(cfg, inst)
        sweeps[st].to_csv(d / "cmax_sweep.csv")
        print("\nCmax sweep:\n", sweeps[st][["status", "hedge Nov/Dec/Jan/Feb", "expected_cost",
                                             "high_price_cost", "cmax_shadow", "budget_shadow", "binding"]].to_string())
        bs = budget_sweep(cfg, inst)
        bs.to_csv(d / "budget_sweep.csv")
        print("\nBudget sweep:\n", bs[["kappa", "budget_share", "status", "hedge Nov/Dec/Jan/Feb",
                                       "expected_cost", "budget_shadow", "cmax_shadow", "binding"]].to_string())
        wi = what_ifs(cfg, inst)
        wi.to_csv(d / "what_ifs.csv")
        print("\nWhat-ifs:\n", wi[["status", "hedge Nov/Dec/Jan/Feb", "expected_cost", "high_price_cost",
                                   "futures_spend", "binding", "cmax_shadow", "budget_shadow"]].to_string())
        costs, stats = monte_carlo(cfg, inst, base.h, n=a.sims)
        stats.to_csv(d / "monte_carlo_stats.csv")
        mcs[st] = stats
        print("\nMonte Carlo:\n", stats.to_string())
        res, summary = gm.train_test(cfg, st, spot, fut, hdd)
        res.to_csv(d / "train_test_by_winter.csv")
        summary.to_csv(d / "train_test_summary.csv")
        tts[st] = res
        print("\nTrain/test:\n", res[["avg_futures", "avg_actual_spot", "h_Nov", "h_Dec", "h_Jan", "h_Feb",
                                      "Model (optimal)", "0% hedge", "50% hedge"]].to_string())
        print(summary.to_string())

    plot_frontier(out, sweeps, strats)
    plot_monte_carlo(out, mcs)
    plot_train_test(out, tts)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
