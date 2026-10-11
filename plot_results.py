#!/usr/bin/env python3
"""Create publication-style figures from the completed diffusion experiment.

This script only reads existing experiment outputs. It does not run or import
the propagation experiment and never overwrites its original CSV files.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
STRATEGIES = ["degree", "dispersed", "random"]
LABELS = {
    "degree": "Global degree",
    "dispersed": "Dept. dispersed",
    "random": "Random",
}
COLORS = {"degree": "#2864A5", "dispersed": "#D17A28", "random": "#858096"}
MARKERS = {"degree": "o", "dispersed": "s", "random": "^"}
PROBABILITIES = [0.01, 0.05, 0.10]
BUDGETS = [5, 10]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, name: str) -> None:
    for suffix in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200)
    width, height = fig.get_size_inches()
    pixel_height, pixel_width = plt.imread(FIGURES / f"{name}.png").shape[:2]
    print(f"{name}: {width:g} x {height:g} in; "
          f"PNG {pixel_width} x {pixel_height} px; PDF vector")
    plt.close(fig)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(RESULTS / "summary.csv")
    paired = pd.read_csv(RESULTS / "paired_differences.csv")
    runs = pd.read_csv(RESULTS / "runs.csv")
    diagnostics = pd.read_csv(RESULTS / "seed_diagnostics.csv")
    if runs.duplicated(["trial", "k", "p", "strategy"]).any():
        raise ValueError("Duplicate trial/configuration records in runs.csv")
    if not (runs["total_reached"] - runs["new_reached"] == runs["k"]).all():
        raise ValueError("Total reach does not equal new reach plus seed budget")
    if summary.duplicated(["k", "p", "strategy"]).any():
        raise ValueError("Duplicate configuration records in summary.csv")
    return summary, paired, runs, diagnostics


def write_report_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "k", "p", "strategy", "repetitions",
        "new_reached_mean", "new_reached_ci_low", "new_reached_ci_high",
        "departments_with_new_reach_mean",
        "departments_with_new_reach_ci_low", "departments_with_new_reach_ci_high",
    ]
    table = summary[columns].copy()
    table["strategy_label"] = table["strategy"].map(LABELS)
    table["total_reached_mean"] = table["new_reached_mean"] + table["k"]
    table["new_recipients_mean_95ci"] = table.apply(
        lambda row: f"{row.new_reached_mean:.2f} "
        f"[{row.new_reached_ci_low:.2f}, {row.new_reached_ci_high:.2f}]", axis=1)
    table["departments_with_new_recipients_mean_95ci"] = table.apply(
        lambda row: f"{row.departments_with_new_reach_mean:.2f} "
        f"[{row.departments_with_new_reach_ci_low:.2f}, "
        f"{row.departments_with_new_reach_ci_high:.2f}]", axis=1)
    table["_order"] = table["strategy"].map({s: i for i, s in enumerate(STRATEGIES)})
    table = table.sort_values(["k", "p", "_order"]).drop(columns="_order")
    table.to_csv(RESULTS / "report_table.csv", index=False, float_format="%.10g")
    return table


def compute_budget_differences(runs: pd.DataFrame) -> pd.DataFrame:
    """Pair k=10 and k=5 by the identical trial, p and strategy.

    The experiment shares live-edge randomness and uses nested seed sets across
    budgets. Subtracting per-trial total reach therefore retains the intended
    coupling. Confidence intervals are mean +/- 1.96 * paired MCSE.
    """
    records = []
    for p in PROBABILITIES:
        for strategy in STRATEGIES:
            group = runs[(runs["p"] == p) & (runs["strategy"] == strategy)]
            paired = group.pivot(index="trial", columns="k", values="total_reached")
            if set(paired.columns) != set(BUDGETS) or paired[BUDGETS].isna().any().any():
                raise ValueError(f"Incomplete budget pairing for p={p}, {strategy}")
            paired = paired.sort_index()
            delta = (paired[10] - paired[5]).to_numpy(dtype=float)
            if len(delta) < 2:
                raise ValueError("At least two paired trials are required")
            if (delta < 0).any():
                raise ValueError("Nested seed sets should not decrease paired total reach")
            mean = float(delta.mean())
            sd = float(delta.std(ddof=1))
            se = sd / np.sqrt(len(delta))
            records.append({
                "k_from": 5, "k_to": 10, "p": p, "strategy": strategy,
                "strategy_label": LABELS[strategy], "repetitions": len(delta),
                "total_reached_k5_mean": float(paired[5].mean()),
                "total_reached_k10_mean": float(paired[10].mean()),
                "mean_difference": mean, "difference_sd": sd, "mcse": se,
                "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
                "fraction_positive": float((delta > 0).mean()),
                "fraction_tied": float((delta == 0).mean()),
            })
    table = pd.DataFrame(records)
    table.to_csv(RESULTS / "budget_differences.csv", index=False, float_format="%.10g")
    return table


def interval_errors(frame: pd.DataFrame, mean: str, low: str, high: str) -> np.ndarray:
    center = frame[mean].to_numpy()
    return np.maximum(0.0, np.vstack([center - frame[low], frame[high] - center]))


def overview_coverage(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.16, top=0.84,
                        hspace=0.15, wspace=0.12)
    metric_rows = [
        ("new_reached", "New recipients\n(excluding seeds)", (0, 720)),
        ("departments_with_new_reach", "Departments with\nnew recipients", (0, 43)),
    ]
    for col, k in enumerate(BUDGETS):
        for row, (metric, label, limits) in enumerate(metric_rows):
            ax = axes[row, col]
            for strategy in STRATEGIES:
                data = summary[(summary["k"] == k) & (summary["strategy"] == strategy)].sort_values("p")
                if len(data) != len(PROBABILITIES):
                    raise ValueError(f"Missing summary rows for k={k}, {strategy}")
                ax.errorbar(
                    data["p"], data[f"{metric}_mean"],
                    yerr=interval_errors(data, f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"),
                    color=COLORS[strategy], marker=MARKERS[strategy], markersize=6.6,
                    markerfacecolor="none" if strategy == "dispersed" else COLORS[strategy],
                    markeredgewidth=1.4, linewidth=1.65, elinewidth=1.15,
                    capsize=3.2, capthick=1.1, label=LABELS[strategy], zorder=3)
            ax.set_ylim(*limits)
            ax.set_xlim(0.004, 0.106)
            ax.set_xticks(PROBABILITIES, ["0.01", "0.05", "0.10"])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.grid(axis="y", color="#E3E3E3", linewidth=0.7, zorder=0)
            ax.text(0.035, 0.94, f"({chr(97 + 2 * row + col)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=11, fontweight="semibold")
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"Seed budget k = {k}", pad=11)
            else:
                ax.set_xlabel("Transmission probability p")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.929),
               ncol=3, frameon=False, handlelength=2.2, columnspacing=1.8)
    fig.suptitle("Seed placement and simulated reach", y=0.985, fontsize=14, fontweight="semibold")
    fig.text(0.115, 0.046, "Markers: means of 500 trials. Error bars: 95% Monte Carlo confidence intervals.", fontsize=9.4)
    fig.text(0.115, 0.020, "A department is counted only when at least one non-seed member is reached.", fontsize=9.4)
    save_figure(fig, "overview_coverage")


def paired_differences(paired: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.1), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.20, top=0.82, wspace=0.30)
    configurations = [(k, p) for k in BUDGETS for p in PROBABILITIES]
    y = np.array([0, 1, 2, 3.6, 4.6, 5.6])
    metrics = [("new_reached", "(a) New recipients"),
               ("departments_with_new_reach", "(b) Departments with new recipients")]
    for ax, (metric, title) in zip(axes, metrics):
        data = paired[(paired["comparison"] == "dispersed-minus-degree") & (paired["metric"] == metric)]
        data = data.set_index(["k", "p"]).loc[configurations]
        center = data["mean_difference"].to_numpy()
        ax.axvline(0, color="#555555", linewidth=1.0, linestyle="--", zorder=1)
        ax.axhline(2.8, color="#DDDDDD", linewidth=0.7, zorder=0)
        ax.errorbar(center, y, xerr=interval_errors(data, "mean_difference", "ci_low", "ci_high"),
                    fmt="o", markersize=6.4, color=COLORS["dispersed"],
                    elinewidth=1.4, capsize=4, capthick=1.2, zorder=3)
        for value, yy in zip(center, y):
            label = f"{value:+.3f}" if value != 0 else "0.000"
            ax.annotate(label, (value, yy), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, color="#333333")
        lo = min(float(data["ci_low"].min()), 0.0)
        hi = max(float(data["ci_high"].max()), 0.0)
        span = max(hi - lo, 0.1)
        ax.set_xlim(lo - 0.15 * span, hi + 0.19 * span)
        ticks = ax.get_xticks()
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(ticks[(ticks >= xmin) & (ticks <= xmax)])
        ax.set_ylim(6.25, -0.8)
        ax.set_title(title, pad=13, fontsize=12)
        ax.set_xlabel("Dept. dispersed minus Global degree")
        ax.grid(axis="x", color="#ECECEC", linewidth=0.6, zorder=0)
        ax.set_yticks(y, [f"k={k}, p={p:.2f}" for k, p in configurations])
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Paired differences reveal the scale of strategy effects", y=0.965,
                 fontsize=14, fontweight="semibold")
    fig.text(0.15, 0.075, "Points: paired mean differences. Error bars: paired 95% Monte Carlo confidence intervals.", fontsize=9.4)
    fig.text(0.15, 0.035, "Positive values favor Dept. dispersed. The same 500 live-edge trials are used for both strategies.", fontsize=9.4)
    save_figure(fig, "paired_differences")


def budget_totals(budget: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 6.35), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.15, top=0.81, wspace=0.25)
    y = np.array([0, 1, 2, 3.8, 4.8, 5.8, 7.6, 8.6, 9.6])
    labels = []
    for row, yy in zip(budget.itertuples(index=False), y):
        color = COLORS[row.strategy]
        a, b = row.total_reached_k5_mean, row.total_reached_k10_mean
        labels.append(f"{row.p:.2f} | {LABELS[row.strategy]}")
        axes[0].plot([a, b], [yy, yy], color=color, linewidth=2.0, zorder=2)
        axes[0].plot(a, yy, marker="o", markersize=6.4, markerfacecolor="white",
                     markeredgecolor=color, markeredgewidth=1.35, zorder=3)
        axes[0].plot(b, yy, marker="s", markersize=5.5, color=color, zorder=4)
        if b < 150:
            xx, align = a, "left"
        elif a > 550:
            xx, align = b, "right"
        else:
            xx, align = (a + b) / 2, "center"
        axes[0].annotate(f"{a:.2f} to {b:.2f}", (xx, yy), xytext=(0, 8),
                         textcoords="offset points", ha=align, va="bottom", fontsize=9.8)
        axes[1].errorbar(row.mean_difference, yy,
                         xerr=np.array([[row.mean_difference - row.ci_low],
                                        [row.ci_high - row.mean_difference]]),
                         fmt=MARKERS[row.strategy], color=color, markersize=6.0,
                         capsize=3.5, elinewidth=1.3, markeredgewidth=1.1, zorder=3)
        axes[1].annotate(f"+{row.mean_difference:.3f}", (row.mean_difference, yy),
                         xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=9.8)
    for ax in axes:
        ax.set_ylim(10.25, -1.0)
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.axhline(2.9, color="#DDDDDD", linewidth=0.7)
        ax.axhline(6.7, color="#DDDDDD", linewidth=0.7)
        ax.grid(axis="x", color="#ECECEC", linewidth=0.6, zorder=0)
    axes[0].set_xlim(-10, 710)
    axes[0].set_xticks([0, 200, 400, 600])
    axes[0].set_title("(a) Final total reach", pad=12)
    axes[0].set_xlabel("Mean recipients, including seeds")
    axes[0].text(-0.02, 1.025, "p | Strategy", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=10.5, fontweight="semibold")
    axes[1].set_title("(b) Gain from five additional seeds", pad=12)
    axes[1].set_xlabel("Paired change in final total reach")
    maximum = float(budget["ci_high"].max())
    axes[1].set_xlim(-0.06 * maximum, 1.12 * maximum)
    ticks = axes[1].get_xticks()
    xmin, xmax = axes[1].get_xlim()
    axes[1].set_xticks(ticks[(ticks >= xmin) & (ticks <= xmax)])
    axes[1].axvline(0, color="#555555", linewidth=1.0, linestyle="--", zorder=1)
    legend_handles = [
        Line2D([], [], color="#555555", marker="o", markerfacecolor="white",
               markeredgewidth=1.3, linestyle="none", markersize=6.8, label="k = 5"),
        Line2D([], [], color="#555555", marker="s", linestyle="none", markersize=6.0, label="k = 10"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.585, 0.915),
               ncol=2, frameon=False, handletextpad=0.6, columnspacing=2.5)
    fig.suptitle("Budget increase: evaluate total reach, including initial seeds", y=0.977,
                 fontsize=13.5, fontweight="semibold")
    fig.text(0.22, 0.057, "Right panel: mean within-trial gain, k=10 minus k=5, with paired 95% Monte Carlo CI.", fontsize=9.2)
    fig.text(0.22, 0.024, "All 500 trial pairs are retained. A gain can be below five when added seeds were already reachable.", fontsize=9.2)
    save_figure(fig, "budget_totals")


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary, paired, runs, diagnostics = load_results()
    report = write_report_table(summary)
    budget = compute_budget_differences(runs)
    overview_coverage(summary)
    paired_differences(paired)
    budget_totals(budget)
    print(f"Derived tables: report_table.csv ({len(report)} rows), "
          f"budget_differences.csv ({len(budget)} rows)")
    print(f"Read {len(diagnostics)} fixed-seed diagnostic rows; "
          "no extra diagnostic figure was needed.")


if __name__ == "__main__":
    main()
