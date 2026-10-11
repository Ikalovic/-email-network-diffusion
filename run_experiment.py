#!/usr/bin/env python3
"""Independent-cascade seed-selection study on the directed email-Eu-core graph.

Every Monte Carlo trial samples one uniform variate per directed edge. All seed
strategies, both budgets, and all probabilities share this trial's variates.
Reachability in the resulting live-edge graph is the IC final activated set.
No empirical transmission probability is inferred from the email data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd


@dataclass
class Network:
    ids: np.ndarray
    departments: np.ndarray
    department_sizes: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    out_degree: np.ndarray
    metadata: dict


def read_pairs(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            yield tuple(map(int, line.split()))


def load_network(data_dir: Path) -> Network:
    label_rows = list(read_pairs(data_dir / "email-Eu-core-department-labels.txt"))
    labels = dict(label_rows)
    if len(labels) != len(label_rows):
        raise ValueError("Repeated node IDs in department labels")
    ids = np.array(sorted(labels), dtype=np.int64)
    lookup = {int(node): i for i, node in enumerate(ids)}
    raw_edges = list(read_pairs(data_dir / "email-Eu-core.txt"))
    if any(u not in lookup or v not in lookup for u, v in raw_edges):
        raise ValueError("An edge endpoint has no department label")
    edges = sorted(set((u, v) for u, v in raw_edges if u != v))
    sources = np.array([lookup[u] for u, _ in edges], dtype=np.int64)
    targets = np.array([lookup[v] for _, v in edges], dtype=np.int64)
    department_ids = sorted(set(labels.values()))
    dep_lookup = {d: i for i, d in enumerate(department_ids)}
    departments = np.array([dep_lookup[labels[int(u)]] for u in ids], dtype=np.int64)
    sizes = np.bincount(departments, minlength=len(department_ids))
    out_degree = np.bincount(sources, minlength=len(ids))
    in_degree = np.bincount(targets, minlength=len(ids))
    metadata = {
        "nodes": len(ids), "departments": len(sizes),
        "original_edges": len(raw_edges),
        "self_loops_removed": sum(u == v for u, v in raw_edges),
        "duplicate_raw_edges": len(raw_edges) - len(set(raw_edges)),
        "clean_edges": len(edges), "zero_out_degree_nodes": int((out_degree == 0).sum()),
        "zero_in_degree_nodes": int((in_degree == 0).sum()),
        "isolated_nodes": int(((out_degree == 0) & (in_degree == 0)).sum()),
        "department_original_ids": department_ids,
        "source_network_url": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        "source_labels_url": "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz",
        "source_documentation_url": "https://snap.stanford.edu/data/email-Eu-core.html",
        "data_retrieved_date": "2026-10-09",
        "node_policy": "Keep every labelled node, including isolated nodes",
        "edge_policy": "Directed, unweighted; remove self-loops and deduplicate ordered pairs",
    }
    return Network(ids, departments, sizes, sources, targets, out_degree, metadata)


def fixed_seed_sets(network: Network, k: int) -> dict[str, np.ndarray]:
    if not 0 < k <= len(network.department_sizes):
        raise ValueError("Department-dispersed selection requires 0 < k <= department count")
    order = sorted(range(len(network.ids)), key=lambda v: (-network.out_degree[v], network.ids[v]))
    dispersed, seen = [], set()
    for v in order:
        d = int(network.departments[v])
        if d not in seen:
            dispersed.append(v)
            seen.add(d)
            if len(dispersed) == k:
                break
    return {"degree": np.array(order[:k]), "dispersed": np.array(dispersed)}


def build_adjacency(n: int, sources: np.ndarray, targets: np.ndarray, mask: np.ndarray):
    adjacency = [[] for _ in range(n)]
    for u, v in zip(sources[mask], targets[mask]):
        adjacency[int(u)].append(int(v))
    return adjacency


def reachable(adjacency: list[list[int]], seeds: np.ndarray):
    """Multi-source directed BFS: return activation mask and shortest wave index."""
    n = len(adjacency)
    distances = np.full(n, -1, dtype=np.int32)
    queue = []
    for s in seeds:
        s = int(s)
        if distances[s] == -1:
            distances[s] = 0
            queue.append(s)
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adjacency[u]:
            if distances[v] == -1:
                distances[v] = distances[u] + 1
                queue.append(v)
    return distances >= 0, distances


def outcome(network: Network, active: np.ndarray, seeds: np.ndarray):
    nonseed_active = active.copy()
    nonseed_active[seeds] = False
    counts = np.bincount(network.departments[nonseed_active], minlength=len(network.department_sizes))
    return {
        "new_reached": int(nonseed_active.sum()),
        "total_reached": int(active.sum()),
        "new_reached_fraction": float(nonseed_active.sum() / (len(active) - len(seeds))),
        "departments_with_new_reach": int((counts > 0).sum()),
        "macro_department_new_fraction": float(np.mean(counts / network.department_sizes)),
    }, counts


def seed_description(network: Network, seeds: np.ndarray, adjacency: list[list[int]]):
    sset = set(map(int, seeds))
    neighbors = [set(adjacency[int(v)]) - sset for v in seeds]
    one_hop = set().union(*neighbors)
    contacts = sum(map(len, neighbors))
    crossing = sum(network.departments[int(u)] != network.departments[int(v)]
                   for u in seeds for v in adjacency[int(u)])
    all_contacts = sum(len(adjacency[int(u)]) for u in seeds)
    counts = Counter(int(network.departments[v]) for v in seeds)
    return {
        "seed_nodes": ";".join(map(str, network.ids[seeds])),
        "seed_department_count": len(counts),
        "seed_department_max_share": max(counts.values()) / len(seeds),
        "seed_mean_out_degree": float(network.out_degree[seeds].mean()),
        "one_hop_unique_nonseed": len(one_hop),
        "one_hop_department_count": len(set(network.departments[list(one_hop)])) if one_hop else 0,
        "redundant_contact_fraction": (1 - len(one_hop) / contacts) if contacts else 0.0,
        "seed_cross_department_edge_fraction": float(crossing / all_contacts) if all_contacts else 0.0,
    }


METRICS = ["new_reached", "new_reached_fraction", "departments_with_new_reach", "macro_department_new_fraction"]


def summarize(runs: pd.DataFrame, results_dir: Path):
    groups = runs.groupby(["k", "p", "strategy"], sort=True)
    summary = []
    for key, group in groups:
        row = dict(zip(["k", "p", "strategy"], key))
        row["repetitions"] = len(group)
        for metric in METRICS:
            values = group[metric].to_numpy()
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            se = sd / np.sqrt(len(values))
            row.update({f"{metric}_mean": mean, f"{metric}_sd": sd,
                        f"{metric}_mcse": se, f"{metric}_ci_low": mean - 1.96 * se,
                        f"{metric}_ci_high": mean + 1.96 * se})
        summary.append(row)
    pd.DataFrame(summary).to_csv(results_dir / "summary.csv", index=False, float_format="%.10g")
    differences, blocks = [], []
    for (k, p), group in runs.groupby(["k", "p"], sort=True):
        for left, right in [("dispersed", "degree"), ("random", "degree"), ("dispersed", "random")]:
            for metric in METRICS:
                wide = group.pivot(index="trial", columns="strategy", values=metric).sort_index()
                delta = (wide[left] - wide[right]).to_numpy()
                mean, se = float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))
                differences.append({"k": k, "p": p, "comparison": f"{left}-minus-{right}",
                    "metric": metric, "mean_difference": mean, "mcse": se,
                    "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
                    "fraction_positive": float((delta > 0).mean()),
                    "fraction_tied": float((delta == 0).mean()), "repetitions": len(delta)})
                if left == "dispersed" and right == "degree":
                    for b, values in enumerate(np.array_split(delta, 5), 1):
                        blocks.append({"k": k, "p": p, "metric": metric, "block": b,
                            "trials_in_block": len(values), "mean_difference": float(values.mean())})
    pd.DataFrame(differences).to_csv(results_dir / "paired_differences.csv", index=False, float_format="%.10g")
    pd.DataFrame(blocks).to_csv(results_dir / "block_stability.csv", index=False, float_format="%.10g")


def run(data_dir: Path, results_dir: Path, repetitions: int, master_seed: int):
    if repetitions < 10:
        raise ValueError("Use at least 10 trials for the five-block diagnostic")
    started = time.perf_counter()
    results_dir.mkdir(parents=True, exist_ok=True)
    net = load_network(data_dir)
    n, m = len(net.ids), len(net.sources)
    budgets, probabilities = [5, 10], [0.01, 0.05, 0.1]
    strategies = ["degree", "dispersed", "random"]
    fixed = {k: fixed_seed_sets(net, k) for k in budgets}
    original_adj = build_adjacency(n, net.sources, net.targets, np.ones(m, dtype=bool))
    descriptions = []
    for k in budgets:
        for strategy, seeds in fixed[k].items():
            descriptions.append({"k": k, "strategy": strategy, **seed_description(net, seeds, original_adj)})
    pd.DataFrame(descriptions).to_csv(results_dir / "seed_diagnostics.csv", index=False, float_format="%.10g")
    components = np.random.SeedSequence(master_seed).spawn(2)
    edge_rng, seed_rng = [np.random.Generator(np.random.PCG64(s)) for s in components]
    rows, random_diagnostics = [], []
    department_sum = {(k, p, s): np.zeros(len(net.department_sizes), dtype=float)
                      for k in budgets for p in probabilities for s in strategies}
    curves = {key: np.zeros(n + 1, dtype=float) for key in department_sum}
    maximum_wave = {key: 0 for key in department_sum}
    for trial in range(repetitions):
        uniforms = edge_rng.random(m)
        permutation = seed_rng.permutation(n)
        random_seeds = {k: permutation[:k] for k in budgets}
        for k in budgets:
            random_diagnostics.append({"trial": trial, "k": k,
                **seed_description(net, random_seeds[k], original_adj)})
        for p in probabilities:
            adj = build_adjacency(n, net.sources, net.targets, uniforms < p)
            for k in budgets:
                for strategy in strategies:
                    seeds = random_seeds[k] if strategy == "random" else fixed[k][strategy]
                    active, distances = reachable(adj, seeds)
                    metrics, by_department = outcome(net, active, seeds)
                    key = k, p, strategy
                    rows.append({"trial": trial, "k": k, "p": p, "strategy": strategy, **metrics})
                    department_sum[key] += by_department
                    histogram = np.bincount(distances[active], minlength=n + 1)
                    curves[key] += np.cumsum(histogram) - k
                    maximum_wave[key] = max(maximum_wave[key], int(distances.max()))
        if (trial + 1) % 100 == 0:
            print(f"Completed {trial + 1}/{repetitions} shared trials", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "runs.csv", index=False, float_format="%.10g")
    summarize(runs, results_dir)
    pd.DataFrame(random_diagnostics).to_csv(results_dir / "random_seed_diagnostics.csv", index=False, float_format="%.10g")
    departments, curve_rows = [], []
    for (k, p, strategy), sums in department_sum.items():
        for d, count in enumerate(sums):
            departments.append({"k": k, "p": p, "strategy": strategy,
                "department": net.metadata["department_original_ids"][d],
                "department_size": int(net.department_sizes[d]),
                "mean_new_reached": count / repetitions,
                "mean_new_fraction": count / repetitions / net.department_sizes[d]})
        for wave in range(maximum_wave[(k, p, strategy)] + 1):
            curve_rows.append({"k": k, "p": p, "strategy": strategy, "wave": wave,
                "mean_cumulative_new_reached": curves[(k, p, strategy)][wave] / repetitions})
    pd.DataFrame(departments).to_csv(results_dir / "department_results.csv", index=False, float_format="%.10g")
    pd.DataFrame(curve_rows).to_csv(results_dir / "diffusion_curves.csv", index=False, float_format="%.10g")
    config = {"model": "Directed unweighted Independent Cascade, live-edge final-state simulation",
        "budgets": budgets, "probabilities": probabilities, "strategies": strategies,
        "monte_carlo_trials_per_configuration": repetitions, "diffusion_runs": len(rows),
        "master_seed": master_seed, "generator": "NumPy PCG64 with two SeedSequence child streams",
        "coupling": "Shared edge uniform variates across strategies, budgets, and probabilities; independent seed-permutation stream",
        "random_baseline": "Resample seeds every trial; same seed set shared across probabilities",
        "tie_breaking": "Descending cleaned out-degree, then ascending original node ID",
        "department_fraction_denominator": "Full fixed size of each department, including seeds",
        "department_coverage_numerator": "Newly reached non-seed nodes only",
        "intervals": "Normal approximation 95% Monte Carlo intervals; paired differences use trial-matched outcomes; conditional on fixed network/model",
        "timing_seconds": time.perf_counter() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
    }
    (results_dir / "network_audit.json").write_text(json.dumps(net.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"diffusion_runs": len(rows), "seconds": config["timing_seconds"], "results": str(results_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=base / "data")
    parser.add_argument("--results-dir", type=Path, default=base / "results")
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    run(args.data_dir, args.results_dir, args.repetitions, args.seed)
