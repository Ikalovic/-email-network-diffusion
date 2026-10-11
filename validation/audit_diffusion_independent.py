"""Small independent audit; no production code changes or full experiment rerun."""
import importlib.util
from itertools import product
from pathlib import Path
import sys

import numpy as np


source = Path(__file__).resolve().parents[1] / 'run_experiment.py'
spec = importlib.util.spec_from_file_location('diffusion_under_audit', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def round_ic(n, edge_list, live, seeds):
    """Synchronous IC oracle: each newly active source tries its outgoing edges once."""
    active, frontier = set(seeds), set(seeds)
    waves = [-1] * n
    for s in seeds:
        waves[s] = 0
    wave = 0
    while frontier:
        new = {v for (u, v), is_live in zip(edge_list, live)
               if u in frontier and v not in active and is_live}
        wave += 1
        for v in new:
            waves[v] = wave
        active.update(new)
        frontier = new
    return np.array([v in active for v in range(n)]), np.array(waves)


def enumerate_graph(name, n, edges, seeds, p, theoretical):
    sources, targets = map(np.array, zip(*edges))
    expected = 0.0
    states = 0
    for live in product([False, True], repeat=len(edges)):
        mask = np.array(live, dtype=bool)
        adjacency = module.build_adjacency(n, sources, targets, mask)
        active, waves = module.reachable(adjacency, np.array(seeds))
        oracle_active, oracle_waves = round_ic(n, edges, live, seeds)
        np.testing.assert_array_equal(active, oracle_active)
        np.testing.assert_array_equal(waves, oracle_waves)
        weight = p ** sum(live) * (1 - p) ** (len(edges) - sum(live))
        expected += weight * active.sum()
        states += 1
    assert abs(expected - theoretical) < 1e-12, (name, expected, theoretical)
    print(f'PASS {name}: {states} exact live-edge states; E[total]={expected:.12g}')


p = 0.5
chain = [(0, 1), (1, 2), (2, 3)]
enumerate_graph('directed chain', 4, chain, [0], p, 1+p+p**2+p**3)
enumerate_graph('fork', 3, [(0, 1), (0, 2)], [0], p, 1+2*p)
diamond = [(0, 1), (0, 2), (1, 3), (2, 3)]
enumerate_graph('two convergent paths', 4, diamond, [0], p, 1+2*p+2*p**2-p**4)
enumerate_graph('two distinct sources can try one target', 4, diamond, [1, 2], p, 2+2*p-p**2)
enumerate_graph('cycle does not retry active sources', 3, [(0, 1), (1, 2), (2, 0)], [0], p, 1+p+p**2)

edge_sources, edge_targets = map(np.array, zip(*chain))
uniforms = np.array([0.0, 0.5, np.nextafter(1.0, 0.0)])
for probability, expected in [(0.0, [True, False, False, False]),
                              (1.0, [True, True, True, True])]:
    adjacency = module.build_adjacency(4, edge_sources, edge_targets, uniforms < probability)
    actual, waves = module.reachable(adjacency, np.array([0]))
    np.testing.assert_array_equal(actual, expected)
full_adjacency = module.build_adjacency(4, edge_sources, edge_targets, np.ones(3, dtype=bool))
actual, _ = module.reachable(full_adjacency, np.array([3]))
np.testing.assert_array_equal(actual, [False, False, False, True])
print('PASS p=0, p=1, and reverse reachability is absent')

sources = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
targets = np.array([1, 2, 4, 0, 3, 5, 0, 3, 0, 2])
network = module.Network(ids=np.array([40, 10, 30, 20, 50, 60]),
                         departments=np.array([0, 0, 1, 1, 2, 2]),
                         department_sizes=np.array([2, 2, 2]),
                         sources=sources, targets=targets,
                         out_degree=np.bincount(sources, minlength=6), metadata={})
seeds = module.fixed_seed_sets(network, 2)
np.testing.assert_array_equal(seeds['degree'], [1, 0])
np.testing.assert_array_equal(seeds['dispersed'], [1, 3])
seeds3 = module.fixed_seed_sets(network, 3)
np.testing.assert_array_equal(seeds3['dispersed'], [1, 3, 4])
print('PASS seed order: out-degree descending, original-ID ties, distinct departments')

active = np.array([True, True, True, False, True, False])
metrics, counts = module.outcome(network, active, np.array([0, 2]))
np.testing.assert_array_equal(counts, [1, 0, 1])
assert metrics['total_reached'] == 4 and metrics['new_reached'] == 2
assert metrics['new_reached_fraction'] == 0.5
assert metrics['departments_with_new_reach'] == 2
assert abs(metrics['macro_department_new_fraction'] - 1/3) < 1e-12
only_seeds = np.array([True, False, True, False, False, False])
metrics0, counts0 = module.outcome(network, only_seeds, np.array([0, 2]))
assert metrics0['new_reached'] == 0 and metrics0['departments_with_new_reach'] == 0
assert counts0.sum() == 0
print('PASS metrics exclude seed NODES; seed departments may count via other activated members')
print('Audit passed; no production experiment was rerun.')
