# Historical demo benchmarks

Single wall-clock runs on this checkout, Python 3.14.6, 20,000 events, seed 42,
moments through L=4, and 3-sigma covariance ellipses. Event generation is excluded
from the reported search times. Timings include numerical witnesses and any conic
fallbacks; they are examples, not worst-case bounds.

| Mode / domain | Candidates | Search time | Passing | Complete |
| --- | ---: | ---: | ---: | --- |
| Unpolarized, 9 waves | 511 | 0.78 s | 124 | yes |
| Unpolarized, 16 waves | 65,535 | 55.94 s | 35,128 | yes |
| Polarized, 18 waves, size <= 3 | 987 | 2.94 s | 1 | yes |
| Polarized, all 18-wave subsets | 262,143 | 180.01 s | unknown | no, time limit |

The independent exhaustive continuous check of the 511-set case took 3.09 s and
returned exactly the same sets. The legacy 180-angle validator took 77.06 s and
also agreed for this dataset; finite angular sampling is not equivalent in general.
The initial continuous prototype took 40.32 s including preparation for this case,
versus 0.91 s after attainable-hull shortcuts, caching, and monotonic inference.

Complete-pool checks (one waveset, **not** all its subsets) took 0.14/0.035 s for
25/36 unpolarized waves and 0.15/0.068 s for 32/50 polarized waves. These easy
passing cases needed no conic solves. CSV/JSON files retain the separate preparation,
check/search times and counters. `--full-only` records no completed search.

These checked-in timings and CSV files are historical measurements of the
previous interface. The current `scripts/benchmark.py` measures public `analyze`
workflows, combines preparation with its first pool check, and reports a separate
fresh search-analysis time including preparation. It compares small searches with
fresh candidate checks and no longer runs a sampled baseline. A subprocess
wall-clock limit applies to the entire case, preserving completed stages and
recording timeouts explicitly. See the commands in the project README.
Searches can remain exponential even when individual large-set checks are fast.
