# Search performance

Pass `on_check(waves, result)` to `mo.analyze` to observe each completed numerical
candidate evaluation. The callback remains attached to the prepared analysis and
covers its initial pool check, direct checks, searches, and minimal certification.
Cached and inferred decisions are silent. Callbacks run on the calling thread,
including parallel searches, and exceptions propagate after retaining completed work.

Momentous does not choose a display library. The demo uses tqdm directly:

```python
from tqdm.auto import tqdm

with tqdm(desc="Waveset checks", unit="check") as progress:
    analysis = mo.analyze(data, pool, on_check=lambda waves, result: progress.update())
    result = analysis.search()
    minimal = result.minimal_wavesets()
```

The number of numerical evaluations is unknown before the search because pruning
resolves entire branches without checking their members. The demo therefore
reports completed checks, elapsed time, and throughput rather than a misleading
percentage or ETA based on the total powerset. Inspect `result.complete` for
unresolved outcomes. tqdm is a development dependency, not a library dependency.

The default uses up to two search threads on ordinary Python and up to eight when
the GIL is disabled, bounded by available CPUs. Set `workers=1` for serial execution
or provide a positive thread count. Prepared arrays are shared, solver templates
are private to each worker, and one coordinator updates certificates and callbacks.
Native numerical and laddu thread settings are unchanged.

Free-threaded Python 3.14 was tested. Importing momentous preserves its disabled
GIL. Loading CVXPY's current `_cvxcore` extension for a conic solve re-enables it;
momentous does not override that safety decision. NumPy operations can still run
concurrently, and explicit workers remain supported after the GIL is enabled.

## Measurements

Measured on this development machine with Python 3.12, laddu 0.26.0, fixed demo
seed 42, 20,000 data events, 50,000 generated MC events, both reflectivities through
l=2, moments through L=4, and n_sigma=3. Search includes all 262,143 nonempty
subsets. Times exclude event generation and moment extraction.

| Search mode | Seconds | Compatible candidates | Unresolved |
| --- | ---: | ---: | ---: |
| Optimized, serial | 241.7 | 62,651 | 0 |
| Optimized, two workers | 156.6 | 62,651 | 0 |

The original default demo was reported to run beyond ten minutes before cancellation;
that is a lower bound, not a completed baseline timing. For a smaller fixed
12-wave benchmark, profiling the previous and optimized serial implementations
gave 38.9 and 7.8 seconds respectively. All 4,095 candidate statuses and certified
minimal wavesets also matched the previously installed library. Runtime depends
on the input covariance, pruning opportunities, native libraries, and CPU load.
Extra threads are not always faster: two beat four on the smaller benchmark.

The main savings come from batching conservative hull witnesses, removing exact
duplicate reference points, and sharing exactly equivalent moment-pair checks.
Repeated-component pairs retain their stricter joint Euclidean tolerance, full
diagnostics still include every labeled pair, and solver fallback preserves
tangency, singular covariance, and tiny positive variances.

Compare worker counts on the same extracted measurement:

```sh
uv run python scripts/benchmark.py --polarized --pool-l 2 --workers 1 --workers 2 --time-limit 600
```

Search retains compressed passing blocks. Counts and status lookup avoid expanding
passing supersets; requested enumeration and worst-case search remain exponential.
Passing is a necessary moment-compatibility condition, not an amplitude fit.
