# momentous

momentous is a Python library for extracting acceptance-corrected moments, checking partial-wave hypotheses, and finding the smallest compatible wavesets. It keeps uncertainties and correlations with your measurements throughout the analysis.

The library supports unpolarized measurements and linear photon polarization. It requires Python 3.12 or newer.

## Install with pip

```sh
pip install momentous
```

## Install with uv

Add momentous to your project:

```sh
uv add momentous
```

You can also install from a local checkout:

```sh
uv add /path/to/momentous
```

To work on the library or run its demos, enter the checkout and install the
development environment:

```sh
uv sync --locked
uv run python scripts/demo.py
```

If you are new to uv, its [installation guide](https://docs.astral.sh/uv/getting-started/installation/) will get you started.

## Start with a measurement

Supply raw moments, their uncertainty, and the pool of waves you want to explore:

```python
import momentous as mo

data = mo.MomentData(
    {(0, 0): 100.0, (1, 1): 20.0 + 4.0j, (2, 0): -5.0},
    covariance=mo.Covariance.from_uncertainties(
        {(0, 0): 2.0, (1, 1): 1.0 + 0.5j, (2, 0): 1.0}
    ),
)
pool = mo.Waveset.from_max_l(2)
analysis = mo.analyze(data, pool)

candidate = analysis.for_waves(mo.Waveset([(0, 0), (1, 1)]))
print(candidate.status())
print(candidate.check(diagnostics=True))
print(candidate.bounds())

h11 = mo.Moment(1, 1)
region = candidate.region(h11.real, h11.imag)

search = analysis.search(max_size=3)
print(search.count)
print(search.minimal_wavesets())
```

Wave labels use spectroscopic notation: `Wave(2, 2, "+")` prints as `D+2(+)`, where the signed projection follows the orbital label and the reflectivity is in parentheses. Moments print as `H(2, 1)`; polarized variants retain their superscript, such as `Im H^2(2, 1)`. A waveset prints its canonically ordered wave labels, for example `{S0(+), D+2(+)}`.

The analysis checks individual scalar components and every pair by default, using `n_sigma=3`. A passing candidate satisfies these necessary moment constraints; it does **not** establish that one simultaneous amplitude fit exists. A numerical failure to resolve a check produces an explicit `unresolved` status.

Search is a separate step, so you can inspect a candidate or its geometry first. Prepared operators and completed checks are reused across candidates and searches. `minimal_wavesets()` certifies minimality within the chosen search size limits.

You can select results that contain particular waves and omit others:

```python
required = mo.Waveset([(0, 0), (1, 1)])
forbidden = mo.Waveset([(2, 0)])
for waves in search.wavesets(include=required, exclude=forbidden):
    print(waves)

minimal = search.minimal_wavesets(include=required, exclude=forbidden)
```

`include` requires all listed waves; `exclude` forbids any listed wave. These filters retain the original search size limits. Filtering minima selects among the original certified minima; it does not redefine minimality with the required waves forced into each candidate.

## Moments and uncertainties

Every measurement must include raw `(0, 0)`, or `(0, 0, 0)` for polarized input. Other moment labels may be selected as needed; complete multiplets are not required. momentous divides by this positive, real normalization moment and propagates its correlations using a first-order Jacobian. Pre-normalized input is unsupported.

The convention is the same Wigner-D convention in both modes: `D_LM = d_LM(theta) * exp(-1j*M*phi)`. At full acceptance, unpolarized raw moments are `sum(w * D_LM)`, with `H00 = sum(w)`. Frame, phase, and polarized sign conventions must match the library; requiring raw input cannot detect a convention mismatch. The [`analyze` docstring](src/momentous/analysis.py) gives the estimators, including a short example.

Choose the covariance constructor that describes your measurement:

- `Covariance.from_uncertainties(...)` takes independent standard deviations for every supplied moment, including the normalizer. For unpolarized moments, `1.0 + 0.5j` means separate real and imaginary deviations.
- `Covariance(matrix, components=...)` takes a full, explicitly labeled **real** covariance matrix. Each unpolarized moment has a real and an imaginary slot,
  even when its measured value is real. Polarized moments have one scalar slot.
- `Covariance.exact()` declares exact input.

For example, correlations between the normalizer and a complex moment can be specified directly:

```python
import numpy as np

h00, h11 = mo.Moment(0, 0), mo.Moment(1, 1)
matrix = np.diag([4.0, 0.0, 1.0, 0.25])
matrix[0, 2] = matrix[2, 0] = 0.5
covariance = mo.Covariance(matrix, components=[h00.real, h00.imag, h11.real, h11.imag])
data = mo.MomentData({h00: 100.0, h11: 30.0 + 4.0j}, covariance=covariance)
```

See the [input and uncertainty reference](docs/usage.md#measurements-and-uncertainty) for covariance ordering, singular uncertainties, and normalization assumptions.

## Extract moments from events

Build three separate samples: data, generated MC, and accepted MC. An `Acceptance` object describes their response and the statistical relationship between the MC samples.

With `laddu` datasets, supply expressions for your analysis frame and identify the column or columns that represent a physical event:

```python
import laddu as ld
import momentous as mo


# Define costheta_expression and phi_expression for your own analysis frame.
def sample(path):
    return mo.EventSample(
        ld.read_parquet(path),
        costheta=costheta_expression,
        phi=phi_expression,
        events=("run_number", "event_number"),
    )


generated = sample("generated.parquet")
accepted = sample("accepted.parquet")
data_events = sample("data.parquet")

pool = mo.Waveset.from_max_l(2)
acceptance = mo.Acceptance(generated, accepted, basis=mo.MomentBasis.from_waves(pool))
extraction = acceptance.extract(data_events)
analysis = mo.analyze(extraction.data, pool)
search = analysis.search()
```

`laddu`'s stored event weights are used automatically. Accepted rows can include multiple hypotheses for one physical event; give them the same event ID so that their covariance includes the cross terms.

For polarized extraction, give each sample a `polarization=mo.Polarization(...)` and use `Waveset.from_max_l(2, reflectivities=("+", "-"))`. `Polarization()` reads `P` and `Phi` by default; custom columns, arrays, and `laddu` expressions work too. `MomentBasis.from_waves(pool)` includes the required ranks and polarization mode.

The same sample constructor accepts mappings, structured NumPy arrays, and tables with named column access:

```python
generated = mo.EventSample(generated_columns, events="parent_id", weights="weight")
accepted = mo.EventSample(accepted_columns, events="parent_id", weights="weight")
data_events = mo.EventSample(data_columns, events="data_id", weights="weight")
extraction = mo.Acceptance(generated, accepted).extract(data_events)
```

Samples use `costheta` and `phi`, with azimuth in radians. Without a source, you can pass numerical arrays directly and group them with `EventGrouping.from_ids(ids)`.

By default, acceptance assumes uniform angular generation and linked MC. Use `MCIntegration.importance(...)` for a known nonuniform generation density or `MCStatistics.independent()` for genuinely independent MC samples. The [extraction guide](docs/extraction.md) explains weights, polarization, MC exposure, and uncertainty propagation.

## Follow search progress

Bring your preferred progress library. The callback receives a `CheckProgress` snapshot with `completed` and `total`, on the calling thread even during parallel searches. Update the bar from those absolute counts:

```python
from tqdm.auto import tqdm

with tqdm(desc="Wavesets", unit="waveset") as bar:

    def on_check(waves, result, progress):
        if progress.phase == "search":
            bar.total = progress.total
            bar.update(progress.completed - bar.n)

    analysis = mo.analyze(data, pool, on_check=on_check)
    search = analysis.search(workers=2)

minimal = search.minimal_wavesets()
```

The total includes every waveset within the search's size limits. Pruning or cached decisions can resolve several at once, making the bar jump without additional numerical evaluations. `progress.evaluated` identifies new numerical checks; outside searches those report `phase="check"` and `completed=total=1`. Use a new bar for each search, and inspect `search.complete` for unresolved results even when progress reaches 100%. See the [search reference](docs/usage.md#explicit-subset-search) and [performance notes](docs/performance.md) for worker settings and free-threaded Python behavior.

## Try the demos and contribute

The demo generates events with `laddu`, applies a detector cut, extracts moments, and scans all nonempty wavesets. Its amplitudes are fixed in the script, and moment ranks extend through twice `--max-wave-l`.

```sh
uv run python scripts/demo.py --events 20000 --mc-events 50000
uv run python scripts/demo.py --polarized --max-wave-l 2 --n-sigma 3
```

To install the commit hook and run all checks:

```sh
uv sync --locked
uv run prek install
uv run prek run --all-files
```

prek (or your favorite pre-commit tool) runs Ruff lint and formatting, ty, pytest (including docstring examples), and yamloom sync using the versions in `uv.lock`. Edit `.yamloom.py` to change GitHub workflows, then run `uv run yamloom sync`. CI also checks the installed wheel on Python 3.12, 3.13, and 3.14.

For more detail, see the [API and conventions reference](docs/usage.md), [algorithm notes](docs/algorithms.md), and [release guide](docs/releasing.md).

> [!NOTE]
> This library was developed with the help of Codex
