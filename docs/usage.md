# Using momentous

This reference covers the input conventions, uncertainty model, and analysis
options. Start with the [README](../README.md) for a first example.

## Measurements and uncertainty

`MomentData(raw, covariance=...)` validates normalization, polarization, values,
and uncertainty coverage at construction and owns immutable snapshots. It accepts
`Moment` or tuple keys. `data.moments` and `data.values` retain raw values;
`analysis.moments`, `analysis.values`, and `analysis.covariance` are normalized.
Both expose `component_labels` and `observables` in scalar-vector order.

Every measurement requires raw `(0, 0)` or polarized `(0, 0, 0)`.
Compatibility analysis also requires at least one other moment; yield-only
extraction is valid. Selected higher moments are allowed; complete multiplets are
not required. Pre-normalized input is unsupported. H00=1 still uses the raw
convention; numbers alone cannot reveal a different normalization, conjugation,
frame, or polarized sign convention.

There is one uncertainty argument with three constructors:

```python
import numpy as np

# Full covariance with explicit scalar row/column labels.
h00, h11 = mo.Moment(0, 0), mo.Moment(1, 1)
matrix = np.diag([4.0, 0, 1.0, 0.25])
matrix[0, 2] = matrix[2, 0] = 0.5
covariance = mo.Covariance(matrix, components=[h00.real, h00.imag, h11.real, h11.imag])
data = mo.MomentData({h00: 100.0, h11: 30.0 + 4.0j}, covariance=covariance)

# Independent deviations must explicitly cover every supplied moment.
independent = mo.Covariance.from_uncertainties({h00: 2.0, h11: 1.0 + 0.5j})
independent_data = mo.MomentData(data.moments, covariance=independent)

# Exact values require no repeated labels or zero arrays.
exact_data = mo.MomentData(data.moments, covariance=mo.Covariance.exact())
```

For `n` unpolarized moments the full covariance has real shape `(2*n, 2*n)`.
Every moment occupies Re/Im slots, even if its measured value is real. A zero
imaginary measurement need not have zero uncertainty, except for the normalizer.
Labels describe matrix order; analysis aligns them to mapping order, preserving
all Re/Im and inter-moment correlations. Exact components have zero covariance
rows and columns. An ordinary complex `(n, n)` covariance cannot encode these
relationships without pseudo-covariance information and is rejected; see
[Neeser and Massey (1993)](https://www.isiweb.ee.ethz.ch/archive/massey_pub/pdf/BI437.pdf).

Polarized values are real observables with real covariance shape `(n, n)`.
Covariance must be finite, symmetric, and positive semidefinite; singular and
very small positive variances are supported. It describes the raw values supplied.

For independent unpolarized deviations, `0.1 + 0.2j` means separate Re/Im standard
deviations. A real deviation declares exact Im. All correlations are assumed zero;
use explicit zeros for exact entries and provide every moment, including H00.
Polarized deviations must be real. Standard deviations are retained without
squaring during normalization, preserving extreme common unit rescalings.

Both modes use Wigner-D moments: raw normalization divides unpolarized H(L,M)
by H00 and polarized H^alpha(L,M) by H^0(0,0). The denominator must be positive and real; imaginary
unpolarized H00 must be exact zero. Its normalized value is exactly one with
zero variance. Complete raw covariance propagates through the analytic first-order
Jacobian as `J @ V @ J.T`, retaining denominator correlations. This is a local
ratio approximation; see [NIST's law of propagation of uncertainty](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty).
`NormalizationWarning` flags a relative normalizer uncertainty of at least 1/3.
Absence of the warning is not an accuracy guarantee.

Raw moments use `D_LM = d_LM(theta) * exp(-1j*M*phi)` in both modes.
Unpolarized full-acceptance estimates are `sum(w * D_LM)`, with H00=`sum(w)`.
Convert earlier unit-normalized harmonic inputs with
`sqrt(4*pi/(2*L+1))` per raw value and covariance row/column; the normalized
predictions remain unchanged. The `analyze` docstring provides the explicit
full-acceptance estimators and a laddu example.

## Extracting moments with acceptance

`EventSample(source, ...)` accepts laddu datasets, mappings, structured NumPy
arrays, and tables with named column access. Supply numerical arrays directly by
omitting the source. The interface is the same for every input type.
No file format is required. All three samples must share the analysis frame;
accepted MC uses truth angles. MC must match the beam exposure and relevant
kinematic distribution. Reconstruction and wrong-hypothesis migration are outside
this acceptance-only model.

The workflow has three separately constructed samples and one response. For laddu datasets,
stored weights are used automatically; supply your own frame expressions when
angles are not already named columns:

```python
import laddu as ld

# Choose your own event-column names; none are assumed by the library.
events = ("run_number", "event_number")
beam = mo.Polarization()  # Reads P and Phi; omit for unpolarized samples.


def sample(path):
    return mo.EventSample(
        ld.read_parquet(path),
        costheta=costheta_expression,
        phi=phi_expression,
        events=events,
        polarization=beam,
    )


generated = sample("generated.parquet")
accepted = sample("accepted.parquet")
data = sample("data.parquet")
acceptance = mo.Acceptance(generated, accepted, basis=mo.MomentBasis(4, polarized=True))
extraction = acceptance.extract(data)
analysis = mo.analyze(
    extraction.data, mo.Waveset.from_max_l(2, reflectivities=("+", "-"))
)
print(extraction.diagnostics, extraction.covariance_scope)
```

`Acceptance` defaults to uniform angular generation and `MCStatistics.linked()`.
Only this object declares the relationship between MC samples: linking checks
shared physical-event IDs and supplied truth angles and beam information.
Accepted MC must provide its own truth coordinates; sample construction never
copies or replaces them. Rejected generated events need no accepted row.
Specify `MCStatistics.independent()` for genuinely independent Poisson MC, or
`data_only()` to condition on the response and exclude MC uncertainty.

`MomentBasis(max_L=4, polarized=False)` defines the complete expansion.
Omitting `basis` in `Acceptance` uses L=4 and the generated sample's polarization
mode. These defaults are scientific assumptions; change them when your generation
or intensity expansion differs.

`EventSample` reads `costheta` and `phi` by default. The polar cosine must lie in
[-1, 1]; azimuth is radians. Wigner-D evaluation converts to theta internally.
`Polarization()` reads `P` and `Phi`, or accepts custom column names, numerical arrays, or laddu expressions.
Magnitude is a fraction in [0, 1], orientation is radians relative to the
production plane, and only linear photon polarization is currently supported.
Zero polarization is allowed, but sufficient overall sensitivity is required.
laddu expressions evaluate together in one traversal, using double-precision JIT
without thread limits by default.

Event grouping is **required**. A string names one ID column; a tuple names a
compound key, such as `(run, event)`. Integers and strings retain exact identity.
Exclude combo numbers: hypotheses of one physical event must share an ID.
For arrays, use `EventGrouping.from_ids(ids)`; when each row is an independent
physical event, explicitly use `EventGrouping.independent_rows()`.
Data IDs have their own namespace and need not match MC.

The same interface works with other sources:

```python
# A dictionary, structured NumPy array, or table with source[name] access.
generated = mo.EventSample(generated_columns, events="parent_id", weights="weight")
accepted = mo.EventSample(accepted_columns, events="parent_id", weights="weight")
data = mo.EventSample(data_columns, events="data_id", weights="weight")
extraction = mo.Acceptance(generated, accepted).extract(data)
```

For ordinary sources, omitted weights mean unit weights. Arrays can also be
supplied directly as `costheta=...`, `phi=...`, and `weights=...`, without a source.
Never rescale hypothesis weights to sum to one: covariance sums outer products
of physical-event contributions, retaining cross terms between hypotheses.
Negative data subtraction weights are allowed; MC weights must be nonnegative.
Weight-estimation and calibration nuisance errors are not inferred from columns.
See [the complete extraction workflow](extraction.md#using-laddu-datasets)
for linked MC, polarized moments, and input conventions.

`MomentBasis.from_waves(pool)` includes every rank through twice the largest
wave rank. Extraction fits the complete basis, including nuisance coordinates,
and restores signed projections and exact zeros with their covariance. Omitted
higher intensity components can bias fitted moments through angular mixing.

`MCIntegration.uniform()` is the default and assumes uniform decay-angle
generation with matching beam exposure. For a known nonuniform density, use
`MCIntegration.importance(relative_density=expression)` with sample scalars
`costheta`, `phi`, and polarized `P`, `Phi`. It divides both MC samples by that
density; do not also include inverse density in their weights.
Accepted weights are divided by **generated exposure**, preserving efficiency.

MC uncertainty follows the policy selected in `Acceptance`:

- `MCStatistics.data_only()` conditions on the response and excludes MC errors.
- `MCStatistics.independent()` requires genuinely independent Poisson MC samples
  with comparable exposure; missing matching IDs do not establish independence.
- `MCStatistics.linked()` (the default) matches grouped IDs to a fixed-size
  generated sample, retaining generated/accepted and hypothesis correlations. Every accepted ID
  must have a generated counterpart; rejected events need no accepted row.

The result exposes `.data`, `.measured`, `.response`, `.data_covariance`,
`.mc_covariance`, and response diagnostics. Covariance propagation is first-order.
Singular or poorly conditioned responses raise `ExtractionError` with their
singular values; no coordinates are silently dropped or regularized.
See [the extraction derivation](extraction.md) for the complete weighted
estimator, statistical assumptions, and covariance equations.

## Wave pools and scalar observables

`Waveset` is an immutable canonical basis. `.waves` contains its individual waves;
`len(pool)` counts them. Iteration lazily yields **nonempty subset Wavesets**.
Powerset enumeration has exponential total size.

```python
pool = mo.Waveset([(0, 0), (1, 0), (1, 1)])
for waves in pool.powerset(min_size=1, max_size=2):
    print(waves)

unpolarized = mo.Waveset.from_max_l(2)
positive = mo.Waveset.from_max_l(2, reflectivities=("+",))
both_sectors = mo.Waveset.from_max_l(2, reflectivities=("+", "-"))
```

Generated pools include all signed projections. The single `reflectivities`
argument selects sectors; omission means unpolarized, and empty or duplicate
sectors are rejected. Explicit wavesets infer their mode from their waves. An
empty set can preserve mode with `Waveset([], polarized=True)`.

Quantum numbers accept integers, floats, `Fraction`, or laddu L/M and convert
internally. Orbital ranks and allowed projections are integral. Nonphysical
fractions, duplicate waves, and mixed polarization are rejected.

`Moment.real` and `Moment.imag` select real scalar `Observable` objects. They
work as covariance labels and region axes; canonical `(L,M,"real"/"imag")` tuple
shorthand remains accepted. Bounds retain canonical tuple keys, available through
`observable.key`. A polarized `Moment(L,M,variant)` already denotes a scalar.
Variant 2 means Im H^2, not a complex H^2 value; its `.imag` is consequently invalid.

```python
h0, h1, h2 = mo.Moment(0, 0, 0), mo.Moment(0, 0, 1), mo.Moment(1, 1, 2)
data = mo.MomentData(
    {h0: 100.0, h1: 90.0, h2: 0.0},
    covariance=mo.Covariance(np.diag([4, 4, 1]), components=[h0, h1, h2]),
)
polarized = mo.analyze(data, mo.Waveset.from_max_l(1, reflectivities=("+", "-")))
region = polarized.region(h1, h2)
```

Signs follow [Mathieu et al. (2019), Eq. (13)](https://arxiv.org/abs/1906.04841):
a pure positive-reflectivity S wave has H^1(0,0)/H^0(0,0)=+1.

## Checks, candidate views, and geometry

`analyze` prepares normalized geometry and checks the full pool. It accepts
independent numerical settings: `max_combination_size=2`, `n_sigma=3`, and
`tolerance=1e-8`. Size 1 checks individual scalar components; size 2 additionally
checks every continuous pair, including Re/Im of one moment and cross-variant
polarized pairs. Larger sizes are explicitly unsupported. There is no sampled
compatibility method. `n_sigma` is not a joint confidence level or p-value;
`tolerance` is separate absolute slack in normalized coordinates.

`analysis.for_waves(waves)` validates membership once and returns a candidate view
with `status()`, `check()`, `operators()`, `bounds()`, and `region(...)`. It shares
analysis caches. The analysis also provides these operations with an optional
candidate argument. Empty candidates check incompatible and have no geometry.

Checks return `compatible`, `incompatible`, or `unresolved`. Compare statuses
explicitly; their Boolean conversion raises. `check(...).valid` and Boolean check
conversion raise for unresolved outcomes. With `diagnostics=True`, a check returns
complete `ProjectionFailure` records for verified exclusions and `UnresolvedPair`
records with an available reason. Unresolved records claim no projection witness.
Detailed checks evaluate the actual candidate even when status could be inferred.

Operators follow `candidate.waves.waves` order. Polarized operators use a whitened
normalization metric. Regions return labels, measured point, covariance, support
normals/values, and an inner boundary approximation. `n_directions=360` changes
boundary display resolution only; point, segment, and flat-face degeneracies are
supported. Regions never determine compatibility verdicts. Advanced array-based
checks remain available as `momentous.geometry.check_pair`.

## Explicit subset search

```python
pool = mo.Waveset.from_max_l(1)
data = mo.MomentData({(0, 0): 1, (1, 0): 0}, covariance=mo.Covariance.exact())
analysis = mo.analyze(data, pool)
search = analysis.search(min_size=1, max_size=3)
for waves in search.wavesets(status="compatible"):
    print(waves)
minimal = search.minimal_wavesets()
```

`SearchResult` owns `count`, `candidate_count`, `size_limits`, `complete`,
`unresolved`, `status(waves)`, `wavesets(status=...)`, and `minimal_wavesets()`.
Its `.analysis` refers to the prepared analysis. Repeated searches can use different
size domains while reusing numerical certificates. Search lookup requires its
selected domain; `analysis.check(waves)` and candidate views accept any pool subset.

Search retains compressed disjoint passing blocks. Counts and lookup do not expand
supersets; enumeration is lazy, but output and worst-case search remain exponential.
Unresolved outcomes never establish passes or exclusions and never prune branches.
Minimal sets are certified by proving every admissible immediate subset incompatible.
Unresolved smaller subsets prevent certification, and incomplete search may omit
true minima. Minimality is relative to `size_limits`. See [algorithm notes](algorithms.md).

Both `wavesets()` and `minimal_wavesets()` accept `include` and `exclude` wavesets:

```python
required = mo.Waveset([(0, 0), (1, 0)])
forbidden = mo.Waveset([(1, 1)])
for waves in search.wavesets(include=required, exclude=forbidden):
    print(waves)

excluded_candidates = search.wavesets(
    status="incompatible", include=required, exclude=forbidden
)
minimal = search.minimal_wavesets(include=required, exclude=forbidden)
```

Every included wave must be present; every excluded wave must be absent. The
filters also apply to `status="unresolved"`. Empty wavesets impose no restriction.
Waves outside the pool, incompatible polarization modes, and a wave appearing in
both filters are rejected. Reflectivity is part of a wave's identity.

Filtering preserves the original size limits, counts, completeness, and candidate
statuses. Compatible branches are restricted before their subsets are expanded.
Filtering minima selects among the original certified minima, so adding required
waves does not redefine minimality.

Supply an `on_check(waves, result, progress)` callback to observe numerical
evaluations and search progress. It receives immutable `Waveset`, `CheckResult`,
and `CheckProgress` objects. During a search, `progress.completed` and
`progress.total` count candidates in the selected size domain, including entire
pruned branches, cached decisions, and unresolved leaves. `progress.evaluated`
distinguishes new numerical evaluations from branch-completion events.

The initial pool check, direct numerical checks, and minimal-set certification
report `phase="check"` and `completed=total=1`. Cached and inferred checks outside
a search remain silent. Callbacks run on the calling thread, including parallel
searches, and exceptions propagate after retaining completed numerical checks.
Choose any progress-display library, for example tqdm:

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

The number of numerical evaluations is still unknown ahead of time, but the
size-limited candidate total is exact. Update from absolute counts instead of
adding one per callback: pruning can process many wavesets in one event. Use a
new bar for each search. A completed traversal can still contain unresolved
outcomes; inspect `search.complete` separately. momentous has no progress-display
dependency; tqdm is used by the demo.

`workers=1` selects serial search. By default, ordinary Python uses up to two workers and
Python with the GIL disabled uses up to eight workers. Explicit thread counts work
on either build, and workers share prepared inputs while owning their solver
templates. Additional threads can increase overhead; benchmark your workload.
CVXPY's current `_cvxcore` extension re-enables the GIL when loaded on free-threaded
Python. Momentous respects that runtime behavior; numerical array operations can
still run concurrently. laddu and native numerical thread settings are unchanged.
