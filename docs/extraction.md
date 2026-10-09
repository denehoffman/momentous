# Acceptance-corrected moment extraction

`Acceptance(...).extract(data)` solves a complete finite angular
expansion and returns raw `MomentData`. It requires data, generated MC, and
accepted MC. Column sources and numerical arrays are supported without prescribing
a file format. All angular evaluation uses laddu Wigner-D expressions with
double-precision JIT by default and no thread limit.

## One sample interface

`EventSample(source, ...)` reads named columns from a mapping, a structured NumPy
array, or a table implementing `source[name]`. It also accepts laddu datasets and
their expressions. To provide arrays directly, omit `source`. Column names are
source-independent: `EventGrouping.from_columns("run", "event")` and
`Polarization()` work with every named-column source. laddu expressions require
a laddu dataset; with other sources, pass calculated arrays or their column names.

`costheta` and `phi` default to those column names. The polar cosine must lie in
[-1, 1]; azimuth is radians. Conversion to theta occurs internally for Wigner-D
evaluation. All values are snapshotted in the source's current row order. laddu's native weights are preserved automatically;
other sources use unit weights unless a weight column or array is supplied.

```python
import momentous as mo

events = ("run_number", "event_number")
beam = mo.Polarization()
generated = mo.EventSample(
    generated_columns, events=events, weights="weight", polarization=beam
)
accepted = mo.EventSample(
    accepted_columns, events=events, weights="weight", polarization=beam
)
data = mo.EventSample(data_columns, events=events, weights="weight", polarization=beam)
```

Here each `*_columns` value can be an ordinary dictionary or a named-column table.
The same constructors apply when those values are laddu datasets; omit `weights`
to retain laddu's existing weight-column convention. Each sample provides its own
angles and beam information. Their relationship is declared by `Acceptance`, not
by sample construction. `events` is required and has no default column name; a
string names one column and a tuple names a compound key. The explicit
`EventGrouping.from_columns(...)` constructor remains available.

## Using laddu datasets

laddu 0.26+ preserves exact integer columns and evaluates a named expression
mapping in one traversal. momentous uses these capabilities automatically.
Read files through laddu; a separate Parquet reader is unnecessary.

This polarized example extracts every moment through L=4 and scans waves through
l=2 in both reflectivity sectors. The files here contain `costheta`, `phi`, `P`,
and `Phi` columns. Supply your own laddu expressions for the analysis frame when
needed. `costheta` is in [-1, 1], `phi` is radians, `P` is a fraction, and `Phi`
is the polarization orientation relative to the production plane.

```python
import laddu as ld
import momentous as mo

events = ("run_number", "event_number")
beam = mo.Polarization()

generated = mo.EventSample(
    ld.read_parquet("generated.parquet"),
    events=events,
    polarization=beam,
)
accepted = mo.EventSample(
    ld.read_parquet("accepted.parquet"),
    events=events,
    polarization=beam,
)
data = mo.EventSample(
    ld.read_parquet("data.parquet"),
    events=events,
    polarization=beam,
)
acceptance = mo.Acceptance(
    generated=generated,
    accepted=accepted,
    basis=mo.MomentBasis(4, polarized=True),
)
extraction = acceptance.extract(data)
pool = mo.Waveset.from_max_l(2, reflectivities=("+", "-"))
search = mo.analyze(extraction.data, pool).search()
print(extraction.diagnostics)
print(search.minimal_wavesets())
```

Column names are reusable declarations: each sample reads its own current
source's IDs. Integers retain exact values even above 2**53; floating-point
ID columns are rejected. Use one column if event numbers are globally unique,
or a compound key to distinguish runs. Multiple combos share their physical-event
key; including a combo number would incorrectly declare them independent.
The data IDs have their own namespace and need not match MC. If data has one
physical event per row and no ID columns, explicitly use
`events=mo.EventGrouping.independent_rows()` for that sample.

`Acceptance(generated, accepted, ...)` defaults to `MCStatistics.linked()` and
`MCIntegration.uniform()`. Every accepted event ID must have a generated
counterpart; repeats and reordered accepted rows are supported. All hypotheses
of an MC event must share truth angles and beam information. Linking checks that
accepted truth coordinates match generated coordinates; it never substitutes them.
Both samples must explicitly supply polarization for a polarized workflow.
Generated MC must represent the experimental beam exposure. These checks do not
establish that its exposure distribution matches the data.

`MomentBasis(max_L=4, polarized=False)` defines a complete expansion. When
`basis` is omitted, `Acceptance` uses L=4 with the generated sample's mode.
For other truncations or nonuniform generation, supply the appropriate basis and
integration object. Independent Poisson MC requires `MCStatistics.independent()`;
`data_only()` explicitly excludes finite-MC uncertainty. These policies are never
inferred from missing or coincidentally matching IDs.

All dataset weights come from laddu's existing weight-column specification.
Accepted weights are neither multiplied by generated weights nor rescaled to an
event sum of one. Data and accepted MC may each have multiple weighted rows.
Angles and beam expressions are evaluated together; materializing IDs and weights
can require additional traversals for streaming sources. The constructors produce
owned numerical snapshots, not a lazy end-to-end extraction.

For an unpolarized workflow, omit polarization in all samples, use
`MomentBasis(4)`, and construct `Waveset.from_max_l(2)`.
An exhaustive two-sector l=2 pool has 262143 nonempty subsets; a search size limit
such as `.search(max_size=3)` narrows the domain when appropriate.

## Using other data sources

Supply numerical columns and IDs directly; pandas, Arrow, ROOT, and other readers
do not become library dependencies:

```python
generated = mo.EventSample(
    costheta=generated_costheta,
    phi=generated_phi,
    weights=generated_weights,
    events=mo.EventGrouping.from_ids(generated_ids),
)
accepted = mo.EventSample(
    costheta=accepted_truth_costheta,
    phi=accepted_truth_phi,
    events=mo.EventGrouping.from_ids(accepted_ids),
    weights=accepted_weights,
)
data = mo.EventSample(
    costheta=data_costheta,
    phi=data_phi,
    weights=data_weights,
    events=mo.EventGrouping.from_ids(data_ids),
)
```

The same `Acceptance(...).extract(data)` workflow then applies. Numerical
beam arrays or constants use `Polarization(magnitude=..., angle=...)`. All samples
are independently constructed; linked statistics check MC IDs and truth coordinates.

## Convention and independent coordinates

Use the Condon–Shortley convention implemented by laddu:

\[
D^L_{M0}(\phi,\theta,0)
=d^L_{M0}(\theta)e^{-iM\phi}
=\sqrt{\frac{4\pi}{2L+1}}Y_{LM}^{*}(\theta,\phi).
\]

Unpolarized raw moments are

\[
H_{LM}=\int I(\Omega)D^L_{M0}(\Omega)\,d\Omega.
\]

Thus H00 is total intensity. Both modes normalize by their raw H00, without
rank factors. To migrate earlier unpolarized unit-normalized harmonic inputs,
multiply each raw moment by \(s_L=\sqrt{4\pi/(2L+1)}\), and covariance by
\(S V S^\mathsf T\), with s_L repeated for each real/imaginary slot. The previous
normalized ratio equals the new one exactly:

\[
\frac{H^{\rm old}_{LM}}{\sqrt{2L+1}H^{\rm old}_{00}}
=\frac{H^{\rm new}_{LM}}{H^{\rm new}_{00}}.
\]

Only independent real coordinates are solved. Write
\(r_{LM}=\operatorname{Re}D^L_{M0}\),
\(t_{LM}=\operatorname{Im}D^L_{M0}\), and
\(a_{LM}=(2L+1)(2-\delta_{M0})\).
For unpolarized intensity, the test functions f are r for M >= 0 and t for M > 0.
The intensity-expansion functions c_j=a_j f_j give

\[
\rho(x;H)=c(x)^\mathsf T H.
\]

Here the reference measure is normalized solid angle \(d\Omega/(4\pi)\), so
rho is 4*pi times the intensity per solid angle. Its expansion coefficients H
are still the raw moments defined above.

For linear polarization, use the existing Mathieu signs. The three test-function
families, with the same angular factors a, are

\[
f^{0}_{LM}=r_{LM},\qquad
f^{1}_{LM}=P\cos(2\Phi)r_{LM},\qquad
f^{2}_{LM}=P\sin(2\Phi)t_{LM}.
\]

The coefficients are H^0, H^1, and Im H^2 respectively. The positive sine term
is consistent with D's negative exponential and the existing convention. Under
full acceptance and uniform beam orientation, the familiar direct dual
estimators are Re D, 2*cos(2*Phi)/P*Re D, and 2*sin(2*Phi)/P*Im D. The response
solver uses the forward expansion instead: it never divides an event by P.
Zero-polarization events are allowed, but sufficient overall sensitivity is
needed to distinguish the polarized coefficients. See
[Mathieu et al., equations 13 and 38](https://arxiv.org/html/1906.04841).

The polarized reference measure combines normalized solid angle with the
experimental beam-exposure distribution. Generated MC must reproduce that
exposure, including polarization magnitudes and orientations. Nonuniform beam
orientations are supported when their exposure is represented correctly in MC.
With nonuniform orientations, corrected H^0_00 is the unpolarized yield
coefficient, which need not equal the observed yield divided by a scalar efficiency.

Public output restores exact zeros and signed projections:

\[
H_{L,-M}=(-1)^M H_{LM}^{*},\qquad \operatorname{Im}H_{L0}=0,
\]

\[
H^{0,1}_{L,-M}=(-1)^M H^{0,1}_{LM},\qquad
\operatorname{Im}H^{2}_{L,-M}=-(-1)^M\operatorname{Im}H^{2}_{LM},\qquad
\operatorname{Im}H^{2}_{L0}=0.
\]

The same linear transformation restores covariance, so duplicated signed
coordinates are perfectly correlated and exact zeros have zero covariance.

## Acceptance, importance sampling, and hypothesis weights

Let q be the MC generation density relative to the reference measure and v=1/q.
`MCIntegration.uniform()` explicitly declares q=1. For a known nonuniform q,
`MCIntegration.importance(relative_density=...)` evaluates a positive real laddu
expression using `costheta`, `phi`, and polarized `P`, `Phi` on both MC samples.
The overall normalization of q cancels. Do not also put 1/q in the sample weights.

Supplied generated weights represent exposure; accepted weights represent the
compatible exposure and selection/hypothesis weighting. MC weights must be
nonnegative, with positive generated exposure. They must recover the stated
reference measure after the inverse-density correction. Arbitrary physics
reweighting can change that measure and must not be mislabeled as uniform MC.

For each generated physical event e, sum its row contributions:

\[
z_e=\sum_{h\in e,\,\mathrm{generated}}w_hv_h,
\qquad Z=\sum_e z_e.
\]

For each accepted physical event e, form

\[
u_e=\sum_{h\in e,\,\mathrm{accepted}}
w_hv_h f(x_h)c(x_h)^\mathsf T,
\qquad U=\sum_e u_e,
\qquad \widehat R=\frac{U}{Z}.
\]

For an ordinary accepted subset, u_e is zero for rejected generated events.
All hypotheses of a physical MC event must share its truth coordinates.
Multiple accepted hypotheses contribute their supplied weights without rescaling.
For example, weights 0.2 and 0.3 produce a total event weight of 0.5, not one.
An event weight can exceed one: this is then a weighted selection response,
not a literal acceptance probability. The analysis weighting must be represented
consistently in accepted MC.

Under this finite expansion, the response estimates

\[
R=\int \epsilon(x)f(x)c(x)^\mathsf T\,d\mu(x),
\]

where epsilon includes the effective selection weighting. Accepted MC is divided
by **generated** exposure. Dividing by accepted exposure would remove efficiency
and bias the corrected yield. Data contributions and the estimator are

\[
g_e=\sum_{h\in e,\,\mathrm{data}}w_hf(x_h),\qquad
y=\sum_e g_e,\qquad \widehat H=\widehat R^{-1}y.
\]

The implementation uses a linear solve, not an explicit inverse. The matrix
approach handles angular mixing rather than dividing each moment by a scalar
efficiency; see the general moment-unfolding approach in
[Beaujean et al.](https://arxiv.org/abs/1503.04100).

Extraction uses every coordinate through the chosen maximum rank. Unknown higher
moments can mix into retained moments through acceptance and cause truncation
bias. `MomentBasis.from_waves(pool)` includes ranks through twice the largest
wave rank; this is sufficient only if the true intensity is described by those
waves. Sparse output interests do not justify a sparse correction basis.

## Data covariance: cluster hypotheses before squaring

Declare physical-event grouping explicitly. `EventGrouping.from_ids(...)` joins
hypotheses; `independent_rows()` declares each row independent. Compound IDs such
as `(run, event)` avoid merging unrelated events. IDs need not match across data
and MC.

For independent compound-Poisson physical events, conditional on supplied weights,

\[
\widehat V_y=\sum_e g_eg_e^\mathsf T.
\]

Two hypotheses from one event give

\[
\begin{aligned}
g_eg_e^\mathsf T={}&w_1^2f_1f_1^\mathsf T+w_2^2f_2f_2^\mathsf T\\
&+w_1w_2(f_1f_2^\mathsf T+f_2f_1^\mathsf T).
\end{aligned}
\]

Independent-row covariance would lose the second line. Negative data subtraction
weights are allowed; the complete event outer product remains positive semidefinite.

For B=R^-1, data-only propagation is

\[
V_H^{\rm data}=B V_y B^\mathsf T.
\]

This is a Poisson event-yield model, not a fixed-count covariance of sample means.
The raw yield carries counting uncertainty. Subsequent ratio normalization
retains all denominator correlations and makes normalized H00 exactly one with
zero variance. Error propagation assumes independent physical events; estimated
weights with shared fit parameters need additional nuisance covariance.

## Finite-MC covariance: explicitly declare the relationship

With \(R=U/Z\), first-order variations obey

\[
\delta R=\frac{\delta U-R\delta Z}{Z},\qquad
\delta H=B\delta y-B(\delta R)\widehat H.
\]

Data and MC are assumed independent. Each policy below therefore reports
\(V_H=V_H^{\rm data}+V_H^{\rm MC}\).

### Conditional data-only policy

`MCStatistics.data_only()` holds the estimated response fixed. It reports
\(V_H^{\rm MC}=0\) as **excluded** uncertainty. The result's `covariance_scope`
states that the errors are conditional on the response; it does not certify MC
as exact.

### Linked fixed-size generated sample

`MCStatistics.linked()` requires a common physical-event ID namespace in both MC
samples, matching truth coordinates, and at least two generated physical events.
Rejected events have u_e=0. For a fixed number N_g of independent generated
physical events, define

\[
k_e=-\frac{1}{Z}B(u_e-Rz_e)\widehat H.
\]

Because \(\sum_e(u_e-Rz_e)=0\), these influences are already centered. The
fixed-size sample-covariance estimator is

\[
V_H^{\rm MC}=\frac{N_g}{N_g-1}\sum_e k_ek_e^\mathsf T.
\]

This retains the generated/accepted covariance automatically. Its expansion
contains both exposure terms and their cross terms; they must not be dropped
when accepted MC originates from the generated sample. All accepted hypotheses
are aggregated within u_e before the outer product.

### Independent Poisson samples

`MCStatistics.independent()` explicitly asserts independent generated and accepted
Poisson realizations with comparable known exposure. Define separate influences

\[
k_e^a=-\frac{1}{Z}B u_e\widehat H,
\qquad
k_e^g=\frac{z_e}{Z}\widehat H.
\]

Then

\[
V_H^{\rm MC}=\sum_e k_e^ak_e^{a\mathsf T}
+\sum_e k_e^gk_e^{g\mathsf T}.
\]

Absent matching IDs do not make two samples independent. This policy is not valid
for a selected subset of the same generated run or independently normalized
fixed-count samples. If the relationship is unavailable, choose conditional
errors rather than inventing the missing cross covariance.

The implementation accumulates moment-vector influences directly. It does not
construct a covariance with one coordinate for every response-matrix entry.
Matrix ratios and inversion are nonlinear; this is first-order propagation,
not an exact finite-sample distribution or a correction for finite-MC estimator bias.

## Diagnostics and limitations

The response must have full rank. Its smallest singular value must exceed 1e-10
times its largest. `ExtractionError.diagnostics` exposes the singular values,
rank, condition number, generated exposure, and physical-event counts. There is
no silent regularization, pseudoinverse truncation, or setting unconstrained
moments to zero. Poor MC coverage or insufficient beam-polarization information
may make extraction impossible even if some individual moments could be measured.

Extraction retains correlated covariance factors through normalization, preserving
very small and very large common changes of raw weight units. Requesting a raw
covariance matrix whose variances lie outside float64 range raises a rescaling
error rather than silently returning zero or infinity; normalized analysis can
still use its stored factor.

The output normalizer must be positive and finite. Signed data weights do not
guarantee this; a failed normalizer raises an actionable error.

Data angles are the chosen measured hypothesis angles, while accepted MC uses
truth angles. The method assumes negligible reconstruction and incorrect-hypothesis
angular migration. Multiple hypotheses are supported statistically, but grouping
does not correct a systematic migration. A future unfolding extension would need
both truth and reconstructed MC coordinates.

The samples must share the frame, cuts, beam exposure, weighting convention, and
relevant kinematic distributions. Reusing a response across different bins or
beam configurations is valid only when those assumptions remain satisfied.
Unmodeled correlations between detector acceptance and other kinematic variables
cannot be inferred from angular columns alone. Supplied calibration, density,
and weights are treated as known; their fitted or systematic uncertainties are
outside the returned statistical covariance.

`MomentData` retains raw values and labels. `analyze` performs the final ratio
normalization with complete correlated covariance. Passing its compatibility
checks remains a necessary moment condition, not a simultaneous amplitude fit.
