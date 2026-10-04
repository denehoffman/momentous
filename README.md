# Moments

Check whether a partial-wave set passes covariance-aware compatibility tests
against measured angular moments. Supply measurements already normalized by
`H(0, 0)` for unpolarized moments or `H^0(0, 0)` for polarized moments, together
with the covariance **of those normalized measurements**.

```python
import numpy as np
from moments import Measurement, Moment, validate, waveset

waves = waveset([(0, 0), (1, 0)])
measurements = [
    Measurement(Moment(1, 0), 0.55),
    Measurement(Moment(2, 0), 0.02),
]

# Order: Re H(1,0), Im H(1,0), Re H(2,0), Im H(2,0).
covariance = np.diag([0.01**2, 0, 0.01**2, 0])
covariance[0, 2] = covariance[2, 0] = 0.5 * 0.01**2

result = validate(waves, measurements, covariance=covariance, n_sigma=3)
print(result.valid)
print([item.moment for item in result.invalid_measurements])
for failure in result.failures:
    print(failure.coefficients, failure.value, failure.bounds,
          failure.uncertainty, failure.sigma_distance)
```

Unpolarized values may be complex. For `n` measurements, covariance has shape
`(2*n, 2*n)` in interleaved real/imaginary order, preserving the input order.
Include zero rows and columns for components known exactly. All correlations,
including between real and imaginary components, are supported. Covariance must
be real, finite, symmetric, and positive semidefinite; singular matrices are valid.

Polarized moments use the same validation function:

```python
from moments import (
    LinearlyPolarizedMoment, PolarizedMeasurement, polarized_waveset,
)

waves = polarized_waveset([(0, 0, "+"), (1, 0, "+")])
measurements = [
    PolarizedMeasurement(LinearlyPolarizedMoment(1, 0, 0), 0.4),
    PolarizedMeasurement(LinearlyPolarizedMoment(1, 0, 1), 0.4),
    PolarizedMeasurement(LinearlyPolarizedMoment(1, 1, 2), 0.0),
]
result = validate(waves, measurements, covariance=np.diag([0.05**2]*3), n_sigma=3)
```

Polarized measurements are real observables: variants 0 and 1 represent `H^0`
and `H^1`, and variant 2 represents `Im H^2`. Their covariance has shape `(n, n)`
in measurement order. Checks include combinations across variants and use the
`H^0(0, 0)` normalization matrix. Both reflectivity sectors may be supplied.
Signs follow Mathieu et al., [Phys. Rev. D 100, 054017 (2019)](https://arxiv.org/pdf/1906.04841),
Eq. (13): `H^1` uses the positive cosine projection, and `Im H^2` uses the
negative sine projection. Thus a pure `S_0^+` wave has normalized `H^1(0,0)=+1`.
This corrects the previous `H^1` operator sign. Measurements produced for the
previous sign must have `H^1` values and corresponding covariance rows/columns
multiplied by `-1` to use the current convention.

Existing calls such as `Measurement(moment, value, uncertainty)` and
`PolarizedMeasurement(moment, value, uncertainty)` continue to work. Without
`covariance=`, these uncertainties are independent standard deviations; a
complex uncertainty specifies the real and imaginary deviations separately.
Omitted uncertainties mean exact measurements. A supplied covariance supersedes
these individual uncertainties. `n_sigma` defaults to 1; use 0 for exact bounds.

Validation collects **all failures**, continuing through individual and pairwise
checks. `result.failures` contains every rejected scalar component and sampled
pairwise direction, with its coefficients, measured value, allowed interval,
propagated standard deviation, distance in standard deviations, and participating
measurements. `result.invalid_measurements` lists the involved measurements once
in input order. A failed pairwise combination does not establish that either
measurement is incompatible by itself. The original `result.measurement` and
`result.bounds` attributes remain available for the first failure.

For a real coefficient vector `w`, validation compares `w @ values` with the
extreme eigenvalues of the combined moment operator and allows an excursion of
`n_sigma * sqrt(w @ covariance @ w)`. Polarized operators use generalized
eigenvalues with the normalization matrix. The default `pairwise=True` checks
all pairs of scalar components, including Re/Im within a single moment.
`n_angles=360` samples directions uniformly over a half-circle, checking both
ends of each interval. Increase it for finer sampling, or use `pairwise=False`
for individual bounds only. Axis-aligned checks are recorded individually.

These are necessary compatibility conditions. Passing finite pairwise checks
does not establish a simultaneous amplitude solution for all moments. `n_sigma`
is a tolerance for each projected measurement, not a global confidence level,
chi-square threshold, or p-value for the waveset. No raw-moment normalization or
covariance propagation through a measured normalization denominator is performed.

Run tests with `uv run python -m unittest discover -s tests -v`.
