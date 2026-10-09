# Continuous compatibility, normalization, and branching

The public workflow is `momentous.analyze(MomentData(...), Waveset(...))`. Unpolarized
raw values are flattened in mapping order as interleaved real/imaginary components;
polarized variants are real components. Every input includes H00 and at least one
other moment. Covariance has explicit scalar labels and is aligned to this order;
`Covariance.from_uncertainties` covers every moment with independent deviations,
or `Covariance.exact()` declares exact data. All constructors use the same required
`covariance` argument to `MomentData`, which owns a validated raw snapshot.
Analysis checks the full pool; `analysis.search(min_size=1, max_size=...)` returns
a separate `SearchResult`. Checks and candidate views remain independent of its
cardinality domain. The search shares the prepared analysis and certificates.

Pair evaluation batches attainable-hull witnesses with NumPy. Exactly equivalent
observables can share pair checks only when their operators, measured values,
covariance rows, and uncertainty-factor rows agree up to a sign. A pair of repeated
observables remains necessary: its Euclidean tolerance is stricter than either
individual component's tolerance. Detailed diagnostics still check every labeled
pair. Exact duplicate reference points are removed without changing their hull.

Parallel search schedules independent candidate evaluations. A single coordinator
owns monotonic certificates, branch pruning, statistics, and check callbacks;
each worker owns its mutable numerical solver templates. Unresolved evaluations
cannot establish certificates. Outstanding work is bounded by the worker count,
and cancellation stops scheduling further evaluations.

Both modes use raw Wigner-D moments and divide by H00 (H^0(0,0) when polarized).
For a real scalar component x_i and normalizer d, y_i=x_i/d, so the Jacobian is
J_ij=delta_ij/d-x_i*delta_jd/d^2. The denominator's
normalized row is exactly constant and its Jacobian row is zero. Covariance
propagates as J V J.T; the implementation propagates its factor as J F to avoid
variance cancellation. The factor transformation uses relative raw quantities to
avoid overflow/underflow from squaring the denominator. A normalizer with relative
uncertainty at least 1/3 emits `NormalizationWarning`. This is a first-order
approximation including denominator
correlations, not an exact ratio distribution.

Let `h` be normalized real scalar moments and `V` their covariance. For a candidate
`S`, restrict the raw operators `C_j` and normalization `G` to `S`, then construct
`A_j = G^(-1/2) C_j G^(-1/2)`. Normalized amplitudes predict `z† A_j z`, with
`z†z = 1`. All candidates share the same measured data and uncertainty multiplier.

```text
Compatible(S):
    reject the empty set
    construct the normalized candidate operators
    reject if any scalar's uncertainty interval misses its spectral bounds
    for every nonredundant pair:
        accept that pair if a known attainable inner hull meets its ellipse
        otherwise intersect angular wedges from known attainable points
        accept that pair if the wedge is empty
        reject if a surviving midpoint gives a positive separating gap
        otherwise solve continuous convex feasibility for that pair
    return incompatible if any pair is excluded
    return unresolved if a remaining pair has no verified decision
    otherwise return compatible
```

Known points come from basis amplitudes, individual operator extremizers (reused
across pairs), and a small collection of supporting eigenvectors. Their hull lies
inside the true two-operator numerical range. A missed inner hull never rejects a
candidate. Angular coverage can establish that the measured center is in the hull;
actual chords supply further feasible points near its boundary. These shortcuts
change computation, not the compatibility predicate.

For a positive-definite pair covariance, whiten with `M = V^(-1/2)`. For each
attainable point `q`, set `d = M(p-q)`. If `||d|| <= tau`, the pair passes.
Otherwise retain normal angles in the circular arc centered at `arg(d)` with
half-width `acos(tau/||d||)`. Wrapped arcs are split at zero and intersected as
ordinary intervals. Midpoint normals are mapped back with `M.T` and tested by

```text
gap(n) = n.T p - lambda_max(n_1 A + n_2 B) - tau ||V^(1/2) n||.
```

A positive gap excludes the ellipse. Failure to find a positive sampled gap never
passes a pair: the fallback solves the following convex distance problem,
equivalent to the zero-distance feasibility problem:

```text
minimize ||(tr(A X), tr(B X)) - p - V^(1/2) t||
subject to X >= 0, tr(X) = 1, ||t|| <= tau.
```

This formulation also handles singular covariance. Repair a returned matrix to
PSD unit trace and recompute its prediction and distance to the ellipse. If that
does not establish a pass, verify the separating direction supplied by the closest
points or solve the dual separation problem. Neither an optimizer's status alone
nor a local angular maximum establishes a decision. The implementation permits a
fixed absolute numerical slack and reports unresolved cases explicitly.

```text
Search(R, U):
    return if the branch has no sets within the permitted size limits
    if R is nonempty and Compatible(R): emit [R,U]; return
    if U is incompatible: return
    if R = U: record its unresolved outcome; return
    choose w in U\R
    Search(R, U\{w})
    Search(R union {w}, U)
```

The implementation uses a stack, exclusion first. Cache identical calls and use
minimal known passing sets and maximal known failing sets to infer decisions
across branches. This is justified because `S subset T` implies that the attainable
regions for `S` are contained in those for `T`. Unresolved decisions never prune.
Output blocks are disjoint and retain size limits; count them combinatorially or
expand them lazily. Exhaustive output and worst-case searching remain exponential.

Moment conventions: [Mathieu et al. (2019)](https://arxiv.org/pdf/1906.04841).
Numerical-range convexity and density-matrix representation:
[Henrion (2010)](https://arxiv.org/pdf/1003.4837).
Separation and conic optimization:
[Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/).

With `max_combination_size=1`, stop after scalar checks. Size 2 uses all
nonredundant scalar pairs continuously; larger sizes are explicitly unsupported.
There is no angular sampling parameter in the compatibility predicate.

Numerical regions sample support eigenvalues over a full circle. Supporting
eigenvectors supply attainable points; tangent extrema inside degenerate
supporting eigenspaces supply both flat-face endpoints. Their convex hull is an
inner polygonal boundary approximation. Region resolution never determines a
compatibility decision.

Minimal output starts from each passing block's smallest admissible members,
removes candidates with known passing proper subsets, and verifies every
admissible immediate subset of each survivor as incompatible. Monotonicity then
excludes every proper subset. Unresolved smaller candidates prevent certification.
Minimality is relative to the searched cardinality limits.

`Moment.real` and `Moment.imag` create scalar `Observable` objects. Polarized
Moment objects already identify a scalar; variant 2 identifies Im H^2.
Canonical tuple labels remain accepted as shorthand and identify matrix order.
Diagnostics distinguish complete `ProjectionFailure` witnesses from
`UnresolvedPair` records with no claimed projection or verdict. Detailed
candidate checks evaluate the actual candidate even when its status was inferred.
