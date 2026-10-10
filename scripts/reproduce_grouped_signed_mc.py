"""Exercise grouped signed MC with disjoint namespaces and exact UInt64 IDs."""

import laddu as ld
import numpy as np

import momentous as mo


def main() -> None:
    """Check strict defaults and explicit models without files or scientific output."""
    rng = np.random.default_rng(101)
    size = 2048
    coordinates = {
        "costheta": rng.uniform(-1, 1, size),
        "phi": rng.uniform(-np.pi, np.pi, size),
        "Phi": rng.uniform(-np.pi, np.pi, size),
        "P": np.full(size, 0.4),
    }
    ids = np.arange(size, dtype=np.uint64) + np.uint64(2**63)
    execution = ld.Execution(backend="cpu", threads=1)

    def sample(
        run: int,
        *,
        repeat: int = 1,
        signed: bool = False,
        reconstructed: bool = False,
    ) -> mo.EventSample:
        scalars = {
            name: np.repeat(values, repeat) for name, values in coordinates.items()
        }
        if reconstructed:
            scalars["phi"][1::2] += 0.03
        weights = np.ones(size * repeat) / repeat
        if signed:
            weights = np.tile([1.0, -1.0 / 6.0], size)
        dataset = ld.Dataset.from_arrays(
            p4s={},
            scalars=scalars,
            weights=weights,
            columns={
                "run_number": np.full(size * repeat, run, dtype=np.uint32),
                "physical_event_number": np.repeat(ids, repeat),
                "event_number": np.repeat(ids % 100, repeat),
            },
        )
        return mo.EventSample(
            dataset,
            events=("run_number", "physical_event_number"),
            polarization=mo.Polarization(magnitude="P", angle="Phi"),
            execution=execution,
        )

    generated, data = sample(1), sample(3)
    pool = mo.Waveset(
        (ell, m, r) for ell in (0, 2) for m in range(-ell, ell + 1) for r in ("+", "-")
    )
    basis = mo.MomentBasis.from_waves(pool)
    for label, accepted, model in (
        (
            "baseline",
            sample(2),
            mo.ResponseModel.truth(),
        ),
        (
            "reconstructed",
            sample(2, repeat=2, reconstructed=True),
            mo.ResponseModel.reconstructed_diagonal(),
        ),
        (
            "signed",
            sample(2, repeat=2, signed=True),
            mo.ResponseModel.truth(),
        ),
        (
            "grouped signed reconstructed",
            sample(2, repeat=2, signed=True, reconstructed=True),
            mo.ResponseModel.reconstructed_diagonal(),
        ),
        (
            "paired truth/reconstruction",
            sample(2, repeat=2, signed=True, reconstructed=True),
            mo.ResponseModel.truth_to_reconstruction(truth=sample(2, repeat=2)),
        ),
    ):
        acceptance = mo.Acceptance(
            generated,
            accepted,
            basis=basis,
            response_model=model,
            statistics=mo.MCStatistics.independent(),
            execution=execution,
        )
        result = acceptance.extract(data)
        assert accepted.n_events == size
        assert np.linalg.eigvalsh(result.data.covariance.matrix).min() > -1e-8
        print(
            f"{label}: passed; {result.diagnostics}; accepted events={accepted.n_events}; "
            f"negative rows={result.diagnostics.accepted_negative_rows}"
        )
        if label in ("reconstructed", "grouped signed reconstructed"):
            try:
                mo.Acceptance(
                    generated,
                    accepted,
                    basis=basis,
                    statistics=mo.MCStatistics.independent(),
                    execution=execution,
                )
            except ValueError:
                pass
            else:
                raise AssertionError(
                    "Default behavior must continue rejecting undeclared assumptions"
                )


if __name__ == "__main__":
    main()
