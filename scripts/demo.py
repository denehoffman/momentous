"""Extract acceptance-corrected moments and search every subset of a wave pool.

The fixed S/P/D truth amplitudes are defined below. Edit the code to change the
model. Moments run through twice the largest search-wave rank; choose at least
l=2 to represent the full truth intensity in the extraction basis.
"""

import argparse

import laddu as ld
import numpy as np

import momentous as mo

TRUTH = {
    mo.Wave(0, 0, "+"): 1.0 + 0j,
    mo.Wave(1, 0, "+"): 0.6 + 0.2j,
    mo.Wave(2, 1, "-"): 0.3 - 0.4j,
}
EVENTS = 20_000
MC_EVENTS = 50_000
SEED = 42
EXECUTION = ld.Execution("jit", precision="f64")

# On-shell gamma p -> X p, X -> K+ K- at fixed beam energy and X mass.
CHANNEL = ld.Channel(
    "gamma p -> K+ K- p",
    edges=[
        ld.Edge(
            "beam",
            p4="beam",
            particle=ld.particles.PHOTON,
            output=True,
            initial_momentum=ld.InitialMomentum.energy(9.0, direction=[0, 0, 1]),
        ),
        ld.Edge(
            "target",
            p4="target",
            particle=ld.particles.PROTON,
            output=True,
            initial_momentum=ld.InitialMomentum.momentum([0, 0, 0]),
        ),
        ld.Edge("X", p4="X", output=True, mass_proposal=ld.MassProposal(1.5)),
        ld.Edge("recoil", p4="recoil", particle=ld.particles.PROTON, output=True),
        ld.Edge("k_plus", p4="k_plus", particle=ld.particles.K_PLUS, output=True),
        ld.Edge("k_minus", p4="k_minus", particle=ld.particles.K_MINUS, output=True),
    ],
    vertices=[
        ld.Vertex(
            "production",
            incoming=["beam", "target"],
            outgoing=["X", "recoil"],
            generation=ld.VertexProposal.t_exchange(
                incoming="beam",
                outgoing="X",
                slope=4.0,
                uniform_fraction=0.2,
            ),
        ),
        ld.Vertex(
            "decay",
            incoming=["X"],
            outgoing=["k_plus", "k_minus"],
            generation=ld.VertexProposal.isotropic(),
        ),
    ],
)
GENERATOR = ld.Generator(
    CHANNEL,
    scalars={
        "pol_magnitude": ld.ScalarSource.uniform(0.2, 0.3),
        "pol_angle": ld.ScalarSource.uniform(0, np.pi),
    },
)

# Helicity angles in the X rest frame: z opposite recoil, y normal to production.
DECAY = CHANNEL.vertex("decay")
BEAM, RECOIL = DECAY.vec3("beam"), DECAY.vec3("recoil")
NORMAL = BEAM.cross(-RECOIL)
COSTHETA = DECAY.costheta("k_plus", z_axis=-RECOIL, y_hint=NORMAL)
PHI = DECAY.phi("k_plus", z_axis=-RECOIL, y_hint=NORMAL)

# Polarization orientation relative to the production plane.
PRODUCTION = CHANNEL.vertex("production")
BEAM, RECOIL = PRODUCTION.vec3("beam"), PRODUCTION.vec3("recoil")
NORMAL = BEAM.cross(-RECOIL)
ANGLE = ld.scalar("pol_angle")
EPSILON = ld.Vec3(ANGLE.cos(), ANGLE.sin(), 0.0)
P = ld.scalar("pol_magnitude")
POLARIZATION_ANGLE = ld.atan2(
    NORMAL.dot(EPSILON), BEAM.unit().dot(EPSILON.cross(NORMAL))
)
SELECTION = (COSTHETA > -0.9) & (COSTHETA < 0.8)


def event_model(polarized: bool = False) -> ld.Model:
    """Build the fixed S/P/D intensity, with optional linear polarization.

    Parameters
    ----------
    polarized : bool, default=False
        Include linear photon polarization in the fixed truth intensity.

    Returns
    -------
    laddu.Model
        Nonnegative event intensity for unweighted generation.

    Notes
    -----
    Polarized reflectivity sectors add incoherently, with coherent amplitudes
    within each sector and the signs of Mathieu Eq. (D13) applied to each.
    Unpolarized mode uses one coherent S/P/D amplitude. Coefficients need no
    normalization: a common scale cancels in sampling and normalized moments.
    """
    terms = [
        (
            wave,
            coefficient,
            ld.spherical_harmonic(wave.L, wave.M, costheta=COSTHETA, phi=PHI),
        )
        for wave, coefficient in TRUTH.items()
    ]
    if not polarized:
        amplitude = sum((c * y for _, c, y in terms), ld.complex(0, 0))
        return ld.Model(amplitude.norm_sqr())

    intensity = ld.complex(0, 0)
    for reflectivity in (-1, 1):
        rotated = [
            (c, y * ld.cis(-POLARIZATION_ANGLE))
            for wave, c, y in terms
            if wave.reflectivity is not None and wave.reflectivity.value == reflectivity
        ]
        real_amplitude = sum((c * y.real() for c, y in rotated), ld.complex(0, 0))
        imag_amplitude = sum((c * y.imag() for c, y in rotated), ld.complex(0, 0))
        intensity += (1 + reflectivity * P) * real_amplitude.norm_sqr()
        intensity += (1 - reflectivity * P) * imag_amplitude.norm_sqr()
    return ld.Model(intensity)


def extract_moments(
    mc_events: int = MC_EVENTS,
    *,
    events: int = EVENTS,
    max_wave_l: int = 2,
    polarized: bool = False,
) -> mo.ExtractionResult:
    """Generate data and MC, then correct the detector acceptance.

    Parameters
    ----------
    mc_events : int, default=50000
        Number of unweighted phase-space MC events, before the detector selection.
    events : int, default=20000
        Number of unweighted model events, before the detector selection.
    max_wave_l : int, default=2
        Largest search-wave rank; extraction includes moments through twice this.
    polarized : bool, default=False
        Include event-wise linear polarization and both reflectivity sectors.

    Returns
    -------
    ExtractionResult
        Raw moments with grouped data and linked finite-MC covariance.

    Notes
    -----
    Truth and measured angles coincide in this toy detector. Generated event IDs
    survive selection and link the separately constructed MC samples. Phase-space
    MC uses a proven envelope. JIT evaluation uses unrestricted threads.
    """
    data, _ = GENERATOR.unweighted(
        events,
        event_model(polarized),
        seed=SEED,
        execution=EXECUTION,
        max_proposals=50_000_000,
        grow_envelope=True,
        index_column="event_id",
    )
    generated, _ = GENERATOR.unweighted(
        mc_events,
        seed=SEED + 1,
        proven_envelope=True,
        execution=EXECUTION,
        index_column="event_id",
    )
    polarization = (
        mo.Polarization(magnitude=P, angle=POLARIZATION_ANGLE) if polarized else None
    )

    def sample(dataset: ld.Dataset) -> mo.EventSample:
        """Snapshot the common helicity frame and stored event IDs and weights."""
        return mo.EventSample(
            dataset,
            costheta=COSTHETA,
            phi=PHI,
            events="event_id",
            polarization=polarization,
        )

    observed = sample(data.select(SELECTION, execution=EXECUTION))
    generated_sample = sample(generated)
    accepted_sample = sample(generated.select(SELECTION, execution=EXECUTION))
    acceptance = mo.Acceptance(
        generated_sample,
        accepted_sample,
        basis=mo.MomentBasis(2 * max_wave_l, polarized=polarized),
    )
    return acceptance.extract(observed)


def main() -> None:
    """Run moment extraction and a complete waveset search from five options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--polarized", action="store_true", help="use linear polarization"
    )
    parser.add_argument("--events", type=int, default=EVENTS, help="model data events")
    parser.add_argument(
        "--mc-events", type=int, default=MC_EVENTS, help="phase-space MC events"
    )
    parser.add_argument("--max-wave-l", type=int, default=2, help="largest wave rank")
    parser.add_argument(
        "--n-sigma", "-n-sigma", type=float, default=3, help="moment uncertainty radius"
    )
    args = parser.parse_args()
    if args.events < 2 or args.mc_events < 2 or args.max_wave_l < 1:
        parser.error("need events and mc-events >= 2 and max-wave-l >= 1")
    if not np.isfinite(args.n_sigma) or args.n_sigma < 0:
        parser.error("need finite n-sigma >= 0")

    print(
        f"Generating {args.events:,} data events and {args.mc_events:,} MC events...",
        flush=True,
    )
    extraction = extract_moments(
        args.mc_events,
        events=args.events,
        max_wave_l=args.max_wave_l,
        polarized=args.polarized,
    )
    print(extraction.diagnostics)
    print(extraction.covariance_scope)

    pool = mo.Waveset.from_max_l(
        args.max_wave_l, reflectivities=("+", "-") if args.polarized else None
    )
    print(
        f"Searching all subsets of {len(pool)} waves with moments through L={2 * args.max_wave_l}...",
        flush=True,
    )
    result = mo.analyze(extraction.data, pool, n_sigma=args.n_sigma).search()
    print(
        f"Passing: {result.count:,}/{result.candidate_count:,}; unresolved: {len(result.unresolved)}"
    )
    print("Certified minimal sets:", result.minimal_wavesets())
    print(result.stats)


if __name__ == "__main__":
    main()
