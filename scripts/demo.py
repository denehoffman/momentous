"""Generate full Laddu events, measure moments, and list compatible wavesets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Literal, cast

import laddu as ld
import numpy as np
from numpy.typing import NDArray

from moments import (
    LinearlyPolarizedMoment,
    Measurement,
    Moment,
    PartialWave,
    PolarizedMeasurement,
    ReflectivityPartialWave,
    validate,
)

Wave = tuple[int, int] | tuple[int, int, str]
DEFAULT_TRUTH = [(0, 0, 1.0, 0.0), (1, 0, 0.6, 0.2), (2, 1, 0.3, -0.4)]
P4_NAMES = ("beam", "target", "X", "recoil", "k_plus", "k_minus")
POL_ANGLES_DEGREES = (0, 45, 90, 135)
POL_MAGNITUDE_RANGE = (0.2, 0.3)


@dataclass(frozen=True)
class TruthWave:
    L: int
    M: int
    amplitude: complex
    reflectivity: str | None = None

    @property
    def key(self) -> Wave:
        if self.reflectivity is None:
            return self.L, self.M
        return self.L, self.M, self.reflectivity


def parse_wave(text: str) -> Wave:
    try:
        parts = text.split(",")
        if len(parts) not in (2, 3):
            raise ValueError
        L, M = (int(part) for part in parts[:2])
        if L < 0 or abs(M) > L:
            raise ValueError
        if len(parts) == 3 and parts[2] not in ("+", "-"):
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected L,M or L,M,+/- with L >= 0 and |M| <= L"
        ) from error
    return (L, M) if len(parts) == 2 else (L, M, parts[2])


def parse_truth(text: str) -> TruthWave:
    try:
        parts = text.split(",")
        if len(parts) not in (4, 5):
            raise ValueError
        key = parse_wave(",".join(parts[:-2]))
        re, im = parts[-2:]
        amplitude = complex(float(re), float(im))
        if not np.isfinite(amplitude) or amplitude == 0:
            raise ValueError
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError(
            "expected L,M,RE,IM or L,M,+/-,RE,IM with a finite, nonzero amplitude"
        ) from error
    return TruthWave(key[0], key[1], amplitude, key[2] if len(key) == 3 else None)


def normalized_amplitudes(truth: list[TruthWave]) -> NDArray[np.complex128]:
    # Scaling first also handles amplitudes with very large or small magnitudes.
    amplitudes = np.array([wave.amplitude for wave in truth], dtype=np.complex128)
    amplitudes /= max(np.max(np.abs(amplitudes.real)), np.max(np.abs(amplitudes.imag)))
    return amplitudes / np.linalg.norm(amplitudes)


def build_channel(beam_energy: float, mass: float) -> ld.Channel:
    """On-shell gamma p -> X p, X -> K+ K- with phase-space proposals."""
    channel = ld.Channel(
        "gamma p -> K+ K- p",
        edges=[
            ld.Edge(
                "beam",
                p4="beam",
                particle=ld.particles.PHOTON,
                output=True,
                initial_momentum=ld.InitialMomentum.energy(
                    beam_energy, direction=[0, 0, 1]
                ),
            ),
            ld.Edge(
                "target",
                p4="target",
                particle=ld.particles.PROTON,
                output=True,
                initial_momentum=ld.InitialMomentum.momentum([0, 0, 0]),
            ),
            ld.Edge("X", p4="X", output=True, mass_proposal=ld.MassProposal(mass)),
            ld.Edge("recoil", p4="recoil", particle=ld.particles.PROTON, output=True),
            ld.Edge("k_plus", p4="k_plus", particle=ld.particles.K_PLUS, output=True),
            ld.Edge(
                "k_minus", p4="k_minus", particle=ld.particles.K_MINUS, output=True
            ),
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
    return channel


def helicity_observables(
    channel: ld.Channel,
) -> tuple[ld.Expr, ld.Expr, ld.Expr, ld.Expr]:
    """K+ angles in X rest frame: z opposite recoil, y normal to production."""
    decay = channel.vertex("decay")
    beam, recoil = decay.vec3("beam"), decay.vec3("recoil")
    normal = beam.cross(-recoil)
    costheta = decay.costheta("k_plus", z_axis=-recoil, y_hint=normal)
    phi = decay.phi("k_plus", z_axis=-recoil, y_hint=normal)
    production = channel.vertex("production")
    beam, recoil = production.vec3("beam"), production.vec3("recoil")
    normal = beam.cross(-recoil)
    angle = ld.scalar("pol_angle")
    epsilon = ld.Vec3(angle.cos(), angle.sin(), 0.0)
    Phi = ld.atan2(normal.dot(epsilon), beam.unit().dot(epsilon.cross(normal)))
    P = ld.scalar("pol_magnitude")
    return costheta, phi, P, Phi


def build_event_model(
    channel: ld.Channel, truth: list[TruthWave], polarized: bool
) -> ld.Model:
    costheta, phi, P, Phi = helicity_observables(channel)
    # Spin-zero kaon daughters: sqrt((2l+1)/(4pi)) D^{l*}_{m0} = Y_l^m.
    basis = [
        np.sqrt((2 * wave.L + 1) / (4 * np.pi))
        * ld.WignerD(wave.L, wave.M, 0).D(alpha=phi, beta=ld.acos(costheta)).conj()
        for wave in truth
    ]
    terms = list(zip(truth, normalized_amplitudes(truth), basis, strict=True))
    if not polarized:
        return ld.Model(
            sum((complex(c) * y for _, c, y in terms), ld.complex(0, 0)).norm_sqr()
        )

    # Mathieu Eq. (D13), one coherent nucleon-helicity component per sector.
    intensity = ld.complex(0, 0)
    for sign, sector in ((1, "+"), (-1, "-")):
        real_sum, imag_sum = ld.complex(0, 0), ld.complex(0, 0)
        for wave, coefficient, angular in terms:
            if wave.reflectivity == sector:
                z = angular * ld.cis(-Phi)
                real_sum += complex(coefficient) * z.real()
                imag_sum += complex(coefficient) * z.imag()
        intensity += (1 + sign * P) * real_sum.norm_sqr()
        intensity += (1 - sign * P) * imag_sum.norm_sqr()
    return ld.Model(intensity)


def generate_events(
    channel: ld.Channel, model: ld.Model, events: int, seed: int
) -> ld.Dataset:
    rng = np.random.default_rng(seed)
    angle_indices = rng.integers(len(POL_ANGLES_DEGREES), size=events)
    p4s = {name: np.empty((events, 4)) for name in P4_NAMES}
    scalars = {name: np.empty(events) for name in ("pol_angle", "pol_magnitude")}
    # Laddu has no discrete scalar source. Random angle assignments select
    # native generator groups; full azimuthal coverage gives equal normalizations.
    for index, angle in enumerate(POL_ANGLES_DEGREES):
        selected = np.flatnonzero(angle_indices == index)
        if not len(selected):
            continue
        generator = ld.Generator(
            channel,
            scalars={
                "pol_magnitude": ld.ScalarSource.uniform(*POL_MAGNITUDE_RANGE),
                "pol_angle": ld.ScalarSource.fixed(float(np.deg2rad(angle))),
            },
        )
        group, _ = generator.unweighted(
            len(selected),
            model,
            seed=int(rng.integers(0, 2**63)),
            execution=ld.Execution("cpu", threads=1, precision="f64"),
            max_proposals=50_000_000,
            grow_envelope=True,
        )
        for name in p4s:
            p4 = ld.Vec4.event(name)
            p4s[name][selected] = np.column_stack(
                [
                    group.evaluate(component, real=True)
                    for component in (p4.e(), p4.px(), p4.py(), p4.pz())
                ]
            )
        for name in scalars:
            scalars[name][selected] = group.evaluate(ld.scalar(name), real=True)
    return ld.Dataset.from_arrays(p4s=p4s, scalars=scalars)


def estimate_moments(
    data: ld.Dataset,
    channel: ld.Channel,
    max_L: int,
    polarized: bool,
) -> tuple[list[Measurement | PolarizedMeasurement], NDArray[np.float64]]:
    costheta, phi, P, Phi = helicity_observables(channel)
    features: list[NDArray[np.float64]] = []
    measurements: list[Measurement | PolarizedMeasurement] = []
    if polarized:
        angle = np.asarray(data.evaluate(Phi, real=True))
        mag = np.asarray(data.evaluate(P, real=True))
        cosine, sine = 2 * np.cos(2 * angle) / mag, -2 * np.sin(2 * angle) / mag
        features.append(cosine)
        measurements.append(
            PolarizedMeasurement(LinearlyPolarizedMoment(0, 0, 1), float(cosine.mean()))
        )
    for L in range(1, max_L + 1):
        for M in range(L + 1):
            f = np.asarray(
                data.evaluate(
                    np.sqrt(4 * np.pi / (2 * L + 1))
                    * ld.spherical_harmonic(L, M, costheta=costheta, phi=phi),
                ),
                dtype=np.complex128,
            )
            if polarized:
                # Mathieu Eq. (13); every event uses its own P.
                columns: list[tuple[Literal[0, 1, 2], NDArray[np.float64]]] = [
                    (0, f.real),
                    (1, cosine * f.real),
                ]
                if M:
                    columns.append((2, sine * f.imag))
                for variant, values in columns:
                    moment = LinearlyPolarizedMoment(L, M, variant)
                    measurements.append(
                        PolarizedMeasurement(moment, float(values.mean()))
                    )
                    features.append(values)
            else:
                measurements.append(Measurement(Moment(L, M), complex(f.mean())))
                features.extend((f.real, f.imag))
    # Full covariance of the sample means, in the validator's observable order.
    covariance = np.cov(np.column_stack(features), rowvar=False) / len(data)
    return measurements, covariance


def as_waves(keys: Sequence[Wave]) -> list[PartialWave | ReflectivityPartialWave]:
    result: list[PartialWave | ReflectivityPartialWave] = []
    for key in keys:
        if len(key) == 2:
            result.append(PartialWave(key[0], key[1]))
        elif key[2] == "+":
            result.append(ReflectivityPartialWave(key[0], key[1], "+"))
        else:
            result.append(ReflectivityPartialWave(key[0], key[1], "-"))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--polarized",
        action="store_true",
        help="use linear polarization and reflectivity waves",
    )
    parser.add_argument("--events", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wave",
        action="append",
        type=parse_truth,
        metavar="L,M,[R,]RE,IM",
        help="truth amplitude; repeat to replace the S/P/D default; R=+/- in polarized mode",
    )
    parser.add_argument(
        "--candidate-wave",
        action="append",
        type=parse_wave,
        metavar="L,M[,R]",
        help="search-pool wave; repeat to replace --max-wave-l",
    )
    parser.add_argument(
        "--max-wave-l",
        type=int,
        default=2,
        help="search all signed m through this l (default: D waves)",
    )
    parser.add_argument(
        "--max-waves",
        type=int,
        help="largest candidate size (default: truth size, capped by pool size)",
    )
    parser.add_argument("--max-moment-l", type=int, default=4)
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=3,
        help="moment projection tolerance in standard errors",
    )
    return parser


def run(args: argparse.Namespace) -> list[tuple[Wave, ...]]:
    truth = args.wave or [
        TruthWave(L, M, complex(re, im), "+" if args.polarized else None)
        for L, M, re, im in DEFAULT_TRUTH
    ]
    if args.candidate_wave:
        pool: list[Wave] = sorted(args.candidate_wave)
    elif args.polarized:
        pool = [
            (L, M, r)
            for L in range(args.max_wave_l + 1)
            for M in range(-L, L + 1)
            for r in ("+", "-")
        ]
    else:
        pool = [(L, M) for L in range(args.max_wave_l + 1) for M in range(-L, L + 1)]
    truth_keys = tuple(sorted(wave.key for wave in truth))
    maximum = min(len(truth), len(pool)) if args.max_waves is None else args.max_waves
    if args.events < 2 or args.max_moment_l < 1 or args.max_wave_l < 0:
        raise ValueError("need events >= 2, max-moment-l >= 1 and max-wave-l >= 0")
    if not 1 <= maximum <= len(pool):
        raise ValueError("max-waves must be between 1 and the pool size")
    if any((len(key) == 3) != args.polarized for key in (*truth_keys, *pool)):
        raise ValueError("include reflectivity +/- on every wave only with --polarized")
    if len(set(truth_keys)) != len(truth_keys) or len(set(pool)) != len(pool):
        raise ValueError("truth and search pool must contain unique waves")
    if args.seed < 0 or not np.isfinite(args.n_sigma) or args.n_sigma < 0:
        raise ValueError("need seed >= 0 and finite n-sigma >= 0")

    print(f"Generating {args.events:,} gamma p -> K+ K- p events...", flush=True)
    channel = build_channel(9.0, 1.5)
    data = generate_events(
        channel,
        build_event_model(channel, truth, args.polarized),
        args.events,
        args.seed,
    )
    measurements, covariance = estimate_moments(
        data, channel, args.max_moment_l, args.polarized
    )
    total = sum(comb(len(pool), size) for size in range(1, maximum + 1))
    print(
        f"Screening {total:,} wavesets with moments through L={args.max_moment_l}...",
        flush=True,
    )

    def screen(keys: Sequence[Wave], pairwise: bool) -> bool:
        waves = as_waves(keys)
        if args.polarized:
            return validate(
                cast(list[ReflectivityPartialWave], waves),
                cast(list[PolarizedMeasurement], measurements),
                covariance=covariance,
                n_sigma=args.n_sigma,
                pairwise=pairwise,
                n_angles=180,
            ).valid
        return validate(
            cast(list[PartialWave], waves),
            cast(list[Measurement], measurements),
            covariance=covariance,
            n_sigma=args.n_sigma,
            pairwise=pairwise,
            n_angles=180,
        ).valid

    passing = []
    for size in range(1, maximum + 1):
        for candidate in combinations(pool, size):
            # Skip pairwise work when the individual bounds already fail.
            if screen(candidate, False) and screen(candidate, True):
                passing.append(candidate)
    print("Truth:", as_waves(truth_keys))
    print(f"Passing wavesets ({len(passing)}/{total}):")
    for candidate in passing:
        print(" ", as_waves(candidate), "[truth]" if candidate == truth_keys else "")
    print("Truth is in passing candidates:", truth_keys in passing)
    if not set(truth_keys).issubset(pool) or len(truth_keys) > maximum:
        print("Truth is excluded by the search pool or --max-waves.")
    return passing


def main() -> None:
    parser = build_parser()
    try:
        run(parser.parse_args())
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
