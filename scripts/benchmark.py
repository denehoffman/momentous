"""Benchmark public momentous workflows with process-isolated time limits."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import platform
from dataclasses import asdict
from math import comb, isfinite
from multiprocessing.connection import Connection
from pathlib import Path
from time import perf_counter

from demo import EVENTS, MC_EVENTS, SEED, TRUTH, extract_moments

from momentous import MomentData, Status, Waveset, analyze


def benchmark_case(
    output: Connection,
    data: MomentData,
    pool: Waveset,
    maximum: int,
    n_sigma: float,
    full_only: bool,
    baseline_limit: int,
    workers: int | None = None,
) -> None:
    """Measure public analyses and send compact results to the parent process."""
    try:
        start = perf_counter()
        prepared = analyze(
            data,
            pool,
            n_sigma=n_sigma,
        )
        row: dict[str, object] = {
            "waves": len(pool),
            "candidates": sum(comb(len(pool), size) for size in range(1, maximum + 1)),
            "prepare_and_full_check_seconds": perf_counter() - start,
            "full_status": prepared.check().status.value,
            "search_complete": False,
            "passing": None,
            "workers": workers if workers is not None else "auto",
        }
        output.send(row.copy())
        if full_only:
            row["search_skipped"] = True
            row.update(asdict(prepared.stats))
        else:
            start = perf_counter()
            result = prepared.search(max_size=maximum, workers=workers)
            row.update(
                search_seconds=perf_counter() - start,
                search_complete=result.complete,
                passing=result.count,
                minimal=len(result.minimal_wavesets()),
                unresolved=len(result.unresolved),
                **asdict(result.stats),
            )
            output.send(row.copy())
            if len(pool) <= baseline_limit:
                start = perf_counter()
                passing = {
                    candidate
                    for candidate in pool.powerset(1, maximum)
                    if analyze(
                        data,
                        candidate,
                        n_sigma=n_sigma,
                    )
                    .check()
                    .status
                    is Status.COMPATIBLE
                }
                row["exhaustive_seconds"] = perf_counter() - start
                row["branching_matches_exhaustive"] = passing == set(result.wavesets())
                if not row["branching_matches_exhaustive"]:
                    raise RuntimeError(
                        "Branching differs from independent candidate checks"
                    )
        row["case_finished"] = True
        output.send(row)
    except Exception as error:
        output.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        output.close()


def run_case(
    data: MomentData,
    pool: Waveset,
    maximum: int,
    n_sigma: float,
    full_only: bool,
    baseline_limit: int,
    time_limit: float,
    workers: int | None = None,
) -> dict[str, object]:
    """Run one case with a hard wall-clock limit and retain completed stages."""
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=benchmark_case,
        args=(
            sender,
            data,
            pool,
            maximum,
            n_sigma,
            full_only,
            baseline_limit,
            workers,
        ),
    )
    start = perf_counter()
    process.start()
    sender.close()
    process.join(time_limit)
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join()
    row: dict[str, object] = {
        "waves": len(pool),
        "search_complete": False,
        "passing": None,
    }
    while receiver.poll():
        try:
            row.update(receiver.recv())
        except EOFError:
            break
    receiver.close()
    row.update(case_seconds=perf_counter() - start, timed_out=timed_out)
    if process.exitcode and not timed_out and "error" not in row:
        row["error"] = f"Worker exited with code {process.exitcode}"
    process.close()
    return row


def main() -> None:
    """Generate one demo dataset and write bounded benchmark cases as JSON/CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polarized", action="store_true")
    parser.add_argument("--events", type=int, default=EVENTS)
    parser.add_argument("--mc-events", type=int, default=MC_EVENTS)
    parser.add_argument("--n-sigma", type=float, default=3)
    parser.add_argument(
        "--workers",
        type=int,
        action="append",
        help="search thread counts to compare; repeat (default: automatic)",
    )
    parser.add_argument(
        "--pool-l", type=int, action="append", help="pool ranks to benchmark; repeat"
    )
    parser.add_argument(
        "--baseline-limit",
        type=int,
        default=9,
        help="largest pool for independent exhaustive checks",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/latest.json"))
    parser.add_argument(
        "--time-limit",
        type=float,
        default=120,
        help="wall-clock limit per entire benchmark case",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="skip search and check the complete pool only",
    )
    args = parser.parse_args()
    levels = args.pool_l or [2, 3]
    worker_counts = args.workers or [None]
    if any(count is not None and count < 1 for count in worker_counts):
        parser.error("workers must be positive")
    if any(level < 0 for level in levels) or args.baseline_limit < 0:
        parser.error("pool-l and baseline-limit must be nonnegative")
    if not isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("time-limit must be finite and positive")
    if (
        args.events < 2
        or args.mc_events < 2
        or not isfinite(args.n_sigma)
        or args.n_sigma < 0
    ):
        parser.error("need events and mc-events >= 2 and finite n-sigma >= 0")
    print(
        f"Generating {args.events:,} data events and {args.mc_events:,} MC events...",
        flush=True,
    )
    extraction = extract_moments(
        args.mc_events,
        events=args.events,
        max_wave_l=max(levels),
        polarized=args.polarized,
    )
    print(extraction.diagnostics)
    print(extraction.covariance_scope)
    rows = []
    for level, workers in (
        (level, workers) for level in levels for workers in worker_counts
    ):
        pool = Waveset.from_max_l(
            level, reflectivities=("+", "-") if args.polarized else None
        )
        maximum = len(pool)
        print(f"Benchmarking {len(pool)} waves...", flush=True)
        row = run_case(
            extraction.data,
            pool,
            maximum,
            args.n_sigma,
            args.full_only,
            args.baseline_limit,
            args.time_limit,
            workers,
        )
        row["pool_l"] = level
        if "error" in row:
            raise RuntimeError(str(row["error"]))
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                python=platform.python_version(),
                events=args.events,
                mc_events=args.mc_events,
                seed=SEED,
                polarized=args.polarized,
                max_moment_l=2 * max(levels),
                n_sigma=args.n_sigma,
                truth=[str(wave) for wave in TRUTH],
                results=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
