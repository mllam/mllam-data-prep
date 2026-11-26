"""
Benchmark convex hull cropping (`mllam_data_prep.ops.cropping`).

A quasi-uniform global grid (points on a Fibonacci sphere, similar to an
ICON grid) is cropped to a margin around a regional lat/lon grid with
`crop_with_convex_hull`. For each source grid size the script reports the
median wall-clock time over a number of repeats, and records which grid
points were kept, so that the results of two versions of `mllam-data-prep`
can be compared.

The script only uses functions that are part of the public cropping API, so
it can be run against any version of `mllam-data-prep` that supports
cropping. To compare a branch with `main`, run it from the root of the
repository once with each version installed, saving the results of the first
run and comparing against them in the second:

    uv run --no-project --with "mllam-data-prep @ git+https://github.com/mllam/mllam-data-prep@main" \
        python benchmarks/cropping.py --n-source 100000 1000000 --label main --save results_main.json

    uv run --no-project --with "mllam-data-prep @ git+https://github.com/<user>/mllam-data-prep@<branch>" \
        python benchmarks/cropping.py --n-source 100000 1000000 --label <branch> --compare-to results_main.json

The second command exits with a non-zero status if the kept grid points
differ from the saved results. With `--max-slowdown-pct` it also exits with
a non-zero status if the median time is more than that percentage slower than
in the saved results.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

import mllam_data_prep
from mllam_data_prep.ops.cropping import crop_with_convex_hull


def fibonacci_sphere_lonlat(n_points: int):
    """
    Create a quasi-uniform grid of `n_points` points on the sphere.

    Returns
    -------
    lon, lat : np.ndarray
        Longitudes in [-180, 180) and latitudes in [-90, 90], in degrees.
    """
    i = np.arange(n_points) + 0.5
    lat = np.rad2deg(np.arcsin(1.0 - 2.0 * i / n_points))
    lon = (np.rad2deg(np.pi * (1.0 + 5**0.5) * i) + 180.0) % 360.0 - 180.0
    return lon, lat


def regional_lonlat(n_per_side: int, lon_range, lat_range):
    """
    Create a regular lat/lon grid with `n_per_side` points in each direction.
    """
    lon, lat = np.meshgrid(
        np.linspace(*lon_range, n_per_side), np.linspace(*lat_range, n_per_side)
    )
    return lon.ravel(), lat.ravel()


def as_dataset(lon: np.ndarray, lat: np.ndarray) -> xr.Dataset:
    """
    Create a dataset with a `grid_index` dimension and lat/lon coordinates, in
    the form produced by `mllam-data-prep` before cropping.
    """
    return xr.Dataset(
        coords=dict(
            grid_index=np.arange(lon.size),
            lon=("grid_index", lon),
            lat=("grid_index", lat),
        )
    )


def run_benchmark(
    n_source: int,
    n_lam: int,
    lon_range,
    lat_range,
    margin: float,
    include_interior_points: bool,
    repeats: int,
):
    """
    Time `crop_with_convex_hull` for a source grid of `n_source` points.

    Returns
    -------
    dict
        The timings (in seconds) of each repeat, their median, and the sorted
        grid indices of the kept points.
    """
    ds = as_dataset(*fibonacci_sphere_lonlat(n_source))
    ds_lam = as_dataset(*regional_lonlat(n_lam, lon_range, lat_range))

    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ds_cropped = crop_with_convex_hull(
            ds=ds,
            ds_reference=ds_lam,
            margin_thickness=margin,
            include_interior_points=include_interior_points,
        )
        timings.append(time.perf_counter() - t0)

    kept = np.sort(ds_cropped.grid_index.values.astype(np.int64))
    return dict(
        timings=timings,
        median=float(np.median(timings)),
        n_kept=int(kept.size),
        kept_grid_index=kept.tolist(),
    )


def compare_results(
    results: dict, reference: dict, max_slowdown_pct: Optional[float] = None
) -> bool:
    """
    Print the timings next to those of `reference` and check that the same
    grid points were kept and, if `max_slowdown_pct` is given, that the median
    time is at most `max_slowdown_pct` percent slower than in `reference`.
    Returns True if all checks pass.
    """
    all_ok = True
    print(
        f"\n{'n_source':>10} {reference['label'][:12]:>12} "
        f"{results['label'][:12]:>12} {'change':>8} {'speed-up':>9}  result"
    )
    for n_source, result in results["runs"].items():
        ref = reference["runs"].get(n_source)
        if ref is None:
            print(
                f"{n_source:>10} {'-':>12} {result['median']:>11.2f}s {'-':>8} "
                f"{'-':>9}  no reference"
            )
            continue

        problems = []
        if result["kept_grid_index"] != ref["kept_grid_index"]:
            problems.append(
                f"DIFFERENT points kept ({result['n_kept']} vs {ref['n_kept']})"
            )
        change_pct = 100.0 * (result["median"] / ref["median"] - 1.0)
        if max_slowdown_pct is not None and change_pct > max_slowdown_pct:
            problems.append(
                f"SLOWER by {change_pct:.1f}% (limit {max_slowdown_pct:g}%)"
            )
        all_ok &= not problems

        status = ", ".join(problems) if problems else "ok, same points kept"
        print(
            f"{n_source:>10} {ref['median']:>11.2f}s {result['median']:>11.2f}s "
            f"{change_pct:>+7.1f}% {ref['median'] / result['median']:>8.1f}x  {status}"
        )
    return all_ok


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-source",
        type=int,
        nargs="+",
        default=[100_000],
        help="number(s) of points in the global source grid",
    )
    parser.add_argument(
        "--n-lam",
        type=int,
        default=250,
        help="number of points per side of the regional (reference) grid",
    )
    parser.add_argument(
        "--lon-range",
        type=float,
        nargs=2,
        default=[0.0, 25.0],
        help="longitude range of the regional grid (degrees)",
    )
    parser.add_argument(
        "--lat-range",
        type=float,
        nargs=2,
        default=[50.0, 65.0],
        help="latitude range of the regional grid (degrees)",
    )
    parser.add_argument(
        "--margin", type=float, default=2.0, help="margin width in degrees"
    )
    parser.add_argument(
        "--include-interior-points",
        action="store_true",
        help="keep the points inside the convex hull as well as the margin",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="number of timed repeats per size"
    )
    parser.add_argument(
        "--label",
        help="name for this run, e.g. the branch name (defaults to the package version)",
    )
    parser.add_argument("--save", type=Path, help="save the results to this JSON file")
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="JSON file with results of a previous run to compare against",
    )
    parser.add_argument(
        "--max-slowdown-pct",
        type=float,
        help=(
            "with --compare-to, fail if the median time is more than this "
            "percentage slower than in the reference run"
        ),
    )
    args = parser.parse_args(argv)
    if args.max_slowdown_pct is not None and args.compare_to is None:
        parser.error("--max-slowdown-pct requires --compare-to")

    version = getattr(mllam_data_prep, "__version__", "unknown")
    results = dict(
        label=args.label or version,
        mllam_data_prep_version=version,
        python_version=platform.python_version(),
        machine=platform.machine(),
        settings=dict(
            n_lam=args.n_lam,
            lon_range=args.lon_range,
            lat_range=args.lat_range,
            margin=args.margin,
            include_interior_points=args.include_interior_points,
            repeats=args.repeats,
        ),
        runs={},
    )
    print(f"{results['label']}: mllam-data-prep {version}")
    for n_source in args.n_source:
        result = run_benchmark(
            n_source=n_source,
            n_lam=args.n_lam,
            lon_range=args.lon_range,
            lat_range=args.lat_range,
            margin=args.margin,
            include_interior_points=args.include_interior_points,
            repeats=args.repeats,
        )
        results["runs"][str(n_source)] = result
        print(
            f"n_source={n_source}: median {result['median']:.2f}s over "
            f"{args.repeats} repeat(s), {result['n_kept']} points kept"
        )

    if args.save is not None:
        args.save.write_text(json.dumps(results))
        print(f"results saved to {args.save}")

    if args.compare_to is not None:
        reference = json.loads(args.compare_to.read_text())
        if reference["settings"] != results["settings"]:
            print(
                "warning: settings differ from the reference run: "
                f"{reference['settings']} vs {results['settings']}"
            )
        print(
            f"reference: {reference['label']} "
            f"(mllam-data-prep {reference['mllam_data_prep_version']})"
        )
        if not compare_results(
            results, reference, max_slowdown_pct=args.max_slowdown_pct
        ):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
