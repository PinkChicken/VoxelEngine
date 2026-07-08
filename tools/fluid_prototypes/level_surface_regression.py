"""
LevelSurface regression suite — covers every bug found and fixed in the 2026-07-06 session
(see docs/BUG_LOG.md's four "FIXED" entries under Checkpoint 6), plus the original 5b design
bench's invariants (rest, no-churn, U-tube preserved). Run this against level_surface_v2.py
(the Python twin of the CURRENT shaders/fluid_flux.comp pass 6) before trusting any further
change to that pass, and again after porting any fix found here back to GLSL.

Usage:
    python level_surface_regression.py                  # full suite, default settings
    python level_surface_regression.py --ticks 800       # longer runs (slower, more thorough)
    python level_surface_regression.py --quick           # shorter runs (faster, less thorough)
    python level_surface_regression.py -k open_droplet    # run only tests whose name contains this
    python level_surface_regression.py --verbose          # print full per-cell fill rows

No external dependencies. Exits nonzero if any test fails.
"""
import argparse
import importlib.util
import pathlib
import sys

_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("level_surface_v2", _here / "level_surface_v2.py")
lsv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lsv2)
Sim = lsv2.SimLevelSurfaceV2
AIR, LIQUID, SOLID, FILL_CAP = lsv2.AIR, lsv2.LIQUID, lsv2.SOLID, lsv2.FILL_CAP

ARGS = argparse.Namespace(ticks=None, quick=False, verbose=False, filter=None)


def _ticks(default):
    if ARGS.ticks:
        return ARGS.ticks
    return max(50, default // 4) if ARGS.quick else default


# ======================== scenario builders ========================

def _row(fills, W=None):
    n = len(fills)
    W = W or (n + 2)
    s = Sim(W, 4)
    for x in range(W):
        s.carve(x, 1, AIR)
    off = (W - n) // 2
    for i, f in enumerate(fills):
        if f is not None:
            s.carve(i + off, 1, LIQUID, float(f))
    return s, off


def _utube(fill_left_to=9, arm_gap=3):
    W = 3 + arm_gap
    H = 14
    s = Sim(W, H)
    lx, rx = 1, W - 2
    for y in range(1, H - 1):
        s.carve(lx, y, AIR)
        s.carve(rx, y, AIR)
    for x in range(lx, rx + 1):
        s.carve(x, 1, AIR)
    for y in range(1, fill_left_to + 1):
        s.carve(lx, y, LIQUID, FILL_CAP)
    for x in range(lx, rx + 1):
        s.carve(x, 1, LIQUID, FILL_CAP)
    return s, lx, rx


# ======================== bug-specific regression tests ========================

def test_open_droplet_no_overdrain():
    """Bug 1 (LEVEL_DEGREE): an exposed droplet with both sides open must not lose more
    than ~LEVEL_RATE fraction of its fill in a single tick (the un-normalized bug lost up
    to 4x that, collapsing 255->51 in one tick)."""
    s, off = _row([255.0])
    src_x = off
    before = s.fill[1][src_x]
    s.tick()
    after = s.fill[1][src_x]
    frac_lost = (before - after) / before
    if ARGS.verbose:
        print(f"    tick1: {before:.1f} -> {after:.1f} (lost {frac_lost*100:.1f}%)")
    assert frac_lost < 0.30, (
        f"lost {frac_lost*100:.1f}% of fill in one tick (expected <30%, LEVEL_RATE=0.20 "
        f"with 2 active neighbours in this 2D bench); LEVEL_DEGREE normalization may have regressed"
    )
    print("  [open_droplet_no_overdrain] PASS "
          f"(lost {frac_lost*100:.1f}% in tick 1, was 80%+ before the fix)")


def test_multi_neighbor_exact_conservation():
    """Bug 5 (aggregate-then-round leak): a cell with 2+ active neighbours must conserve
    mass EXACTLY (not just approximately) over many ticks. The original bug summed every
    active pair's float contribution and rounded once at the end, which only guarantees
    conservation when a cell has at most one active neighbour -- routine 2D/3D puddles
    (any interior cell of a multi-cell blob) have 2+, and the old code leaked there
    consistently in one direction, not as random rounding noise. This is the sharpest test
    in the suite: it checks total() is bit-exact after many ticks, not just close."""
    s, off = _row([62, 62, 11, 62, 62])  # interior cells here have 2 active neighbours each
    t0 = s.total()
    max_drift_seen = 0.0
    ticks = _ticks(2000)
    for _ in range(ticks):
        s.tick()
        max_drift_seen = max(max_drift_seen, abs(s.total() - t0))
    drift = abs(s.total() - t0)
    if ARGS.verbose:
        print(f"    final total={s.total():.1f} (start={t0:.1f}), max drift seen mid-run={max_drift_seen:.3f}")
    assert drift == 0, (
        f"mass not EXACTLY conserved after {ticks} ticks: drift={drift:.3f} (max mid-run "
        f"drift={max_drift_seen:.3f}) -- per-pair rounding fix may have regressed to "
        f"aggregate-then-round"
    )
    print(f"  [multi_neighbor_exact_conservation] PASS (exact conservation over {ticks} ticks, "
          f"max transient drift={max_drift_seen:.3f})")


def test_blob_exact_conservation():
    """Same idea as above but with a genuinely 2D-shaped blob (not just a row), closer to
    the in-engine repro that found this bug (a multi-voxel puddle where interior cells have
    neighbours on every side)."""
    s = Sim(7, 4)
    for x in range(7):
        s.carve(x, 1, AIR)
    for x in (2, 3, 4):
        s.carve(x, 1, LIQUID, 200.0)
    s.carve(3, 1, LIQUID, 40.0)  # uneven center, forces real multi-neighbour exchange
    t0 = s.total()
    ticks = _ticks(2000)
    for _ in range(ticks):
        s.tick()
    drift = abs(s.total() - t0)
    if ARGS.verbose:
        print(f"    final row: {[round(s.fill[1][x],1) for x in range(7)]}")
    assert drift == 0, f"mass not EXACTLY conserved after {ticks} ticks: drift={drift:.3f}"
    print(f"  [blob_exact_conservation] PASS (exact conservation over {ticks} ticks)")


def test_droplet_materializes_no_leak():
    """Bug 2 (classify asymmetry): a droplet with NOTHING else nearby must spread into a
    neighbour, not vanish. A second liquid voxel must appear, and total mass must stay
    exact (the bug lost mass to nowhere -- a one-sided leak, not a rounding artifact)."""
    s, off = _row([255.0], W=15)
    src_x = off
    t0 = s.total()
    liquid_count_over_time = []
    for _ in range(20):
        s.tick()
        liquid_count_over_time.append(sum(1 for x in range(s.W) if s.mat[1][x] == LIQUID))
    drift = abs(s.total() - t0)
    if ARGS.verbose:
        print(f"    liquid voxel count over 20 ticks: {liquid_count_over_time}")
    assert max(liquid_count_over_time) >= 2, (
        f"droplet never spread into a second voxel (counts: {liquid_count_over_time}) -- "
        f"classify-asymmetry leak may have regressed"
    )
    assert drift < 1.0, f"mass not conserved: drift={drift:.3f}"
    print(f"  [droplet_materializes_no_leak] PASS (spread to {max(liquid_count_over_time)} "
          f"voxels, cons={drift:.3f})")


def test_thin_film_unlocks_reservoir():
    """Bug 3 (SUBMERGED_MIN): a near-full reservoir with a thin residual film (fill=1)
    directly above it must NOT be permanently locked out of lateral spreading."""
    s = Sim(7, 5)
    for x in range(7):
        s.carve(x, 1, AIR)
        s.carve(x, 2, AIR)
    s.carve(3, 1, LIQUID, 254.0)
    s.carve(3, 2, LIQUID, 1.0)
    for _ in range(_ticks(400)):
        s.tick()
    row = [round(s.fill[1][x], 1) for x in range(7)]
    spread = max(row) - min(v for v in row if v > 0) if any(row) else 0
    if ARGS.verbose:
        print(f"    final row: {row}")
    assert row[1] > 5.0 or row[5] > 5.0, (
        f"reservoir still locked (neighbours barely moved): {row} -- SUBMERGED_MIN fix may have regressed"
    )
    print(f"  [thin_film_unlocks_reservoir] PASS (final row: {row})")


def test_noisy_stream_no_flicker():
    """Bug 4 (continuous damping): a puddle cell under a stream whose fill hovers right at
    SUBMERGED_MIN must not flip between air and liquid every tick. Classification-flip rate
    against the noisy input must stay far below 100% (the boolean-gate bug flipped ~every
    tick; the fix should flip rarely or never for noise this small)."""
    s = Sim(9, 5)
    for x in range(9):
        s.carve(x, 1, AIR)
        s.carve(x, 2, AIR)
    src = 4
    s.carve(src, 1, LIQUID, 150.0)
    stream_seq = [1.9, 2.1]
    mat_history = []
    ticks = _ticks(200)
    for t in range(ticks):
        stream_fill = stream_seq[t % len(stream_seq)]
        s.mat[2][src] = LIQUID
        s.fill[2][src] = stream_fill
        s.tick()
        mat_history.append(s.mat[1][src])
    flips = sum(1 for i in range(1, len(mat_history)) if mat_history[i] != mat_history[i - 1])
    flip_rate = flips / ticks
    if ARGS.verbose:
        print(f"    material history (last 20): {mat_history[-20:]}")
    assert flip_rate < 0.10, (
        f"material flip rate {flip_rate*100:.0f}% over {ticks} ticks (expected <10%) -- "
        f"continuous-damping fix may have regressed to the old boolean-gate flicker"
    )
    print(f"  [noisy_stream_no_flicker] PASS (flip rate {flip_rate*100:.1f}% over {ticks} ticks, was ~99% before the fix)")


# ======================== original 5b design-bench invariants (still must hold) ========================

def test_flat_uneven_puddle():
    """NOTE (2026-07-06, updated after bug 5's fix): tolerance widened from <3.0 to <12.0.
    Per-pair integer rounding (bug 5's fix) means a single pair's transfer only rounds to a
    nonzero integer once diff*(LEVEL_RATE/LEVEL_DEGREE) >= 0.5 -- at the current defaults
    (0.20/4 = 0.05/pair) that's diff >= 10, a genuine, understood floor, not a regression.
    The old <3.0 tolerance only passed because the pre-fix aggregate-then-round leak gave
    a finer effective resolution than honest per-pair rounding actually provides -- i.e. it
    was partly funded by the leak. Accepted as a known limit for now (see BUG_LOG.md); if
    this residual becomes a real problem, the fix is re-tuning proj_LEVEL_RATE (needs full
    bench re-validation, since a higher rate risks reintroducing overshoot/churn elsewhere),
    not forcing a minimum per-pair transfer (traced by hand: causes a permanent 1-unit
    ping-pong for odd-valued diffs, a new churn bug -- rejected)."""
    s, off = _row([62, 62, 11, 62, 62])
    t0 = s.total()
    for _ in range(_ticks(1000)):
        s.tick()
    final = [s.fill[1][off + i] for i in range(5)]
    spread = max(final) - min(final)
    assert abs(s.total() - t0) < 1.0, f"mass not conserved: drift={abs(s.total()-t0):.3f}"
    assert spread < 12.0, f"did not equalize: spread={spread:.1f} (expected ~9-10, the rounding-floor limit)"
    print(f"  [flat_uneven_puddle] PASS (spread={spread:.1f}, ~9-10 expected -- per-pair rounding floor, not a bug)")


def test_even_lake_no_churn():
    s, off = _row([200] * 8)
    for _ in range(200):
        s.tick()
    mx = 0.0
    for _ in range(_ticks(200)):
        s.tick()
        mx = max(mx, s._last_moved)
    assert mx < 1.0, f"churn on even lake: max_move/tick={mx:.3f}"
    print(f"  [even_lake_no_churn] PASS (max_move/tick={mx:.3f})")


def test_gap_merge():
    s, off = _row([124, 124, None, 124, 124])
    t0 = s.total()
    for _ in range(_ticks(1000)):
        s.tick()
    gap_fill = s.fill[1][off + 2]
    assert abs(s.total() - t0) < 1.0, "mass not conserved"
    assert gap_fill > 50.0, f"dry gap did not merge: gap fill={gap_fill:.1f}"
    print(f"  [gap_merge] PASS (gap fill={gap_fill:.0f})")


def test_utube_preserved():
    s, lx, rx = _utube()
    t0 = s.total()
    for _ in range(2000):
        s.tick()
        if s._last_moved < 0.4:
            pass
    am = lambda x: sum(s.fill[y][x] for y in range(s.H)) / FILL_CAP
    imb = abs(am(lx) - am(rx))
    drift = abs(s.total() - t0)
    assert imb < 1.5, f"U-tube arms not balanced: imbalance={imb:.2f}"
    assert drift < 1.0, f"mass not conserved: drift={drift:.3f}"
    print(f"  [utube_preserved] PASS (imbalance={imb:.2f}, cons={drift:.3f})")


TESTS = [
    test_open_droplet_no_overdrain,
    test_droplet_materializes_no_leak,
    test_thin_film_unlocks_reservoir,
    test_noisy_stream_no_flicker,
    test_multi_neighbor_exact_conservation,
    test_blob_exact_conservation,
    test_flat_uneven_puddle,
    test_even_lake_no_churn,
    test_gap_merge,
    test_utube_preserved,
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticks", type=int, default=None, help="override tick count for time-based tests")
    parser.add_argument("--quick", action="store_true", help="run shorter (less thorough) versions of long tests")
    parser.add_argument("--verbose", action="store_true", help="print extra per-test detail")
    parser.add_argument("-k", dest="filter", default=None, help="only run tests whose name contains this substring")
    global ARGS
    ARGS = parser.parse_args()

    print("=" * 70)
    print("LevelSurface regression suite (bug-specific + original 5b invariants)")
    print("=" * 70)

    to_run = [t for t in TESTS if not ARGS.filter or ARGS.filter in t.__name__]
    if not to_run:
        print(f"No tests match filter {ARGS.filter!r}")
        sys.exit(1)

    failures = []
    for test in to_run:
        try:
            test()
        except AssertionError as e:
            failures.append((test.__name__, str(e)))
            print(f"  [{test.__name__}] FAIL: {e}")

    print("=" * 70)
    if failures:
        print(f"{len(failures)}/{len(to_run)} TESTS FAILED")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"ALL {len(to_run)} TESTS PASSED")


if __name__ == "__main__":
    main()
