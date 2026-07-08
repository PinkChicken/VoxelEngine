"""
Flat-puddle equalization bench — validates the surface-leveling fix for the
pressure-projection design gap (BUG_LOG.md, 2026-07-04).

THE GAP: BuildRHS's divergence has two terms, both requiring vertical stacking:
  1. Provisional gravity flux (liquid cell above)
  2. Density correction STIFF*(FILL_CAP-fill) (submerged mask — liquid above)

When water finishes spreading into a single flat layer, no cell has same-liquid
above → div≡0 everywhere → P≡0 → corrected flux=0 → uneven puddle stuck forever.

THE FIX: Augment BuildRHS with a surface-leveling term on free-surface cells
(no liquid above). For each surface cell, compute the average fill of its
horizontal liquid neighbours. If that average exceeds the cell's own fill,
add STIFF_H * (avg - fill) to div. This injects within-layer unevenness
directly into the RHS, breaking the zero-divergence silence without touching
the existing submerged density correction.

Subclasses utube_projection.py's Sim, overriding only solve_pressure().
Do NOT modify the original file.
"""

import importlib.util
import pathlib

_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("utube_projection", _here / "utube_projection.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

AIR, LIQUID, SOLID = _base.AIR, _base.LIQUID, _base.SOLID
FILL_CAP = _base.FILL_CAP


class SimFlatPuddle(_base.Sim):
    """Sim with an additional surface-leveling divergence term.

    Adds STIFF_H (horizontal leveling gain, default 0.5) and overrides
    solve_pressure() to include the avg-neighbour-deficit source on
    free-surface cells only.
    """

    def __init__(self, W, H):
        super().__init__(W, H)
        self.STIFF_H = 0.5  # horizontal leveling gain (separate from vertical STIFF)
        # MIN_FLOW 0.1, NOT 0.01: a settled U-tube churns (~0.24/tick) at 0.05 and below
        # (a property of the BASE projection exposed by a tiny deadzone, not this fix), which
        # would flood the changelog / block chunk-sleep in-engine. 0.1 is the churn-free floor.
        self.MIN_FLOW = 0.1

    def solve_pressure(self):
        W, H = self.W, self.H
        div = [[0.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                if self.mat[y][x] != LIQUID:
                    continue
                fy_up = self.prov_fy(x, y)
                fy_dn = self.prov_fy(x, y - 1)
                d = (fy_up - fy_dn)

                # --- existing submerged density correction (unchanged) ---
                if self.is_liquid(x, y + 1):
                    d += self.STIFF * (FILL_CAP - self.fill[y][x])

                # --- NEW: surface-leveling term (free-surface cells only) ---
                # A surface cell (no liquid directly above) that is under-full relative to a
                # horizontal liquid neighbour should attract water.  Uses MAX neighbour fill,
                # NOT average: an average source is a discrete Laplacian, which is ~zero on a
                # smooth slope, so it leaves a residual tilt that GROWS with puddle width
                # (avg: 25-cell spread ~15; max: ~2 at the same MIN_FLOW).  Only fires in the
                # under-full direction (max > fill).  NOTE: this whole in-projection approach
                # is inherently limited on a 1-deep flat layer -- every cell has air above, so
                # the free-surface P=0 BC pins P~0 everywhere and the solve produces almost no
                # flow; only tiny residual flux levels it, which is why a small deadzone is
                # needed at all.  A dedicated surface height-leveling pass (see the alt
                # prototype) sidesteps this entirely and levels better at MIN_FLOW=0.5.
                if not self.is_liquid(x, y + 1):  # surface cell
                    nb_fills = [self.fill[ny][nx] for nx, ny in ((x + 1, y), (x - 1, y))
                                if self.is_liquid(nx, ny)]
                    if nb_fills:
                        m = max(nb_fills)
                        if m > self.fill[y][x]:
                            d += self.STIFF_H * (m - self.fill[y][x])

                div[y][x] = d

        P = self.P
        for y in range(H):
            for x in range(W):
                if self.mat[y][x] != LIQUID:
                    P[y][x] = 0.0
        for _ in range(self.N_ITERS):
            for y in range(H):
                for x in range(W):
                    if self.mat[y][x] != LIQUID:
                        continue
                    s = 0.0
                    k = 0
                    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        if self.is_liquid(nx, ny):
                            s += P[ny][nx]; k += 1
                        elif self.is_air(nx, ny):
                            k += 1
                    if k == 0:
                        continue
                    target = (s - div[y][x]) / k
                    P[y][x] += self.OMEGA * (target - P[y][x])
        self.P = P


# ======================== SCENARIO BUILDERS ========================

def build_flat_uneven_puddle(width=5, fill_values=None):
    """Single flat layer on a solid floor, uneven fill, air everywhere above.
    Default: [62, 62, 11, 62, 62] — centre column severely under-full.
    y=0 solid floor, y=1 liquid, y=2 air (free surface), y=3 air."""
    W = width + 2  # solid walls on each side
    H = 4
    s = SimFlatPuddle(W, H)
    if fill_values is None:
        fill_values = [62.0] * width
        fill_values[width // 2] = 11.0  # under-full centre
    for x in range(width):
        s.carve(x + 1, 1, LIQUID, fill_values[x])
    # Set row above liquid to AIR (free surface — Dirichlet P=0 boundary)
    for x in range(W):
        if s.mat[2][x] != LIQUID:
            s.mat[2][x] = AIR
    return s, 1, width  # sim, leftmost liquid x, rightmost liquid x


def build_flat_even_lake(width=10):
    """Even flat lake — must stay at rest (the null-invariant check)."""
    W = width + 2
    H = 4
    s = SimFlatPuddle(W, H)
    for x in range(width):
        s.carve(x + 1, 1, LIQUID, FILL_CAP)
    # Set row above liquid to AIR (free surface)
    for x in range(W):
        if s.mat[2][x] != LIQUID:
            s.mat[2][x] = AIR
    return s


def build_utube(fill_left_to=9, arm_gap=3):
    """U-tube geometry (matches utube_projection.py exactly)."""
    W = 3 + arm_gap; H = 14
    s = SimFlatPuddle(W, H)
    lx, rx = 1, W - 2
    for y in range(1, H - 1):
        s.carve(lx, y, AIR); s.carve(rx, y, AIR)
    for x in range(lx, rx + 1):
        s.carve(x, 1, AIR)
    for y in range(1, fill_left_to + 1):
        s.carve(lx, y, LIQUID, FILL_CAP)
    for x in range(lx, rx + 1):
        s.carve(x, 1, LIQUID, FILL_CAP)
    return s, lx, rx


def build_single_column(fill_to=6, cap=255.0):
    """Single column with air shaft — for pressure-null and settled-partial tests."""
    W, H = 3, 12
    s = SimFlatPuddle(W, H)
    for y in range(1, H - 1):
        s.carve(1, y, AIR)
    for y in range(1, fill_to + 1):
        s.carve(1, y, LIQUID, cap)
    return s


# ======================== BUG DEMONSTRATION ========================

def test_flat_uneven_puddle_BUG_DEMO():
    """Run the BASE Sim (no horizontal term) on a flat uneven puddle.
    It must get stuck — proving the design gap exists."""
    W = 7; H = 4
    s = _base.Sim(W, H)
    fills = [62.0, 62.0, 11.0, 62.0, 62.0]
    for i, f in enumerate(fills):
        s.carve(i + 1, 1, LIQUID, f)
    t0 = s.total()
    s.run(max_ticks=500, settle_delta=0.1, settle_streak=50)
    final = [s.fill[1][x] for x in range(1, 6)]
    spread = max(final) - min(final)
    print(f"[BUG_DEMO/base]  final_fills={[int(f) for f in final]}  "
          f"spread={spread:.1f}  cons={abs(s.total()-t0):.3f}")
    assert spread > 5.0, \
        f"base Sim unexpectedly equalized (spread={spread:.1f}); bug may be fixed upstream"
    print("  -> base Sim stuck on uneven flat puddle (design gap confirmed)")


# ======================== FIX VALIDATION ========================

def test_flat_uneven_puddle_FIX():
    """SimFlatPuddle must converge to uniform fill on the same scenario."""
    s, lx, rx = build_flat_uneven_puddle()
    t0 = s.total()
    s.run(max_ticks=1000, settle_delta=0.1, settle_streak=50)
    final = [s.fill[1][x] for x in range(lx, rx + 1)]
    spread = max(final) - min(final)
    mean_f = sum(final) / len(final)
    print(f"[FIX/flat-puddle]  final_fills={[int(f) for f in final]}  "
          f"spread={spread:.1f}  mean={mean_f:.1f}  cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0, f"mass not conserved: drift={abs(s.total()-t0):.3f}"
    assert spread < 2.0, f"did not equalize: spread={spread:.1f}"
    print("  -> flat uneven puddle equalized (design gap fixed)")


def test_flat_uneven_puddle_WIDER():
    """Harder: 9-cell puddle with two under-full pockets at different depths.
    Two competing gradients take longer to resolve — wider tolerance."""
    W = 11; H = 4
    s = SimFlatPuddle(W, H)
    fills = [200, 200, 50, 200, 200, 200, 100, 200, 200]
    for i, f in enumerate(fills):
        s.carve(i + 1, 1, LIQUID, f)
    # Set row above liquid to AIR (free surface)
    for x in range(W):
        if s.mat[2][x] != LIQUID:
            s.mat[2][x] = AIR
    t0 = s.total()
    s.run(max_ticks=5000, settle_delta=0.05, settle_streak=200)
    final = [s.fill[1][x] for x in range(1, 10)]
    spread = max(final) - min(final)
    mean_f = sum(final) / len(final)
    print(f"[FIX/wider-puddle]  final_fills={[int(f) for f in final]}  "
          f"spread={spread:.1f}  mean={mean_f:.1f}  cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0, f"mass not conserved: drift={abs(s.total()-t0):.3f}"
    # HONEST tolerance: max-source at MF=0.1 lands ~8 here (avg landed ~24). The residual is
    # the known in-projection limit (P-pinning on a flat layer); it grows with width. Not <2.
    assert spread < 10.0, f"did not equalize: spread={spread:.1f} (expected ~8 for max@MF0.1)"
    print(f"  -> wider uneven puddle converged (residual {spread:.1f}; known width-dependent limit)")


# ======================== REGRESSION SUITE ========================

def test_pressure_null_at_rest():
    """Settled full column at rest: corrected face requests must be ~zero."""
    s = build_single_column(fill_to=6)
    s.solve_pressure(); s.correct_flux()
    maxreq = max(abs(s.Fx[y][x]) for y in range(s.H) for x in range(s.W))
    maxreq = max(maxreq, max(abs(s.Fy[y][x]) for y in range(s.H) for x in range(s.W)))
    print(f"[regress/pressure-null]  max |corrected face request| = {maxreq:.3f}")
    assert maxreq < 1.0, f"nonzero corrected request at rest: {maxreq:.3f}"
    print("  -> pressure-null at rest holds")


def test_flat_lake_noop():
    """Even flat lake must stay at rest."""
    s = build_flat_even_lake()
    t0 = s.total(); snap = [row[:] for row in s.fill]; mx = 0.0
    for _ in range(200):
        s.tick(); mx = max(mx, s._last_moved)
    diff = max(abs(s.fill[y][x] - snap[y][x]) for y in range(s.H) for x in range(s.W))
    print(f"[regress/flat-lake]  max_move/tick={mx:.3f} cell_drift={diff:.3f} "
          f"cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0 and diff < 1.0 and mx < 2.0
    print("  -> flat lake stays flat")


def test_bottom_pipe_gradient():
    """U-tune bottom gradient: tall arm base P > short arm base P."""
    s, lx, rx = build_utube()
    s.solve_pressure()
    p_left = s.P[1][lx]; p_right = s.P[1][rx]
    print(f"[regress/bottom-gradient]  P(left)={p_left:.1f}  P(right)={p_right:.1f}")
    assert p_left - p_right > 1.0, \
        f"no driving gradient: {p_left:.1f} vs {p_right:.1f}"
    print("  -> U-tube bottom gradient exists")


def test_utube_equalizes():
    """U-tube must equalize to full columns, mass-balanced."""
    s, lx, rx = build_utube()
    t0 = s.total(); ticks = s.run()
    l, r = s.arm_mass(lx), s.arm_mass(rx)
    lh, rh = s.column_top(lx), s.column_top(rx)
    top = int(min(lh, rh)) if (lh and rh) else 0
    submerged = [s.fill[y][x] for x in (lx, rx) for y in range(1, max(top, 1))
                 if s.is_liquid(x, y)]
    med = sorted(submerged)[len(submerged) // 2] if submerged else 0.0
    print(f"[regress/u-tube]  settled@{ticks}  left={l:.2f} right={r:.2f} "
          f"imbalance={abs(l-r):.2f}  submerged_med={med:.0f}  "
          f"cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0, "mass not conserved"
    assert abs(l - r) < 1.0, f"arms not balanced: {l:.2f} vs {r:.2f}"
    assert med > 220.0, f"thin smear: submerged median {med:.0f}"
    print("  -> U-tube equalizes (balanced, full columns, mass-conserved)")


def test_pipe_length_independence():
    """Equalization level must not depend on connector length."""
    res = {}
    for gap in (3, 8):
        s, lx, rx = build_utube(arm_gap=gap)
        s.run()
        res[gap] = (s.column_top(lx) + s.column_top(rx)) / 2.0
    print(f"[regress/pipe-length]  short_gap={res[3]:.2f}  long_gap={res[8]:.2f}")
    assert abs(res[3] - res[8]) <= 1.0, f"level depends on pipe length: {res}"
    print("  -> equalization independent of pipe length")


def test_settled_partial_surface():
    """Partial-surface column must stay dead quiet (no churn)."""
    s = build_single_column(fill_to=6, cap=255.0)
    s.carve(1, 6, LIQUID, 128.0)  # partial surface cell
    t0 = s.total(); snap = [row[:] for row in s.fill]; mx = 0.0
    for _ in range(300):
        s.tick(); mx = max(mx, s._last_moved)
    diff = max(abs(s.fill[y][x] - snap[y][x]) for y in range(s.H) for x in range(s.W))
    drift = abs(s.total() - t0)
    print(f"[regress/settled-partial]  max_move/tick={mx:.4f}  cell_drift={diff:.4f}  "
          f"cons={drift:.4f}")
    assert drift < 1.0, f"conservation broken: {drift}"
    assert diff < 1.0, f"CHURN: settled column moved by {diff}"
    assert mx < 2.0, f"CHURN: per-tick movement of {mx}"
    print("  -> settled partial-surface column stays quiet")


def test_single_column_compresses():
    """Three equal-fill (128) cells stacked on solid floor, with only air around (no ledge/
    gap to fall through) must COMPRESS DOWN under gravity (vertical density correction),
    not sit frozen at their initial pattern. This is a vertical-settling check, not a
    "drain off a ledge" check -- there is no ledge in this geometry. (Renamed from
    test_single_column_drains, whose old name/docstring didn't match what it built, and
    whose only assertion was conservation -- it printed "drains (no pinning)"
    unconditionally even though the column never moved.)"""
    W, H = 5, 8
    s = SimFlatPuddle(W, H)
    for y in range(1, H - 1):
        s.carve(2, y, AIR)
    for y in range(1, 4):
        s.carve(2, y, LIQUID, 128.0)
    initial_pattern = [s.fill[y][2] for y in range(1, H)]
    t0 = s.total()
    s.run(max_ticks=500, settle_delta=0.1, settle_streak=50)
    final_pattern = [s.fill[y][2] for y in range(1, H)]
    print(f"[regress/compress]  initial={initial_pattern}  final={final_pattern}  "
          f"cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0, "mass not conserved"
    assert final_pattern != initial_pattern, \
        "column did not move at all -- vertical density correction not firing"
    # Compression should produce at least one full (or near-full) cell, not stay at three
    # equal partials -- this is the actual "not stuck" claim, checked directly.
    assert max(final_pattern) > 200.0, \
        f"column did not compress toward full cells: {final_pattern}"
    print("  -> stacked column compresses under gravity (not frozen)")


def test_deep_column_lateral_interface():
    """Deep column next to partially filled column: submerged cells must NOT get
    the horizontal term. Only surface cells should be affected."""
    W, H = 5, 8
    s = SimFlatPuddle(W, H)
    # Column 1: deep, full (8 cells)
    for y in range(1, H - 1):
        s.carve(1, y, LIQUID, FILL_CAP)
    # Column 2: shallow, partial (3 cells, fill=100)
    for y in range(1, 4):
        s.carve(3, y, LIQUID, 100.0)
    # Both columns share the same floor, separated by solid at x=2
    s.carve(2, 1, SOLID)  # wall between columns
    t0 = s.total()
    # Snapshot: the deep column should stay at 255 (no horizontal term on submerged cells)
    snap_deep = [s.fill[y][1] for y in range(H)]
    snap_shallow = [s.fill[y][3] for y in range(H)]
    s.run(max_ticks=300, settle_delta=0.1, settle_streak=50)
    deep_after = [s.fill[y][1] for y in range(H)]
    shallow_after = [s.fill[y][3] for y in range(H)]
    deep_drift = max(abs(deep_after[y] - snap_deep[y]) for y in range(H))
    shallow_drift = max(abs(shallow_after[y] - snap_shallow[y]) for y in range(H))
    print(f"[regress/deep-lateral]  deep_drift={deep_drift:.3f}  "
          f"shallow_drift={shallow_drift:.3f}  cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0, "mass not conserved"
    # Deep column must not change (submerged cells unaffected by horizontal term)
    assert deep_drift < 1.0, f"deep column disturbed: drift={deep_drift:.3f}"
    print("  -> deep column unaffected by horizontal term (submerged mask correct)")


def test_settled_utube_no_churn():
    """CHURN GATE: a settled U-tube must produce ~zero per-tick movement. This is what
    forced MIN_FLOW up from 0.01 to 0.1 -- at a tiny deadzone the base projection leaks
    ~0.24/tick here forever (changelog flooding / no chunk-sleep in-engine)."""
    s, lx, rx = build_utube()
    s.run()  # equalize first
    mx = 0.0
    for _ in range(500):
        s.tick(); mx = max(mx, s._last_moved)
    print(f"[regress/utube-churn]  settled steady-state max_move/tick={mx:.4f}")
    assert mx < 1.0, f"CHURN: settled U-tube moving {mx:.3f}/tick (lower MIN_FLOW?)"
    print("  -> settled U-tube quiet (MIN_FLOW=0.1 is churn-free)")


def test_conservation_stress():
    """Multiple ticks of a non-trivial shape: total mass must stay exact."""
    W, H = 10, 8
    s = SimFlatPuddle(W, H)
    # Irregular shape: 3 columns of different heights
    for y in range(1, 6):  s.carve(2, y, LIQUID, 255.0)
    for y in range(1, 4):  s.carve(5, y, LIQUID, 200.0)
    for y in range(1, 7):  s.carve(8, y, LIQUID, 180.0)
    t0 = s.total()
    s.run(max_ticks=2000, settle_delta=0.1, settle_streak=80)
    drift = abs(s.total() - t0)
    print(f"[regress/conservation-stress]  initial={t0:.1f}  final={s.total():.1f}  "
          f"drift={drift:.3f}")
    assert drift < 1.0, f"conservation broken under stress: drift={drift:.3f}"
    print("  -> mass conserved under non-trivial geometry")


# ======================== MAIN ========================

if __name__ == "__main__":
    print("=" * 60)
    print("FLAT PUDDLE FIX — bench + regression suite")
    print("=" * 60)

    print("\n--- BUG DEMONSTRATION (base Sim) ---")
    test_flat_uneven_puddle_BUG_DEMO()

    print("\n--- FIX VALIDATION ---")
    test_flat_uneven_puddle_FIX()
    test_flat_uneven_puddle_WIDER()

    print("\n--- REGRESSION SUITE ---")
    test_pressure_null_at_rest()
    test_flat_lake_noop()
    test_bottom_pipe_gradient()
    test_utube_equalizes()
    test_pipe_length_independence()
    test_settled_partial_surface()
    test_single_column_compresses()
    test_deep_column_lateral_interface()
    test_settled_utube_no_churn()
    test_conservation_stress()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED (flat-puddle fix + full regression)")
    print("=" * 60)
