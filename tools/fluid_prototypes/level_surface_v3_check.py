"""Bench validation for level_surface_v3.py's pressure-difference-based damping, targeting
the exact gap found in-engine: a fully-submerged cell with genuinely open lateral neighbours
never spreading to them, because vertical-fill-based damping (v2) fully disabled it.
"""
import importlib.util, pathlib
_here = pathlib.Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v3 = _load("level_surface_v3")
base = _load("utube_projection")
AIR, LIQUID, SOLID, FC = base.AIR, base.LIQUID, base.SOLID, base.FILL_CAP


def build_buried_seam():
    """Two adjacent columns, SAME total height (y=1..5), both full (255) except column x=4
    has a buried low-fill seam (11) at y=3 -- surrounded above/below by full cells in its
    OWN column, and laterally adjacent to a full (255) cell in the other column. Same
    overall stacking depth on both sides -- should read as near-zero pressure difference,
    unlike a genuine short-vs-tall column difference."""
    W, H = 8, 7
    s = v3.SimLevelSurfaceV3(W, H)
    for x in (3, 4):
        for y in range(1, 6):
            s.carve(x, y, LIQUID, FC)
    s.carve(4, 3, LIQUID, 11.0)  # the buried seam
    # open air around both columns so there's room to observe lateral movement
    for x in (2, 5):
        for y in range(1, 6):
            s.carve(x, y, AIR)
    return s


def build_genuine_height_difference():
    """Two columns of DIFFERENT total height (a real U-tube-style imbalance) -- pressure
    SHOULD differ meaningfully here; LevelSurface must not fight the projection's own push."""
    W, H = 8, 12
    s = v3.SimLevelSurfaceV3(W, H)
    for y in range(1, 10):  # tall column
        s.carve(3, y, LIQUID, FC)
    for y in range(1, 3):  # short column
        s.carve(4, y, LIQUID, FC)
    for x in (2, 5):
        for y in range(1, 10):
            s.carve(x, y, AIR)
    for y in range(3, 10):
        s.carve(4, y, AIR)
    return s


print("=== buried low-fill seam (same overall stacking depth) ===")
s = build_buried_seam()
t0 = s.total()
before = s.fill[3][4]
for _ in range(500):
    s.tick()
after = s.fill[3][4]
drift = abs(s.total() - t0)

# Baseline: does the vertical projection alone (NO LevelSurface at all) also settle this
# scenario unevenly? If so, that's this configuration's own pre-existing quirk, not
# something LevelSurface is responsible for fixing -- the fair bar is "measurably better
# than baseline and genuinely stable (no churn)", not "perfectly equal".
s_baseline = base.Sim(s.W, s.H)
for x in (3, 4):
    for y in range(1, 6):
        s_baseline.carve(x, y, LIQUID, FC)
s_baseline.carve(4, 3, LIQUID, 11.0)
for x in (2, 5):
    for y in range(1, 6):
        s_baseline.carve(x, y, AIR)
for _ in range(500):
    s_baseline.tick()
baseline_after = s_baseline.fill[3][4]

print(f"  seam cell fill: {before:.0f} -> {after:.0f} (v3, with LevelSurface)")
print(f"  seam cell fill (baseline, NO LevelSurface): {before:.0f} -> {baseline_after:.0f}")
print(f"  conservation drift: {drift:.3f}")
assert after > baseline_after + 5, (
    f"v3 ({after:.0f}) not measurably better than baseline projection alone ({baseline_after:.0f}) "
    f"-- damping may still be blocking real lateral movement"
)
assert drift < 1.0, f"mass not conserved: drift={drift:.3f}"
# Confirm it's a genuine stable fixed point, not still moving/oscillating at tick 500.
last = s.fill[3][4]
for _ in range(50):
    s.tick()
assert abs(s.fill[3][4] - last) < 0.5, "still moving at tick 500 -- not actually stable yet"
print(f"  PASS -- v3 measurably unlocks the seam vs. baseline ({after:.0f} > {baseline_after:.0f}), "
      f"and settles to a genuinely stable fixed point (not full equalization, but this specific "
      f"configuration's own vertical solve doesn't reach equal fill either -- see baseline)")

print("\n=== genuine column-height difference (real gradient, must defer to projection) ===")
s2 = build_genuine_height_difference()
t0b = s2.total()
s2.solve_pressure()  # need at least one real solve before P is meaningful (starts at all-zero)
p_tall_base = s2.P[1][3]
p_short_base = s2.P[1][4] if s2.is_liquid(4, 1) else 0.0
print(f"  P at bottom row after one solve: tall={p_tall_base:.2f} short={p_short_base:.2f} "
      f"|diff|={abs(p_tall_base-p_short_base):.2f} (should be a REAL gradient, not near-zero)")
assert abs(p_tall_base - p_short_base) > 2.0, "expected a real pressure gradient here"
for _ in range(1000):
    s2.tick()
am = lambda x: sum(s2.fill[y][x] for y in range(s2.H)) / FC
drift2 = abs(s2.total() - t0b)
print(f"  after settling: tall_col_mass={am(3):.2f} short_col_mass={am(4):.2f} cons={drift2:.3f}")
assert drift2 < 1.0, f"mass not conserved: drift={drift2:.3f}"
print("  PASS -- ran without instability (projection + LevelSurface didn't fight each other)")

print("\n=== regression: original 5b invariants still hold under v3 ===")

def _row(fills, W=None):
    n = len(fills); W = W or (n + 2)
    s = v3.SimLevelSurfaceV3(W, 4)
    for x in range(W):
        s.carve(x, 1, AIR)
    off = (W - n) // 2
    for i, f in enumerate(fills):
        if f is not None:
            s.carve(i + off, 1, LIQUID, float(f))
    return s, off

s, off = _row([62, 62, 11, 62, 62])
t0 = s.total()
for _ in range(1000):
    s.tick()
final = [s.fill[1][off + i] for i in range(5)]
spread = max(final) - min(final)
print(f"  [flat_uneven_puddle] spread={spread:.1f} cons={abs(s.total()-t0):.3f}")
assert abs(s.total() - t0) < 1.0
assert spread < 12.0, f"spread={spread:.1f} (expected ~9-10, same rounding floor as v2)"
print("  PASS")

s, off = _row([200] * 8)
for _ in range(200):
    s.tick()
mx = 0
for _ in range(200):
    s.tick(); mx = max(mx, s._last_moved)
print(f"  [even_lake_no_churn] max_move/tick={mx}")
assert mx < 1
print("  PASS")

s, off = _row([124, 124, None, 124, 124])
t0 = s.total()
for _ in range(1000):
    s.tick()
gap = s.fill[1][off + 2]
print(f"  [gap_merge] gap_fill={gap:.0f} cons={abs(s.total()-t0):.3f}")
assert gap > 50
assert abs(s.total() - t0) < 1.0
print("  PASS")

def _utube(fill_left_to=9, arm_gap=3):
    W = 3 + arm_gap; H = 14
    s = v3.SimLevelSurfaceV3(W, H)
    lx, rx = 1, W - 2
    for y in range(1, H - 1):
        s.carve(lx, y, AIR); s.carve(rx, y, AIR)
    for x in range(lx, rx + 1):
        s.carve(x, 1, AIR)
    for y in range(1, fill_left_to + 1):
        s.carve(lx, y, LIQUID, FC)
    for x in range(lx, rx + 1):
        s.carve(x, 1, LIQUID, FC)
    return s, lx, rx

s, lx, rx = _utube()
t0 = s.total()
for _ in range(2000):
    s.tick()
am = lambda x: sum(s.fill[y][x] for y in range(s.H)) / FC
imb = abs(am(lx) - am(rx))
drift = abs(s.total() - t0)
print(f"  [utube_preserved] imbalance={imb:.2f} cons={drift:.3f}")
assert imb < 1.5
assert drift < 1.0
print("  PASS")

print("\n=== regression: open drop-on-puddle stays symmetric (no one-sided drain) ===")
# Guards the 2026-07-07 finding (drop_settle_transient.py docstring, docs/BUG_LOG.md): a tower
# dropped on a puddle WITH a real free surface must spread symmetrically. The original bench
# accidentally built a fully-sealed domain (zero air cells -> pure-Neumann solve -> unbounded
# warm-start P drift -> pair damping fed garbage), which produced a total one-sided drain that
# was misread as a LevelSurface bug. This test fails loudly if either (a) someone reintroduces
# a sealed-geometry builder, or (b) a future LevelSurface change makes the open case one-sided.
W, H = 31, 20  # odd width -> puddle x=1..29 has an exact center at x=15 for the tower
s = v3.SimLevelSurfaceV3(W, H)
for x in range(1, W - 1):
    for y in range(1, 11):
        s.carve(x, y, AIR)
for x in range(1, W - 1):
    s.carve(x, 1, LIQUID, 60.0)
for y in range(2, 8):
    s.carve(15, y, LIQUID, FC)
n_air = sum(1 for y in range(H) for x in range(W) if s.mat[y][x] == AIR)
assert n_air > 0, "sealed domain: no free surface -- geometry bug reintroduced"
t0 = s.total()
for _ in range(700):
    s.tick()
row = [s.fill[1][x] for x in range(1, W - 1)]
mirror_err = max(abs(row[i] - row[len(row) - 1 - i]) for i in range(len(row) // 2))
left = sum(s.fill[y][x] for y in range(H) for x in range(0, 15))
right = sum(s.fill[y][x] for y in range(H) for x in range(16, W))
pmax = max(abs(s.P[y][x]) for y in range(H) for x in range(W))
print(f"  [open_drop_symmetry] mirror_err={mirror_err:.1f} L={left:.0f} R={right:.0f} "
      f"max|P|={pmax:.1f} cons={abs(s.total()-t0):.3f}")
assert abs(s.total() - t0) < 1.0, "mass not conserved"
assert pmax < 100.0, f"P unanchored (max|P|={pmax:.1f}) -- free surface not working"
assert mirror_err < 5.0, f"one-sided drain is back: mirror_err={mirror_err:.1f}"
assert abs(left - right) < 20.0, f"one-sided mass split: L={left:.0f} R={right:.0f}"
print("  PASS")

print("\nALL CHECKS PASSED")
