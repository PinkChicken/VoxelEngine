"""Bench investigation: in-engine, a voxel dropped from height onto an existing puddle showed a
striking transient during settling -- liquid_voxels count collapsing hard (e.g. 45 -> 15) before
climbing back (15 -> 18 -> 22 -> ... -> 61), i.e. mass consolidating into fewer, fuller cells then
re-spreading, alongside large corrective fy flux right after LevelSurface acts. User asked to
investigate whether this is a LevelSurface-induced coupling bug or pre-existing behavior.

Hypothesis: utube_projection.Sim's STIFF density-correction term (solve_pressure, line ~82-83)
adds `STIFF * (FILL_CAP - fill[y][x])` to a cell's divergence whenever it has same-liquid directly
above it ("submerged"). A falling stream settling onto a puddle naturally creates multi-layer
stacks with partial fill -- exactly what this term reacts to -- independent of LevelSurface. This
bench isolates the mechanism: run the SAME "drop a tall column onto a puddle" scenario twice, once
with LevelSurface enabled and once with the base Sim (LevelSurface entirely absent), and compare
the liquid-cell-count trajectory in both.

RESOLVED (2026-07-07): the "one-sided drain" this bench originally showed was a BENCH GEOMETRY
BUG, not a LevelSurface bug. The first version of build_drop_on_puddle carved AIR at y=1 and then
immediately overwrote those same cells with LIQUID -- leaving ZERO air cells in the whole domain
(everything above the puddle stayed SOLID). That made the pressure solve a pure-Neumann problem
with no free-surface (Dirichlet P=0) anchor anywhere, and the STIFF source under the tower made
the RHS incompatible, so the warm-started P field drifted unboundedly (observed: -2000 -> -3150
and falling over ~10 ticks). LevelSurface's pair damping compares |P_self - P_nb| against
PRESSURE_MIN=2, so those meaningless drifting gradients modulated damping arbitrarily and, once a
cell dried, |P_liquid - 0| >> PRESSURE_MIN permanently vetoed re-wetting. The one-sided DIRECTION
was seeded by the tower sitting at W//2=15 while the 28-cell puddle's true center is 14.5 (a
perfectly centered tower in W=31 stays mirror-exact for 700+ ticks; the drain side follows the
tower's offset). With the geometry fixed (real air headspace, below), the drain does not exist:
LevelSurface spreads the tower symmetrically and stalls at its known per-pair rounding floor
(slope ~10 fill/cell => a pyramid, not a flat puddle). See level_surface_v3_check.py's
open_drop_symmetry regression and docs/BUG_LOG.md.
"""
import importlib.util, pathlib
_here = pathlib.Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v3 = _load("level_surface_v3")
base = _load("utube_projection")
AIR, LIQUID, SOLID, FC = base.AIR, base.LIQUID, base.SOLID, base.FILL_CAP


def build_drop_on_puddle(sim_cls, W=30, H=20):
    """A flat, wide puddle (fill=60 per cell, well below full) with a tall, fully-full column
    dropped on top of its center -- mirrors 'drop a voxel from height onto an existing puddle'.
    Wide enough (W=30) to give a collapsed stack real room to re-expand laterally, unlike the
    first attempt (W=12) which plateaued instead of showing the in-engine recovery phase."""
    s = sim_cls(W, H)
    # air headspace ABOVE the puddle, not just at y=1 (the puddle carve below overwrites y=1).
    # Without this the whole domain is sealed solid -> pure-Neumann pressure solve with no
    # free-surface anchor -> unbounded warm-start P drift -> garbage LevelSurface damping
    # (the original one-sided-drain artifact, see module docstring).
    for x in range(1, W - 1):
        for y in range(1, 11):
            s.carve(x, y, AIR)
    # existing puddle: one layer, partially filled, several cells wide
    for x in range(1, W - 1):
        s.carve(x, 1, LIQUID, 60.0)
    # tall stack landing on the center -- simulates an accumulated falling stream
    cx = W // 2
    for y in range(2, 8):
        s.carve(cx, y, LIQUID, FC)
    n_air = sum(1 for y in range(s.H) for x in range(s.W) if s.mat[y][x] == AIR)
    assert n_air > 0, "domain is sealed (no free surface) -- pressure solve has no P=0 anchor"
    return s


def liquid_cell_count(s):
    return sum(1 for y in range(s.H) for x in range(s.W) if s.is_liquid(x, y))


def run_trace(sim_cls, label, ticks=3000):
    s = build_drop_on_puddle(sim_cls)
    t0 = s.total()
    counts = []
    moved = []
    for i in range(ticks):
        s.tick()
        counts.append(liquid_cell_count(s))
        moved.append(s._last_moved)
    drift = abs(s.total() - t0)
    lo = min(counts)
    lo_tick = counts.index(lo)
    print(f"[{label}] start_count={liquid_cell_count(build_drop_on_puddle(sim_cls))} "
          f"min_count={lo} @tick{lo_tick}  end_count={counts[-1]}  cons_drift={drift:.3f}")
    # print a coarse trajectory (every N ticks) so the collapse-then-recover shape is visible
    step = ticks // 30
    print("   count traj:", [counts[i] for i in range(0, ticks, step)])
    print("   moved traj:", [round(moved[i], 1) for i in range(0, ticks, step)])
    spread = max(s.fill[1][x] for x in range(1, s.W - 1)) - min(s.fill[1][x] for x in range(1, s.W - 1))
    print(f"   final bottom-row spread (max-min fill) = {spread:.1f}")
    return counts, drift


print("=== drop-on-puddle transient: baseline (NO LevelSurface) vs v3 (WITH LevelSurface) ===")
counts_base, drift_base = run_trace(base.Sim, "baseline")
counts_v3, drift_v3 = run_trace(v3.SimLevelSurfaceV3, "with-LevelSurface")

dip_base = min(counts_base)
dip_v3 = min(counts_v3)
print(f"\nDeepest collapse: baseline min={dip_base}  with-LevelSurface min={dip_v3}")
if dip_v3 < dip_base - 3:
    print("=> LevelSurface measurably WORSENS the collapse-then-recover transient.")
elif dip_base < dip_v3 - 3:
    print("=> LevelSurface measurably IMPROVES the transient versus baseline.")
else:
    print("=> No meaningful difference -- the collapse-then-recover shape is inherent to the "
          "vertical STIFF-driven settle of a falling stream, not a LevelSurface-specific bug.")
