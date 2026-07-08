"""Combines LevelSurface v3 (pressure-difference damping) with the red-black pressure solve
(utube_projection_redblack.SimRB) instead of v3's inherited serial row-major Gauss-Seidel.

Why: drop_settle_transient.py found LevelSurface causing a striking ONE-SIDED mass drain (a
monotonic ramp on one side, near-zero on the other) when settling a tower dropped onto a flat
puddle. That bench's solve_pressure() sweeps cells in fixed row-major order (for y: for x:),
which has a small inherent directional bias -- confirmed present even in the LevelSurface-free
baseline (36 vs 77 at the two puddle edges, a much smaller asymmetry than the ~180-unit ramp
seen with LevelSurface). The real GLSL engine instead uses RED-BLACK checkerboard passes
(solve_red/solve_black), which cancel that kind of directional bias by construction. This file
swaps in the already-existing red-black solve (from Checkpoint 1's M1 parity bench) via multiple
inheritance, so the same drop-on-puddle scenario can be re-run under a solver that actually
matches the engine's ordering -- to tell whether the asymmetry is a real LevelSurface bug or a
row-major-sweep artifact of the naive bench.
"""
import importlib.util, pathlib
_here = pathlib.Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

v3 = _load("level_surface_v3")
redblack = _load("utube_projection_redblack")


# NOTE: level_surface_v3.py and utube_projection_redblack.py each load their own SEPARATE
# copy of utube_projection.py's Sim class (independent exec_module calls), so plain multiple
# inheritance (class X(v3.SimLevelSurfaceV3, redblack.SimRB)) does NOT reliably put SimRB's
# solve_pressure ahead of v3's inherited (different-object) Sim.solve_pressure in the MRO --
# confirmed by hand, it silently picked the wrong one. Single-inherit from SimRB instead and
# borrow only the leaf methods that don't call super() (level_surface, _pressure_at,
# _classify) -- tick() is reimplemented directly rather than borrowed, since v3's tick() uses
# zero-arg super(), whose __class__ closure cell is bound to the class it was DEFINED in, not
# wherever it gets reassigned -- borrowing it here would silently resolve super() against the
# wrong class hierarchy.
class SimLevelSurfaceV3RB(redblack.SimRB):
    _pressure_at = v3.SimLevelSurfaceV3._pressure_at
    _classify = v3.SimLevelSurfaceV3._classify
    level_surface = v3.SimLevelSurfaceV3.level_surface

    def __init__(self, W, H):
        super().__init__(W, H)
        self.LEVEL_RATE = 0.20
        self.LEVEL_MIN = 1.0
        self.PRESSURE_MIN = 2.0
        self._last_moved = 0.0

    def tick(self):
        self.solve_pressure()   # SimRB's red-black version
        self.correct_flux()
        self.transport()
        m = self.level_surface()
        self._last_moved = max(self._last_moved, m)


if __name__ == "__main__":
    # quick sanity check: solve_pressure really is SimRB's red-black version.
    s = SimLevelSurfaceV3RB(6, 6)
    assert type(s).solve_pressure is redblack.SimRB.solve_pressure, "did not pick red-black solve_pressure"
    print("check OK: solve_pressure resolves to SimRB's red-black implementation")
