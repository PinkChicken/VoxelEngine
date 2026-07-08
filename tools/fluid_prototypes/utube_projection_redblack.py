"""
Checkpoint 1 (pressure-projection port plan, M1) — red-black Gauss-Seidel parity check.

utube_projection.py's solve_pressure() uses SERIAL Gauss-Seidel/SOR (each cell reads
neighbours' values from the SAME in-progress sweep, in row-major order) — a genuine
sequential dependency that cannot run on a GPU (no cross-workgroup ordering guarantee
within one dispatch). The actual GLSL port must use RED-BLACK coloring instead: split
each iteration into two half-sweeps by (x+y+z) parity, so within a half-sweep every cell
reads only the OTHER color's values (already fixed for this iteration) — parallel-safe.

This script subclasses Sim and overrides only solve_pressure() with red-black ordering,
re-running the exact same bench invariants from utube_projection.py, to confirm the
design still holds under the ordering the GPU will actually use, and to get a rough
iteration-count/omega starting point before writing any GLSL. Serial numbers (OMEGA=1.7,
N_ITERS=60) do NOT transfer directly -- re-measured here.
"""

import importlib.util
import pathlib

_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("utube_projection", _here / "utube_projection.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

AIR, LIQUID, SOLID = _base.AIR, _base.LIQUID, _base.SOLID
FILL_CAP = _base.FILL_CAP


class SimRB(_base.Sim):
    """Same physics as Sim, but solve_pressure() uses red-black coloring instead of
    serial in-place Gauss-Seidel -- the ordering the GPU port will actually use."""

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
                if self.is_liquid(x, y + 1):
                    d += self.STIFF * (FILL_CAP - self.fill[y][x])
                div[y][x] = d

        P = self.P
        for y in range(H):
            for x in range(W):
                if self.mat[y][x] != LIQUID:
                    P[y][x] = 0.0

        for _ in range(self.N_ITERS):
            for color in (0, 1):
                for y in range(H):
                    for x in range(W):
                        if self.mat[y][x] != LIQUID:
                            continue
                        if (x + y) & 1 != color:
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


def build_utube(fill_left_to=9, arm_gap=3):
    W = 3 + arm_gap; H = 14
    s = SimRB(W, H)
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


def test_bottom_pipe_gradient():
    s, lx, rx = build_utube()
    s.solve_pressure()
    p_left_base = s.P[1][lx]
    p_right_base = s.P[1][rx]
    print(f"[bottom-gradient/RB]  P(left base)={p_left_base:.1f}  P(right base)={p_right_base:.1f}")
    assert p_left_base - p_right_base > 1.0, \
        f"no driving gradient across the pipe: {p_left_base:.1f} vs {p_right_base:.1f}"
    print("  -> tall arm has higher base pressure: real lateral gradient exists (red-black)")


def test_pressure_null_at_rest():
    W, H = 3, 10
    s = SimRB(W, H)
    for y in range(1, H - 1):
        s.carve(1, y, AIR)
    for y in range(1, 6):
        s.carve(1, y, LIQUID, FILL_CAP)
    s.solve_pressure(); s.correct_flux()
    maxreq = max(abs(s.Fx[y][x]) for y in range(H) for x in range(W))
    maxreq = max(maxreq, max(abs(s.Fy[y][x]) for y in range(H) for x in range(W)))
    print(f"[pressure-null/RB]  max |corrected face request| at rest = {maxreq:.3f}")
    assert maxreq < 1.0, f"nonzero corrected request at rest: {maxreq:.3f}"
    print("  -> settled column produces zero corrected flux (red-black)")


def test_flat_lake_noop():
    W, H = 12, 8
    s = SimRB(W, H)
    for x in range(1, W - 1):
        for y in range(1, 5):
            s.carve(x, y, LIQUID, FILL_CAP)
    t0 = s.total(); snap = [row[:] for row in s.fill]; mx = 0.0
    for _ in range(150):
        s.tick(); mx = max(mx, s._last_moved)
    diff = max(abs(s.fill[y][x] - snap[y][x]) for y in range(H) for x in range(W))
    print(f"[flat-lake/RB]  max_move/tick={mx:.3f} cell_drift={diff:.3f} cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0 and diff < 1.0 and mx < 2.0
    print("  -> flat lake stays flat (red-black)")


def test_utube_equalizes():
    s, lx, rx = build_utube()
    t0 = s.total(); ticks = s.run()
    l, r = s.arm_mass(lx), s.arm_mass(rx)
    lh, rh = s.column_top(lx), s.column_top(rx)
    top = int(min(lh, rh)) if (lh and rh) else 0
    submerged = [s.fill[y][x] for x in (lx, rx) for y in range(1, max(top, 1)) if s.is_liquid(x, y)]
    med = sorted(submerged)[len(submerged) // 2] if submerged else 0.0
    print(f"[u-tube/RB]  settled@{ticks}  left_arm={l:.2f} right_arm={r:.2f} imbalance={abs(l-r):.2f} "
          f"submerged_med={med:.0f} cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0, "mass not conserved"
    assert abs(l - r) < 1.0, f"arms not balanced: {l:.2f} vs {r:.2f}"
    assert med > 220.0, f"thin smear: submerged median {med:.0f}"
    print("  -> U-tube equalizes (balanced arms, full columns, mass conserved) -- red-black")


def test_pipe_length_same_level():
    res = {}
    for gap in (3, 8):
        s, lx, rx = build_utube(arm_gap=gap)
        s.run()
        res[gap] = (s.column_top(lx) + s.column_top(rx)) / 2.0
    print(f"[pipe-length/RB]  short_gap_level={res[3]:.2f}  long_gap_level={res[8]:.2f}")
    assert abs(res[3] - res[8]) <= 1.0, f"level depends on pipe length: {res}"
    print("  -> short and long U-tubes reach the same level (red-black)")


if __name__ == "__main__":
    test_pressure_null_at_rest()
    test_bottom_pipe_gradient()
    test_flat_lake_noop()
    test_utube_equalizes()
    test_pipe_length_same_level()
    print("\nALL TESTS PASSED (red-black)")
