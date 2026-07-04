"""
2D emulator v4 — QUASI-STATIC PRESSURE PROJECTION (Chorin-style) for the fill field.

The max-relaxation family was falsified: max-propagation computes a reachable upper
envelope and flattens the lateral gradient that must exist while arms are unequal
(proved: bottom-row P uniform at 2:1 mass imbalance). Projection fixes this because the
pressure RHS is the *divergence of a provisional flux*, which differs between arms.

Per-tick pipeline (pressure is recomputed, never stored; transport is the only writer):
  1. provisional face flux u* from gravity (downward on faces with water above).
  2. per-liquid-cell divergence  div = net provisional outflow.
  3. Jacobi-solve  laplacian(P) = div   over the LIQUID domain, with:
        - liquid-liquid face : normal coupling
        - liquid-air (free surface) : Dirichlet P = 0
        - solid / sealed face : Neumann (no flow) -> dropped from the stencil
  4. correct:  F = u* - grad(P).
  5. existing shared s_out / s_in budget + conservative transport.
  (No flux persistence in this first version -> quasi-static, isolates correctness.)

Coords: x horizontal, y up. Owned faces: F_x(p) between p,p+X ; F_y(p) between p,p+Y (up).
"""

AIR, LIQUID, SOLID = 0, 1, 2
FILL_CAP = 255.0


class Sim:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.mat = [[SOLID] * W for _ in range(H)]
        self.fill = [[0.0] * W for _ in range(H)]
        self.P = [[0.0] * W for _ in range(H)]
        self.Fx = [[0.0] * W for _ in range(H)]   # corrected face flux (recomputed each tick)
        self.Fy = [[0.0] * W for _ in range(H)]

        self.G = 64.0          # provisional gravity flux magnitude (per open vertical face)
        self.STIFF = 0.5       # density-correction: pull submerged under-full cells toward full
        self.N_ITERS = 60      # SOR iterations per pressure solve
        self.OMEGA = 1.7       # over-relaxation factor (1 = Gauss-Seidel)
        self.MIN_FLOW = 0.5
        self.use_projection = True
        self._last_moved = 0.0

    def inb(self, x, y): return 0 <= x < self.W and 0 <= y < self.H
    def is_liquid(self, x, y): return self.inb(x, y) and self.mat[y][x] == LIQUID
    def is_air(self, x, y): return self.inb(x, y) and self.mat[y][x] == AIR
    def is_solid(self, x, y): return (not self.inb(x, y)) or self.mat[y][x] == SOLID
    def eff(self, v): return v if abs(v) >= self.MIN_FLOW else 0.0

    # ---- provisional gravity flux on owned faces ----
    # F*_y(p) = -G (downward) when there is water in the cell ABOVE (p+Y) that can fall into p,
    # and neither side is solid. F*_x = 0. Represents gravity as a body force before pressure.
    def prov_fy(self, x, y):
        if self.is_solid(x, y) or self.is_solid(x, y + 1):
            return 0.0
        if self.is_liquid(x, y + 1):     # water above wants to fall down into (x,y)
            # Scale by the WEIGHT (fill) of the water above, not a flat constant. This is
            # what injects the fill-imbalance into the divergence RHS: a thin column presses
            # down less than a full one, so it produces less pressure -> a real lateral
            # gradient survives instead of a topology-only constant-at-elevation field.
            return -self.G * (self.fill[y + 1][x] / FILL_CAP)
        return 0.0

    # ---- Jacobi pressure solve: laplacian(P) = divergence, over liquid cells ----
    def solve_pressure(self):
        W, H = self.W, self.H
        # provisional divergence per liquid cell (net outflow of u*)
        div = [[0.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                if self.mat[y][x] != LIQUID:
                    continue
                fy_up = self.prov_fy(x, y)        # face to (x,y+1)
                fy_dn = self.prov_fy(x, y - 1)    # face (x,y-1)->(x,y)
                # F*_x are all 0, so only vertical contributes
                d = (fy_up - fy_dn)               # + (fx_r - fx_l), both 0
                # Density correction (Bridson-style): a SUBMERGED cell (liquid directly above)
                # that is under-full should attract water. Adds a source to the RHS so the
                # corrected flux has net inflow proportional to the fill deficit. MASKED OFF
                # at the free surface (air above) so a legitimately-partial surface cell is
                # never force-filled -> preserves the rest state / null invariant.
                if self.is_liquid(x, y + 1):
                    d += self.STIFF * (FILL_CAP - self.fill[y][x])
                div[y][x] = d

        # Gauss-Seidel SOR, warm-started from last tick's field (P changes slowly, so a
        # moderate iteration count stays converged). Warm-start is a legitimate optimisation
        # as long as the solve still converges to the current-state solution each tick.
        P = self.P
        for y in range(H):                      # drop any stale P in non-liquid cells
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
                            k += 1                 # free surface: Dirichlet P=0
                        # solid / off-grid: Neumann -> face dropped
                    if k == 0:
                        continue
                    target = (s - div[y][x]) / k   # laplacian(P)=div => P_p=(sum_nb - div)/k
                    P[y][x] += self.OMEGA * (target - P[y][x])
        self.P = P

    def _p(self, x, y):
        # pressure sample with BCs: liquid = solved P, air (free surface) = 0.
        if self.is_liquid(x, y):
            return self.P[y][x]
        return 0.0

    # ---- corrected face flux  F = u* - grad(P) ----
    def correct_flux(self):
        W, H = self.W, self.H
        Fx = [[0.0] * W for _ in range(H)]
        Fy = [[0.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                # +X face between (x,y),(x+1,y)
                if not self.is_solid(x, y) and not self.is_solid(x + 1, y) \
                        and (self.is_liquid(x, y) or self.is_liquid(x + 1, y)):
                    Fx[y][x] = 0.0 - (self._p(x + 1, y) - self._p(x, y))
                # +Y face between (x,y),(x,y+1)
                if not self.is_solid(x, y) and not self.is_solid(x, y + 1) \
                        and (self.is_liquid(x, y) or self.is_liquid(x, y + 1)):
                    Fy[y][x] = self.prov_fy(x, y) - (self._p(x, y + 1) - self._p(x, y))
        self.Fx, self.Fy = Fx, Fy

    # ---- conservative budgeted transport (unchanged core) ----
    def transport(self):
        W, H = self.W, self.H
        Rx = [[self.eff(self.Fx[y][x]) for x in range(W)] for y in range(H)]
        Ry = [[self.eff(self.Fy[y][x]) for x in range(W)] for y in range(H)]

        s_out = [[1.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                if self.is_solid(x, y):
                    s_out[y][x] = 0.0; continue
                fillc = self.fill[y][x] if self.is_liquid(x, y) else 0.0
                t = 0.0
                if Rx[y][x] > 0: t += Rx[y][x]
                if Ry[y][x] > 0: t += Ry[y][x]
                if x - 1 >= 0 and Rx[y][x - 1] < 0: t += -Rx[y][x - 1]
                if y - 1 >= 0 and Ry[y - 1][x] < 0: t += -Ry[y - 1][x]
                s_out[y][x] = min(1.0, fillc / t) if t > 0 else 1.0

        s_in = [[1.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                if self.is_solid(x, y):
                    s_in[y][x] = 0.0; continue
                fillc = self.fill[y][x] if self.is_liquid(x, y) else 0.0
                cap = FILL_CAP - fillc
                t = 0.0
                if Rx[y][x] < 0: t += (-Rx[y][x]) * s_out[y][x + 1]
                if Ry[y][x] < 0: t += (-Ry[y][x]) * s_out[y + 1][x]
                if x - 1 >= 0 and Rx[y][x - 1] > 0: t += Rx[y][x - 1] * s_out[y][x - 1]
                if y - 1 >= 0 and Ry[y - 1][x] > 0: t += Ry[y - 1][x] * s_out[y - 1][x]
                s_in[y][x] = min(1.0, cap / t) if t > 0 else 1.0

        before = [row[:] for row in self.fill]
        delta = [[0.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                if self.is_solid(x, y): continue
                d = 0.0
                v = Rx[y][x]
                if v > 0:   d -= v * s_out[y][x] * s_in[y][x + 1]
                elif v < 0: d -= v * s_out[y][x + 1] * s_in[y][x]
                v = Ry[y][x]
                if v > 0:   d -= v * s_out[y][x] * s_in[y + 1][x]
                elif v < 0: d -= v * s_out[y + 1][x] * s_in[y][x]
                if x - 1 >= 0:
                    w = Rx[y][x - 1]
                    if w > 0:   d += w * s_out[y][x - 1] * s_in[y][x]
                    elif w < 0: d += w * s_out[y][x] * s_in[y][x - 1]
                if y - 1 >= 0:
                    w = Ry[y - 1][x]
                    if w > 0:   d += w * s_out[y - 1][x] * s_in[y][x]
                    elif w < 0: d += w * s_out[y][x] * s_in[y - 1][x]
                delta[y][x] = d

        for y in range(H):
            for x in range(W):
                if self.is_solid(x, y): continue
                d = delta[y][x]
                if self.is_air(x, y):
                    if d <= 1e-9: continue
                    self.mat[y][x] = LIQUID
                    self.fill[y][x] = max(0.0, min(FILL_CAP, d))
                else:
                    nf = max(0.0, min(FILL_CAP, self.fill[y][x] + d))
                    self.fill[y][x] = nf
                    if nf <= 1e-9:
                        self.mat[y][x] = AIR; self.fill[y][x] = 0.0
        self._last_moved = sum(abs(self.fill[y][x] - before[y][x])
                               for y in range(H) for x in range(W))

    def tick(self):
        self.solve_pressure()
        self.correct_flux()
        self.transport()

    def total(self): return sum(self.fill[y][x] for y in range(self.H) for x in range(self.W))
    def carve(self, x, y, mat, fill=0.0):
        self.mat[y][x] = mat; self.fill[y][x] = fill

    def column_top(self, x):
        for y in range(self.H - 1, -1, -1):
            if self.is_liquid(x, y) and self.fill[y][x] > 1.0:
                return y + self.fill[y][x] / FILL_CAP
        return None

    def arm_mass(self, x):
        return sum(self.fill[y][x] for y in range(self.H)) / FILL_CAP

    def run(self, max_ticks=4000, settle_delta=0.4, settle_streak=25):
        quiet = 0
        for i in range(max_ticks):
            self.tick()
            if self._last_moved < settle_delta:
                quiet += 1
                if quiet >= settle_streak:
                    return i + 1
            else:
                quiet = 0
        return max_ticks


def build_utube(fill_left_to=9, arm_gap=3):
    W = 3 + arm_gap; H = 14
    s = Sim(W, H)
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


# ---- solver-specific invariants (the two new checks) ----
def test_bottom_pipe_gradient():
    """Unequal arms must produce a nonzero, correctly-directed P gradient across the bottom
    connector BEFORE any mass moves. This is the direct antidote to the flat-P max failure."""
    s, lx, rx = build_utube()
    s.solve_pressure()   # one solve on the initial (tall-left, empty-right) state
    p_left_base = s.P[1][lx]
    p_right_base = s.P[1][rx]
    print(f"[bottom-gradient]  P(left base)={p_left_base:.1f}  P(right base)={p_right_base:.1f}")
    assert p_left_base - p_right_base > 1.0, \
        f"no driving gradient across the pipe: {p_left_base:.1f} vs {p_right_base:.1f}"
    print("  -> tall arm has higher base pressure: real lateral gradient exists")


def test_pressure_null_at_rest():
    """Equal arms at rest: corrected face requests must be ~zero everywhere (not merely
    zero net mass drift)."""
    W, H = 3, 10
    s = Sim(W, H)
    for y in range(1, H - 1):
        s.carve(1, y, AIR)
    for y in range(1, 6):
        s.carve(1, y, LIQUID, FILL_CAP)     # settled full column, surface at y=5
    s.solve_pressure(); s.correct_flux()
    maxreq = max(abs(s.Fx[y][x]) for y in range(H) for x in range(W))
    maxreq = max(maxreq, max(abs(s.Fy[y][x]) for y in range(H) for x in range(W)))
    print(f"[pressure-null]  max |corrected face request| at rest = {maxreq:.3f}")
    assert maxreq < 1.0, f"nonzero corrected request at rest: {maxreq:.3f}"
    print("  -> settled column produces zero corrected flux (hydrostatic P cancels gravity)")


def test_flat_lake_noop():
    W, H = 12, 8
    s = Sim(W, H)
    for x in range(1, W - 1):
        for y in range(1, 5):
            s.carve(x, y, LIQUID, FILL_CAP)
    t0 = s.total(); snap = [row[:] for row in s.fill]; mx = 0.0
    for _ in range(150):
        s.tick(); mx = max(mx, s._last_moved)
    diff = max(abs(s.fill[y][x] - snap[y][x]) for y in range(H) for x in range(W))
    print(f"[flat-lake]  max_move/tick={mx:.3f} cell_drift={diff:.3f} cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0 and diff < 1.0 and mx < 2.0
    print("  -> flat lake stays flat")


def test_utube_equalizes():
    s, lx, rx = build_utube()
    t0 = s.total(); ticks = s.run()
    l, r = s.arm_mass(lx), s.arm_mass(rx)
    lh, rh = s.column_top(lx), s.column_top(rx)
    # strict per-arm balance + real full columns
    top = int(min(lh, rh)) if (lh and rh) else 0
    submerged = [s.fill[y][x] for x in (lx, rx) for y in range(1, max(top, 1)) if s.is_liquid(x, y)]
    med = sorted(submerged)[len(submerged) // 2] if submerged else 0.0
    print(f"[u-tube]  settled@{ticks}  left_arm={l:.2f} right_arm={r:.2f} imbalance={abs(l-r):.2f} "
          f"submerged_med={med:.0f} cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0, "mass not conserved"
    assert abs(l - r) < 1.0, f"arms not balanced: {l:.2f} vs {r:.2f}"
    assert med > 220.0, f"thin smear: submerged median {med:.0f}"
    print("  -> U-tube equalizes (balanced arms, full columns, mass conserved)")


if __name__ == "__main__":
    test_pressure_null_at_rest()
    test_bottom_pipe_gradient()
    test_flat_lake_noop()
    test_utube_equalizes()
    print("\nALL TESTS PASSED")
