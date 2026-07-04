"""
2D emulator for the surface-elevation-field + non-persistent conservative static
correction, to validate the U-tube logic BEFORE touching fluid_flux.comp.

Faithful to the shader pipeline:
  - persistent dynamic flux_buf (Accelerate): head-driven, elevation-corrected on Y
  - surface-elevation field L (no decay, max-propagation, seed = y + fill/255)
  - non-persistent static Y-transfer, gated on (L(src) - y(dst) >= 1) AND local
    vertical drive near equilibrium AND dst air/under-filled
  - q_requested = q_dynamic + q_static, budgeted through the SAME s_out/s_in so
    every transfer is symmetric => mass conserved by construction.

Coords: x = horizontal, y = vertical, y increasing = UP. Gravity pulls to -Y.
Owned faces per cell: +X (to x+1) and +Y (to y+1, the cell ABOVE).
Positive +Y flux = upward transfer (lower cell -> upper cell).
"""

import math

AIR, LIQUID, SOLID = 0, 1, 2
FILL_CAP = 255.0
NEG_INF = float("-inf")


class Sim:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.mat = [[SOLID] * W for _ in range(H)]
        self.fill = [[0.0] * W for _ in range(H)]
        self.fx = [[0.0] * W for _ in range(H)]   # persistent owned +X flux
        self.fy = [[0.0] * W for _ in range(H)]   # persistent owned +Y flux (upward)
        self.L = [[NEG_INF] * W for _ in range(H)]  # persistent surface-elevation field

        # tuning
        self.g_dt = 0.15
        self.damp = 0.97
        self.bias = 3.0
        self.MIN_FLOW = 1.0
        self.HEAD_MAX = 24
        # static-correction tuning
        self.static_on = True
        self.STATIC_RATE = 0.2   # fraction of deficit-difference moved per tick (diffusion; <0.5 for stability)
        self.STATIC_CAP = 255.0
        self.STATIC_DEAD = 2.0   # ignore deficit differences smaller than this (kills churn)

    # ---- topology helpers ----
    def inb(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def is_liquid(self, x, y):
        return self.inb(x, y) and self.mat[y][x] == LIQUID

    def is_air(self, x, y):
        return self.inb(x, y) and self.mat[y][x] == AIR

    def is_solid(self, x, y):
        return (not self.inb(x, y)) or self.mat[y][x] == SOLID

    # ---- hydrostatic pressure head (straight-up column walk, matches shader) ----
    def head(self, x, y):
        if not self.is_liquid(x, y):
            return 0.0
        f = self.fill[y][x]
        total = f
        if f < FILL_CAP:
            return total
        yy = y
        for _ in range(self.HEAD_MAX):
            yy += 1
            if not self.is_liquid(x, yy):
                break
            af = self.fill[yy][x]
            total += af
            if af < FILL_CAP:
                break
        return total

    def eff(self, v):
        return v if abs(v) >= self.MIN_FLOW else 0.0

    # ---- Accelerate: integrate persistent dynamic flux ----
    def _accel_face(self, ax, ay, bx, by, axis, old):
        if self.is_solid(ax, ay) or self.is_solid(bx, by):
            return 0.0
        a_liq = self.is_liquid(ax, ay)
        b_liq = self.is_liquid(bx, by)
        if not a_liq and not b_liq:
            return 0.0  # both air: nothing to move
        dh = self.head(ax, ay) - self.head(bx, by)
        if axis == 1:
            # Pull-back suppression: where the UPPER cell is submerged below the connected
            # surface, its water is held up by connected pressure that straight-up head()
            # can't see. Let the static correction own vertical motion there; zero the
            # dynamic drive so it can't spuriously "collapse" a not-yet-full submerged
            # column. Unsubmerged faces (real free surfaces, falling water) are untouched.
            if self.connected_surface(bx, by) >= by + 1.0:
                accel = 0.0
                nf = old * self.damp
                return nf
            accel = (dh - FILL_CAP) * self.g_dt * self.bias
            cap = FILL_CAP * self.bias
        else:
            accel = dh * self.g_dt
            cap = FILL_CAP
        nf = old * self.damp + accel
        return max(-cap, min(cap, nf))

    def accelerate(self):
        nfx = [[0.0] * self.W for _ in range(self.H)]
        nfy = [[0.0] * self.W for _ in range(self.H)]
        for y in range(self.H):
            for x in range(self.W):
                nfx[y][x] = self._accel_face(x, y, x + 1, y, 0, self.fx[y][x])
                nfy[y][x] = self._accel_face(x, y, x, y + 1, 1, self.fy[y][x])
        self.fx, self.fy = nfx, nfy

    # ---- L: surface-elevation field, no decay, 1 hop/tick max-propagation ----
    def relax_L(self):
        nL = [[NEG_INF] * self.W for _ in range(self.H)]
        for y in range(self.H):
            for x in range(self.W):
                if not self.is_liquid(x, y):
                    continue
                best = NEG_INF
                # free surface (air directly above, or open top) seeds its own elevation
                if (not self.inb(x, y + 1)) or self.is_air(x, y + 1):
                    best = y + self.fill[y][x] / FILL_CAP
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if self.is_liquid(nx, ny) and self.L[ny][nx] > best:
                        best = self.L[ny][nx]
                nL[y][x] = best
        self.L = nL

    # ---- deficit-driven all-axis static correction ----
    # connected_surface: the surface elevation of the connected body a cell belongs to.
    # For liquid cells it's the relaxed field L; for air cells (candidate fill sites) it's
    # the max L of liquid neighbours (so an empty cell inside the body knows the true level).
    def connected_surface(self, x, y):
        if self.is_liquid(x, y):
            return self.L[y][x]
        if self.is_air(x, y):
            best = NEG_INF
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if self.is_liquid(nx, ny) and self.L[ny][nx] > best:
                    best = self.L[ny][nx]
            return best
        return NEG_INF

    # target_fill: what this cell's fill "should" be given the connected surface.
    #   fully below surface  -> 255 (submerged)
    #   the surface cell     -> partial, (Ls - y)*255
    #   above surface / none -> 0
    def target_fill(self, x, y):
        Ls = self.connected_surface(x, y)
        if Ls == NEG_INF:
            return 0.0
        if Ls >= y + 1.0:
            return FILL_CAP
        if Ls > y:
            return (Ls - y) * FILL_CAP
        return 0.0

    def deficit(self, x, y):
        # Unknown connected surface (L not yet propagated here) => no opinion, no action.
        # Without this, a cell reads target 0 ("should be empty") before L arrives and
        # dumps its fill into neighbouring air on the first few ticks (startup churn).
        if self.connected_surface(x, y) == NEG_INF:
            return 0.0
        cur = self.fill[y][x] if self.is_liquid(x, y) else 0.0
        return self.target_fill(x, y) - cur

    # Static request on an owned face A->B (B = A + axis dir). Positive = A donates to B.
    # Drives water from surplus (low deficit) toward deficit (high deficit); zero at target.
    def q_static_face(self, ax, ay, bx, by):
        if not self.static_on:
            return 0.0
        if self.is_solid(ax, ay) or self.is_solid(bx, by):
            return 0.0
        if not (self.is_liquid(ax, ay) or self.is_liquid(bx, by)):
            return 0.0
        diff = self.deficit(bx, by) - self.deficit(ax, ay)
        if abs(diff) < self.STATIC_DEAD:      # deadzone: no churn on settled bodies
            return 0.0
        q = max(-self.STATIC_CAP, min(self.STATIC_CAP, diff * self.STATIC_RATE))
        # Air-materialisation guard: never push into an air cell that isn't a FULL cell
        # below the connected surface. Prevents leaking above the free surface, which
        # would create a spurious higher surface and ratchet L upward (runaway).
        if q > 0 and self.is_air(bx, by) and self.connected_surface(bx, by) < by + 1.0:
            return 0.0
        if q < 0 and self.is_air(ax, ay) and self.connected_surface(ax, ay) < ay + 1.0:
            return 0.0
        return q

    # ---- Transport: budgeted symmetric apply of (dynamic + static) ----
    def transport(self):
        W, H = self.W, self.H
        Rx = [[0.0] * W for _ in range(H)]
        Ry = [[0.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                Rx[y][x] = self.eff(self.fx[y][x]) + self.q_static_face(x, y, x + 1, y)
                Ry[y][x] = self.eff(self.fy[y][x]) + self.q_static_face(x, y, x, y + 1)

        s_out = [[1.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                if self.is_solid(x, y):
                    s_out[y][x] = 0.0
                    continue
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
                    s_in[y][x] = 0.0
                    continue
                fillc = self.fill[y][x] if self.is_liquid(x, y) else 0.0
                cap = FILL_CAP - fillc
                t = 0.0
                if Rx[y][x] < 0: t += (-Rx[y][x]) * s_out[y][x + 1]
                if Ry[y][x] < 0: t += (-Ry[y][x]) * s_out[y + 1][x]
                if x - 1 >= 0 and Rx[y][x - 1] > 0: t += Rx[y][x - 1] * s_out[y][x - 1]
                if y - 1 >= 0 and Ry[y - 1][x] > 0: t += Ry[y - 1][x] * s_out[y - 1][x]
                s_in[y][x] = min(1.0, cap / t) if t > 0 else 1.0

        delta = [[0.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                if self.is_solid(x, y):
                    continue
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
                if self.is_solid(x, y):
                    continue
                d = delta[y][x]
                if self.is_air(x, y):
                    if d <= 1e-9:
                        continue
                    self.mat[y][x] = LIQUID
                    self.fill[y][x] = max(0.0, min(FILL_CAP, d))
                else:
                    nf = max(0.0, min(FILL_CAP, self.fill[y][x] + d))
                    self.fill[y][x] = nf
                    if nf <= 1e-9:
                        self.mat[y][x] = AIR
                        self.fill[y][x] = 0.0

    def tick(self):
        self.relax_L()
        self.accelerate()
        self.transport()

    # ---- diagnostics ----
    def total(self):
        return sum(self.fill[y][x] for y in range(self.H) for x in range(self.W))

    def column_top(self, x):
        for y in range(self.H - 1, -1, -1):
            if self.is_liquid(x, y) and self.fill[y][x] > 1.0:
                return y + self.fill[y][x] / FILL_CAP
        return None

    def carve(self, x, y, mat, fill=0.0):
        self.mat[y][x] = mat
        self.fill[y][x] = fill

    def run(self, max_ticks=4000, settle_delta=0.5, settle_streak=30):
        prev = self.total()
        quiet = 0
        for i in range(max_ticks):
            self.tick()
            cur = self.total()
            # settling gauge: how much fill moved this tick
            moved = self._last_moved
            if moved < settle_delta:
                quiet += 1
                if quiet >= settle_streak:
                    return i + 1
            else:
                quiet = 0
            prev = cur
        return max_ticks

    # track movement for settle detection
    _last_moved = 0.0


# instrument transport to record total absolute movement (for settle detection)
_orig_transport = Sim.transport
def _transport_tracked(self):
    before = [row[:] for row in self.fill]
    _orig_transport(self)
    self._last_moved = sum(
        abs(self.fill[y][x] - before[y][x])
        for y in range(self.H) for x in range(self.W)
    )
Sim.transport = _transport_tracked


# ============================ TESTS ============================

def build_utube(fill_left_to=9):
    """U-shaped cavity in rock. Left & right vertical shafts joined by a bottom row."""
    W, H = 7, 14
    s = Sim(W, H)
    lx, rx = 1, 5
    for y in range(1, H - 1):
        s.carve(lx, y, AIR)
        s.carve(rx, y, AIR)
    for x in range(lx, rx + 1):
        s.carve(x, 1, AIR)           # bottom connecting channel
    # water: fill left shaft + bottom channel
    for y in range(1, fill_left_to + 1):
        s.carve(lx, y, LIQUID, FILL_CAP)
    for x in range(lx, rx + 1):
        s.carve(x, 1, LIQUID, FILL_CAP)
    return s, lx, rx


def test_utube_baseline_reproduces_bug():
    s, lx, rx = build_utube()
    s.static_on = False
    t0 = s.total()
    s.run()
    lh, rh = s.column_top(lx), s.column_top(rx)
    dc = abs(s.total() - t0)
    print(f"[baseline/static-off] left={lh} right={rh} conservation_drift={dc:.3f}")
    assert dc < 1.0, f"conservation broken: drift={dc}"
    # THE BUG: right arm cannot climb to match the left.
    assert (rh is None) or (lh - rh > 2.0), \
        f"expected baseline to FAIL to equalize, but left={lh} right={rh}"
    print("  -> baseline correctly fails to equalize (reproduces the U-tube bug)")


def test_utube_static_equalizes():
    s, lx, rx = build_utube()
    s.static_on = True
    t0 = s.total()
    ticks = s.run()
    lh, rh = s.column_top(lx), s.column_top(rx)
    dc = abs(s.total() - t0)
    print(f"[static-on] settled@{ticks} left={lh} right={rh} conservation_drift={dc:.3f}")
    assert dc < 1.0, f"conservation broken: drift={dc}"
    assert rh is not None, "right arm never received water"
    assert abs(lh - rh) <= 1.0, f"arms did not equalize: left={lh} right={rh}"
    # Physical-reality check: the water must form real FULL columns to the volume-implied
    # level, not thin partial films smeared up to a stale surface. ~13 cells of water in a
    # cavity that is 5 wide at the bottom row + 2 wide above => equilibrium level ~5.
    submerged = [s.fill[y][x] for x in (lx, rx) for y in range(1, int(min(lh, rh)) - 1)
                 if s.is_liquid(x, y)]
    med = sorted(submerged)[len(submerged) // 2] if submerged else 0.0
    assert med > 220.0, f"THIN SMEAR: submerged cells median fill {med:.0f}, expected ~255"
    print("  -> U-tube equalizes to matching level with real full columns, mass conserved")


def test_settled_partial_surface_stays_quiet():
    """Churn regression: a full column with a partial top cell must stay dead quiet."""
    W, H = 3, 10
    s = Sim(W, H)
    for y in range(1, H - 1):
        s.carve(1, y, AIR)
    for y in range(1, 6):
        s.carve(1, y, LIQUID, FILL_CAP)
    s.carve(1, 6, LIQUID, 128.0)   # partial surface cell
    t0 = s.total()
    snapshot = [row[:] for row in s.fill]
    max_move = 0.0
    for _ in range(300):
        s.tick()
        max_move = max(max_move, s._last_moved)
    drift = abs(s.total() - t0)
    diff = max(abs(s.fill[y][x] - snapshot[y][x])
               for y in range(H) for x in range(W))
    print(f"[settled-partial] max_move/tick={max_move:.4f} max_cell_drift={diff:.4f} "
          f"conservation_drift={drift:.4f}")
    assert drift < 1.0, f"conservation broken: {drift}"
    assert diff < 1.0, f"CHURN: settled column moved by {diff}"
    assert max_move < 2.0, f"CHURN: settled column had per-tick movement of {max_move}"
    print("  -> settled partial-surface column stays quiet (no churn, no off-by-one)")


def test_flat_lake_noop():
    W, H = 12, 8
    s = Sim(W, H)
    for x in range(1, W - 1):
        for y in range(1, 5):
            s.carve(x, y, LIQUID, FILL_CAP)
        s.carve(x, 5, AIR)  # air above
    for y in range(5, H):
        for x in range(1, W - 1):
            s.carve(x, y, AIR)
    t0 = s.total()
    snapshot = [row[:] for row in s.fill]
    max_move = 0.0
    for _ in range(200):
        s.tick()
        max_move = max(max_move, s._last_moved)
    drift = abs(s.total() - t0)
    diff = max(abs(s.fill[y][x] - snapshot[y][x])
               for y in range(H) for x in range(W))
    print(f"[flat-lake] max_move/tick={max_move:.4f} max_cell_drift={diff:.4f} "
          f"conservation_drift={drift:.4f}")
    assert drift < 1.0, f"conservation broken: {drift}"
    assert diff < 1.0, f"flat lake disturbed by {diff}"
    assert max_move < 2.0, f"flat lake had per-tick movement of {max_move}"
    print("  -> flat lake stays flat (no-op)")


if __name__ == "__main__":
    test_flat_lake_noop()
    test_settled_partial_surface_stays_quiet()
    test_utube_baseline_reproduces_bug()
    test_utube_static_equalizes()
    print("\nALL TESTS PASSED")
