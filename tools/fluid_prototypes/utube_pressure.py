"""
2D emulator v3 — BOUNDED PRESSURE FIELD (piezometric head) replacing head() in the drive.

Design under test (per the pivot):
  - P is a hydrostatic-head field, REBUILT FROM SCRATCH every tick (no cross-tick memory):
        P0(p) = local_vertical_head(p)  for liquid cells, else 0     (fresh live seeds)
        repeat N Jacobi iterations:
            P(p) = max( P0(p),
                        max over same-liquid neighbours nb of ( P(nb) - cost(nb->p) ) )
        cost(nb->p) = +FILL_CAP if p is ABOVE nb (climbing)   -> loses a cell of head
                      -FILL_CAP if p is BELOW nb (descending) -> gains a cell of head
                       0        horizontal                    -> free
    Rebuild-from-live-seeds (not persistent max) => a drained source cannot leave a
    self-sustaining stale plateau; its pressure is simply absent from next tick's seeds.
    N bounds propagation distance per step (speed), NOT the equilibrium (no horizontal decay).

  - P REPLACES head() in the Accelerate drive (single force, no additive correction):
        horizontal A->B : drive = P(A) - P(B)
        vertical   A->B : drive = P(A) - P(B) - FILL_CAP       (B directly above A)
    A genuinely supported column has drive 0; an unsupported film has no ladder so
    gravity wins. No suppression hack, no deficit term.

  - flux_buf stays the sole persistent momentum; Transport (conservative s_out/s_in) unchanged.

Coords: x horizontal, y up. Owned faces per cell: +X (to x+1), +Y (to y+1, ABOVE).
"""

AIR, LIQUID, SOLID = 0, 1, 2
FILL_CAP = 255.0


class Sim:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.mat = [[SOLID] * W for _ in range(H)]
        self.fill = [[0.0] * W for _ in range(H)]
        self.fx = [[0.0] * W for _ in range(H)]   # persistent owned +X flux
        self.fy = [[0.0] * W for _ in range(H)]   # persistent owned +Y flux (upward)
        self.P = [[0.0] * W for _ in range(H)]     # rebuilt every tick, no cross-tick memory

        self.g_dt = 0.15
        self.damp = 0.97
        self.bias = 3.0
        self.MIN_FLOW = 1.0
        self.HEAD_MAX = 40
        self.use_pressure = True   # False => fall back to local head() (reproduces the bug)
        self.N_ITERS = 60          # Jacobi iterations per pressure rebuild (propagation radius)
        self._last_moved = 0.0

    def inb(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def is_liquid(self, x, y):
        return self.inb(x, y) and self.mat[y][x] == LIQUID

    def is_air(self, x, y):
        return self.inb(x, y) and self.mat[y][x] == AIR

    def is_solid(self, x, y):
        return (not self.inb(x, y)) or self.mat[y][x] == SOLID

    # local straight-up hydrostatic head (fill + consecutive full liquid above)
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

    # ---- rebuild P from scratch: fresh seeds + N Jacobi max-with-cost iterations ----
    def rebuild_pressure(self):
        W, H = self.W, self.H
        P0 = [[(self.head(x, y) if self.mat[y][x] == LIQUID else 0.0)
               for x in range(W)] for y in range(H)]
        P = [row[:] for row in P0]
        for _ in range(self.N_ITERS):
            nxt = [row[:] for row in P0]   # re-seed floor each iteration
            changed = False
            for y in range(H):
                for x in range(W):
                    if self.mat[y][x] != LIQUID:
                        continue
                    best = nxt[y][x]
                    # neighbour below -> climbing to p: cost +FILL_CAP
                    if self.is_liquid(x, y - 1):
                        c = P[y - 1][x] - FILL_CAP
                        if c > best: best = c
                    # neighbour above -> descending to p: cost -FILL_CAP (gain head)
                    if self.is_liquid(x, y + 1):
                        c = P[y + 1][x] + FILL_CAP
                        if c > best: best = c
                    # horizontal neighbours: free
                    if self.is_liquid(x - 1, y):
                        c = P[y][x - 1]
                        if c > best: best = c
                    if self.is_liquid(x + 1, y):
                        c = P[y][x + 1]
                        if c > best: best = c
                    if best > nxt[y][x]:
                        nxt[y][x] = best
                        changed = True
            P = nxt
            if not changed:
                break
        self.P = P

    def _pfield(self, x, y):
        if self.inb(x, y) and self.mat[y][x] == LIQUID:
            return self.P[y][x]
        return 0.0   # air / solid contribute no propagated pressure

    # ---- Accelerate: integrate persistent flux from the P-gradient drive ----
    def _accel_face(self, ax, ay, bx, by, axis, old):
        if self.is_solid(ax, ay) or self.is_solid(bx, by):
            return 0.0
        a_liq = self.is_liquid(ax, ay)
        b_liq = self.is_liquid(bx, by)
        if not a_liq and not b_liq:
            return 0.0
        if self.use_pressure:
            pa, pb = self._pfield(ax, ay), self._pfield(bx, by)
        else:
            pa, pb = self.head(ax, ay), self.head(bx, by)   # baseline: local head only
        if axis == 1:
            drive = pa - pb - FILL_CAP
            accel = drive * self.g_dt * self.bias
            cap = FILL_CAP * self.bias
        else:
            drive = pa - pb
            accel = drive * self.g_dt
            cap = FILL_CAP
        return max(-cap, min(cap, old * self.damp + accel))

    def accelerate(self):
        nfx = [[0.0] * self.W for _ in range(self.H)]
        nfy = [[0.0] * self.W for _ in range(self.H)]
        for y in range(self.H):
            for x in range(self.W):
                nfx[y][x] = self._accel_face(x, y, x + 1, y, 0, self.fx[y][x])
                nfy[y][x] = self._accel_face(x, y, x, y + 1, 1, self.fy[y][x])
        self.fx, self.fy = nfx, nfy

    # ---- Transport: conservative budgeted symmetric apply (unchanged) ----
    def transport(self):
        W, H = self.W, self.H
        Rx = [[self.eff(self.fx[y][x]) for x in range(W)] for y in range(H)]
        Ry = [[self.eff(self.fy[y][x]) for x in range(W)] for y in range(H)]

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

        before = [row[:] for row in self.fill]
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
        self._last_moved = sum(abs(self.fill[y][x] - before[y][x])
                               for y in range(H) for x in range(W))

    def tick(self):
        if self.use_pressure:
            self.rebuild_pressure()
        self.accelerate()
        self.transport()

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

    def run(self, max_ticks=4000, settle_delta=0.5, settle_streak=25):
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


# ============================ TESTS ============================

def build_utube(fill_left_to=9, arm_gap=3):
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


def test_flat_lake_noop():
    W, H = 12, 8
    s = Sim(W, H)
    for x in range(1, W - 1):
        for y in range(1, 5):
            s.carve(x, y, LIQUID, FILL_CAP)
    t0 = s.total()
    snap = [row[:] for row in s.fill]
    mx = 0.0
    for _ in range(150):
        s.tick(); mx = max(mx, s._last_moved)
    diff = max(abs(s.fill[y][x] - snap[y][x]) for y in range(H) for x in range(W))
    print(f"[flat-lake]  max_move/tick={mx:.3f} cell_drift={diff:.3f} cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0 and diff < 1.0 and mx < 2.0
    print("  -> flat lake stays flat")


def test_settled_partial_quiet():
    W, H = 3, 10
    s = Sim(W, H)
    for y in range(1, H - 1):
        s.carve(1, y, AIR)
    for y in range(1, 6):
        s.carve(1, y, LIQUID, FILL_CAP)
    s.carve(1, 6, LIQUID, 128.0)
    t0 = s.total()
    snap = [row[:] for row in s.fill]
    mx = 0.0
    for _ in range(200):
        s.tick(); mx = max(mx, s._last_moved)
    diff = max(abs(s.fill[y][x] - snap[y][x]) for y in range(H) for x in range(W))
    print(f"[settled-partial]  max_move/tick={mx:.3f} cell_drift={diff:.3f} cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0 and diff < 1.0 and mx < 2.0
    print("  -> settled partial-surface column stays quiet")


def test_utube_baseline_bug():
    s, lx, rx = build_utube()
    s.use_pressure = False
    t0 = s.total(); s.run()
    lh, rh = s.column_top(lx), s.column_top(rx)
    print(f"[baseline/head-only]  left={lh} right={rh} cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0
    assert (rh is None) or (lh - rh > 2.0)
    print("  -> baseline (local head) fails to equalize: reproduces the bug")


# === STRICT metric ================================================================
# column_top() and "median submerged fill over BOTH arms" are DECEPTIVE for this
# approach: a lopsided thin smear reaches the same column HEIGHT in both arms at very
# different DENSITIES, so column_top reports "equalized" while the arms hold very
# different amounts of water. The only trustworthy equalization metric is PER-ARM MASS.
# (A reviewer running this file's old, weaker tests was fooled into thinking max-relax
# worked; see docs/BUG_LOG.md. Do not judge equalization by column_top.)
def arm_mass(s, x):
    return sum(s.fill[y][x] for y in range(s.H)) / FILL_CAP


def build_utube_empty_pipe():
    """Harder case: connecting pipe starts EMPTY (only the left arm is filled), so water
    must genuinely migrate across the pipe, not just climb off a shared full base."""
    W, H = 6, 14
    s = Sim(W, H)
    lx, rx = 1, 4
    for y in range(1, H - 1):
        s.carve(lx, y, AIR); s.carve(rx, y, AIR)
    for x in range(lx, rx + 1):
        s.carve(x, 1, AIR)                       # bottom pipe empty
    for y in range(1, 10):
        s.carve(lx, y, LIQUID, FILL_CAP)         # only the left arm holds water
    return s, lx, rx


def demo_max_relax_falsified():
    """FALSIFICATION DEMO (expected to FAIL equalization). Runs the U-tube on both the
    pre-filled-pipe and empty-pipe geometries and judges by PER-ARM MASS, not column_top.
    Returns True iff the arms actually balanced (they do not)."""
    balanced_all = True
    for label, (s, lx, rx) in (("prefilled-pipe", build_utube()),
                               ("empty-pipe", build_utube_empty_pipe())):
        t0 = s.total(); ticks = s.run()
        l, r = arm_mass(s, lx), arm_mass(s, rx)
        lh, rh = s.column_top(lx), s.column_top(rx)
        imb = abs(l - r)
        ok = imb < 1.0
        balanced_all = balanced_all and ok
        print(f"[max-relax/{label:14s}] settled@{ticks:4d}  "
              f"column_top L/R={lh:.2f}/{rh:.2f} (LOOKS equal)  |  "
              f"ARM_MASS L/R={l:.2f}/{r:.2f} imbalance={imb:.2f}  |  cons={abs(s.total()-t0):.3f}  "
              f"=> {'BALANCED' if ok else 'LOPSIDED SMEAR'}")
    return balanced_all


def test_thin_film_falls():
    """An unsupported 1-cell-thick shelf, connected to a column but sitting over air,
    must drain down (not stay pinned merely because it is horizontally connected)."""
    W, H = 8, 10
    s = Sim(W, H)
    # floor cavity
    for x in range(1, W - 1):
        for y in range(1, H - 1):
            s.carve(x, y, AIR)
    # a full column on the left, up to y=7
    for y in range(1, 8):
        s.carve(1, y, LIQUID, FILL_CAP)
    # a 1-cell-thick shelf at y=7 reaching right over air (x=2..6)
    for x in range(2, W - 1):
        s.carve(x, 7, LIQUID, FILL_CAP)
    t0 = s.total()
    shelf_before = sum(s.fill[7][x] for x in range(2, W - 1))
    s.run()
    shelf_after = sum(s.fill[7][x] for x in range(2, W - 1))
    floor = sum(s.fill[1][x] for x in range(1, W - 1))
    print(f"[thin-film]  shelf {shelf_before:.0f}->{shelf_after:.0f} floor_now={floor:.0f} cons={abs(s.total()-t0):.3f}")
    assert abs(s.total() - t0) < 1.0
    assert shelf_after < shelf_before * 0.25, f"film stayed pinned: {shelf_after:.0f}"
    print("  -> unsupported film drains and pools (not pinned by connectivity)")


def test_drain_no_stale_plateau():
    """Rebuild-from-live-seeds invariant: delete part of a converged body and confirm P
    carries no stale pressure there on the very next rebuild."""
    W, H = 3, 12
    s = Sim(W, H)
    for y in range(1, 11):
        s.carve(1, y, LIQUID, FILL_CAP)
    s.rebuild_pressure()
    p_deep_before = s.P[1][1]
    # remove the top 5 cells (source disappears)
    for y in range(6, 11):
        s.carve(1, y, AIR)
    s.rebuild_pressure()
    p_removed = max(s.P[y][1] for y in range(6, 11))
    p_deep_after = s.P[1][1]
    print(f"[drain]  P_bottom {p_deep_before:.0f}->{p_deep_after:.0f}  P_at_removed_cells={p_removed:.0f}")
    assert p_removed == 0.0, f"stale pressure survived at removed cells: {p_removed}"
    # bottom pressure must reflect the NEW (shorter) column, not the old one
    assert p_deep_after < p_deep_before - FILL_CAP, "bottom pressure did not drop with source"
    print("  -> drained region carries zero stale pressure; field reflects current reality")


def test_pipe_length_same_level():
    """Final level must not depend on pipe length (no horizontal decay)."""
    res = {}
    for gap in (3, 8):
        s, lx, rx = build_utube(arm_gap=gap)
        s.N_ITERS = 80
        s.run()
        res[gap] = (s.column_top(lx) + s.column_top(rx)) / 2.0
    print(f"[pipe-length]  short_gap_level={res[3]:.2f}  long_gap_level={res[8]:.2f}")
    assert abs(res[3] - res[8]) <= 1.0, f"level depends on pipe length: {res}"
    print("  -> short and long U-tubes reach the same level")


if __name__ == "__main__":
    print("=" * 78)
    print("FALSIFIED PROTOTYPE — approach #3: P replaces head() via max-with-cost relaxation.")
    print("Some invariants hold (rest, drain, thin-film); U-TUBE EQUALIZATION DOES NOT.")
    print("Judge equalization by PER-ARM MASS, never column_top. See docs/BUG_LOG.md.")
    print("=" * 78)
    # Invariants this approach legitimately satisfies:
    test_flat_lake_noop()
    test_settled_partial_quiet()
    test_drain_no_stale_plateau()
    test_thin_film_falls()
    test_utube_baseline_bug()
    # The falsification: U-tube equalization (strict per-arm-mass metric).
    print("\n--- U-tube equalization (the falsifier) ---")
    balanced = demo_max_relax_falsified()
    print()
    if balanced:
        raise SystemExit("UNEXPECTED: max-relaxation balanced the arms. If real, revisit "
                         "docs/BUG_LOG.md — the rejection of approach #3 would need updating.")
    print("RESULT: FALSIFIED AS DOCUMENTED — max-relaxation reaches a lopsided thin smear "
          "(arm-mass imbalance ~2), not equalization. The uniform-at-elevation pressure "
          "field has no lateral gradient to rebalance the arms. Use utube_projection.py.")
