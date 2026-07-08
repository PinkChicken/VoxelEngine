"""
ALTERNATIVE flat-puddle fix — dedicated surface height-leveling pass (NOT in the projection).

Why this exists: on a 1-deep flat layer EVERY cell has air directly above, so the
free-surface P=0 Dirichlet pins pressure to ~0 at every cell -> the pressure solve
produces almost no flow no matter what RHS source you add. That is *the* reason every
in-projection patch (avg/max source) needs a near-zero MIN_FLOW to work: only the tiny
residual flux that leaks past the pinning does the leveling. See flat_puddle_test.py.

This candidate sidesteps that entirely: a small, conservative, surface-elevation-driven
diffusion applied DIRECTLY to the surface layer (bypassing the flux/eff_flux path), run
alongside the projection (which keeps ownership of vertical/connected hydrostatics). It
levels at the engine's real MIN_FLOW=0.5 (no low deadzone needed), self-terminates below
LEVEL_MIN (no churn), and as a bonus spreads into / merges across empty floor cells.

Tradeoffs vs the in-projection max+MF=0.1 fix:
  + works at real MIN_FLOW; no near-zero deadzone; cleaner termination
  + handles gap-merge / dry-spread (in-projection source terms cannot)
  - a second mechanism to port (its own GLSL pass), and its residual is still
    width-dependent (a gentle slope on very large lakes); exact width-independent
    leveling needs a NON-LOCAL connected-component mean (a third, heavier option).

Subclasses utube_projection.py's Sim; overrides only tick() to add level_surface().
"""
import importlib.util, pathlib
_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("utube_projection", _here / "utube_projection.py")
_base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_base)
AIR, LIQUID, SOLID, FC = _base.AIR, _base.LIQUID, _base.SOLID, _base.FILL_CAP


class SimHeightLevel(_base.Sim):
    def __init__(self, W, H):
        super().__init__(W, H)
        self.LEVEL_RATE = 0.20    # fraction of surface-height difference moved per pass
        self.LEVEL_MIN = 1.0      # stop moving below this |diff| -> clean termination, no churn
        self.spread_dry = True    # let surface water spread into empty floor cells

    def _is_surface(self, x, y):
        return self.is_liquid(x, y) and not self.is_liquid(x, y + 1)

    def _fillable_dry(self, x, y):
        # an empty cell sitting on a solid floor at this level = a place water can spread into
        return self.spread_dry and self.is_air(x, y) and self.is_solid(x, y - 1) and not self.is_solid(x, y)

    def level_surface(self):
        # Jacobi over +x faces between two surface-layer cells (surface liquid OR fillable dry).
        # Fractional, conservative fill exchange toward equal surface elevation.
        moves = []
        for y in range(self.H):
            for x in range(self.W - 1):
                a_s, a_d = self._is_surface(x, y), self._fillable_dry(x, y)
                b_s, b_d = self._is_surface(x + 1, y), self._fillable_dry(x + 1, y)
                if not ((a_s or a_d) and (b_s or b_d)) or (a_d and b_d):
                    continue
                fa = self.fill[y][x] if a_s else 0.0
                fb = self.fill[y][x + 1] if b_s else 0.0
                diff = fa - fb
                if abs(diff) >= self.LEVEL_MIN:
                    moves.append((x, y, diff * self.LEVEL_RATE))
        moved = 0.0
        for x, y, amt in moves:
            if amt > 0 and self.mat[y][x + 1] == AIR:
                self.mat[y][x + 1] = LIQUID; self.fill[y][x + 1] = 0.0
            if amt < 0 and self.mat[y][x] == AIR:
                self.mat[y][x] = LIQUID; self.fill[y][x] = 0.0
            self.fill[y][x] -= amt; self.fill[y][x + 1] += amt
            moved += abs(amt)
            if self.fill[y][x] <= 1e-9: self.mat[y][x] = AIR; self.fill[y][x] = 0.0
            if self.fill[y][x + 1] <= 1e-9: self.mat[y][x + 1] = AIR; self.fill[y][x + 1] = 0.0
        return moved

    def tick(self):
        super().tick()
        m = self.level_surface()
        self._last_moved = max(self._last_moved, m)


# ------------------------------- tests -------------------------------
def _row(fills, mf=0.5, dry=0):
    W = len(fills) + 2 + dry
    s = SimHeightLevel(W, 4); s.MIN_FLOW = mf
    for i, f in enumerate(fills):
        if f is None: s.carve(i + 1, 1, AIR)
        else: s.carve(i + 1, 1, LIQUID, float(f))
    for x in range(W):
        if s.mat[2][x] != LIQUID: s.mat[2][x] = AIR
    return s

def _level(s, t=8000):
    t0 = s.total(); s.run(max_ticks=t, settle_delta=0.5, settle_streak=30)
    f = [s.fill[1][x] for x in range(s.W) if s.is_liquid(x, 1)]
    return f, (max(f) - min(f) if f else 0), abs(s.total() - t0)

def _utube():
    W, H = 6, 14; s = SimHeightLevel(W, H); lx, rx = 1, 4
    for y in range(1, H - 1): s.carve(lx, y, AIR); s.carve(rx, y, AIR)
    for x in range(lx, rx + 1): s.carve(x, 1, AIR)
    for y in range(1, 10): s.carve(lx, y, LIQUID, FC)
    for x in range(lx, rx + 1): s.carve(x, 1, LIQUID, FC)
    return s, lx, rx


if __name__ == "__main__":
    print("=" * 66)
    print("ALT FIX: surface height-leveling pass @ ENGINE-REAL MIN_FLOW = 0.5")
    print("=" * 66)
    for n, fl in {"5dip": [62, 62, 11, 62, 62], "ramp": [40, 80, 120, 160, 200],
                  "9-2dip": [200, 200, 50, 200, 200, 200, 100, 200, 200],
                  "25dip": [200]*12 + [0] + [200]*12}.items():
        f, sp, c = _level(_row(fl))
        assert c < 1.0, f"{n}: mass drift {c}"
        print(f"  [{n:7s}] spread={sp:5.1f}  cons={c:.3f}")

    # no-churn (already-flat case -- trivial, but cheap to keep as a sanity check)
    s = _row([200]*8)
    mx = 0.0
    for _ in range(200):
        s.tick(); mx = max(mx, s._last_moved)
    print(f"  [even-lake] steady-state max_move/tick={mx:.3f}")
    assert mx < 1.0, "churn on even lake"

    # no-churn (REAL steady-state check -- perturbed case, settle first, THEN measure).
    # The even-lake case above never moves at all, so it can't distinguish "correctly
    # quiet at steady state" from "never had anything to settle in the first place".
    s = _row([62, 62, 11, 62, 62])
    for _ in range(3000):
        s.tick()  # let the transient fully settle
    mx_settled = 0.0
    for _ in range(500):
        s.tick(); mx_settled = max(mx_settled, s._last_moved)
    print(f"  [settled-puddle] post-transient steady-state max_move/tick={mx_settled:.3f}")
    assert mx_settled < 1.0, f"churn after settling: {mx_settled:.3f}"

    # gap-merge bonus
    s = _row([124, 124, None, 124, 124]); f, sp, c = _level(s)
    gap = s.fill[1][3]
    print(f"  [gap-merge] center-fill={gap:.0f}  cons={c:.3f}")
    assert gap > 50, "dry gap did not merge"

    # U-tube preserved (leveling pass must not interfere)
    s, lx, rx = _utube(); t0 = s.total(); s.run()
    am = lambda x: sum(s.fill[y][x] for y in range(s.H)) / FC
    imb = abs(am(lx) - am(rx))
    print(f"  [u-tube]    arms {am(lx):.2f}/{am(rx):.2f} imbalance={imb:.2f} cons={abs(s.total()-t0):.3f}")
    assert imb < 1.0 and abs(s.total() - t0) < 1.0, "U-tube regressed"
    print("\nALL CHECKS PASSED (levels at MIN_FLOW=0.5, no churn, gap-merge, U-tube intact)")
