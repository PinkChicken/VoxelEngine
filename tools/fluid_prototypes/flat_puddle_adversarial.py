"""Adversarial stress tests for the flat-puddle surface-leveling fix.
Targets the places an averaging, one-directional RHS term is most likely wrong:
 - linear ramps (Laplacian of a linear function is 0)
 - the one-directional `avg > fill` guard (over-full cells never pushed down)
 - the MIN_FLOW=0.01 confounder riding along with the fix
 - merging across a dry gap / spreading into dry cells on a flat floor
 - checkerboard oscillation
 - wide-layer propagation
"""
import importlib.util, pathlib
_here = pathlib.Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
base = _load("utube_projection")
fp = _load("flat_puddle_test")
AIR, LIQUID, SOLID, FC = base.AIR, base.LIQUID, base.SOLID, base.FILL_CAP
SimFix = fp.SimFlatPuddle


class SimSymmetric(SimFix):
    """Same as the fix but the horizontal term is SYMMETRIC (also pushes over-full
    surface cells down), to see if the one-directional guard is what plateaus."""
    def solve_pressure(self):
        W, H = self.W, self.H
        div = [[0.0] * W for _ in range(H)]
        for y in range(H):
            for x in range(W):
                if self.mat[y][x] != LIQUID:
                    continue
                d = self.prov_fy(x, y) - self.prov_fy(x, y - 1)
                if self.is_liquid(x, y + 1):
                    d += self.STIFF * (FC - self.fill[y][x])
                if not self.is_liquid(x, y + 1):
                    nb = [self.fill[y][nx] for nx in (x + 1, x - 1) if self.is_liquid(nx, y)]
                    if nb:
                        d += self.STIFF_H * (sum(nb) / len(nb) - self.fill[y][x])  # BOTH signs
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
                    s = 0.0; k = 0
                    for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                        if self.is_liquid(nx, ny): s += P[ny][nx]; k += 1
                        elif self.is_air(nx, ny): k += 1
                    if k == 0: continue
                    P[y][x] += self.OMEGA * ((s - div[y][x]) / k - P[y][x])
        self.P = P


def flat_row(cls, fills, min_flow=None):
    # min_flow=None -> leave the class's own shipped default untouched (SimFlatPuddle now
    # ships MIN_FLOW=0.1; see flat_puddle_test.py). Do NOT default this to 0.01 here: that is
    # exactly the deadzone the handoff doc rejects (settled-U-tube churn, §3c), and a stray
    # low default previously made sections [1]/[2]/[6]/[7] silently benchmark a config that
    # was never shipped, making the width-dependent residual look smaller than it really is.
    W = len(fills) + 2; H = 4
    s = cls(W, H)
    if min_flow is not None:
        s.MIN_FLOW = min_flow
    for i, f in enumerate(fills):
        if f is None:  # air gap
            s.carve(i + 1, 1, AIR)
        else:
            s.carve(i + 1, 1, LIQUID, float(f))
    for x in range(W):
        if s.mat[2][x] != LIQUID:
            s.mat[2][x] = AIR
    return s


def run_row(s, ticks=20000):
    t0 = s.total()
    s.run(max_ticks=ticks, settle_delta=0.02, settle_streak=300)
    fills = [s.fill[1][x] for x in range(s.W) if s.is_liquid(x, 1)]
    return fills, (max(fills) - min(fills) if fills else 0), abs(s.total() - t0)


print("=" * 78)
print("ADVERSARIAL: flat-puddle fix")
print("=" * 78)

print("\n[1] LINEAR RAMP — should flatten to level; Laplacian term is 0 on a ramp")
for name, cls in (("fix (one-dir)", SimFix), ("symmetric", SimSymmetric)):
    s = flat_row(cls, [40, 80, 120, 160, 200])
    f, spread, cons = run_row(s)
    print(f"   {name:14s} -> {[int(v) for v in f]}  spread={spread:.1f}  cons={cons:.3f}")

print("\n[2] WIDER puddle (agent's own case) run to 40k ticks — plateau or converge?")
for name, cls in (("fix (one-dir)", SimFix), ("symmetric", SimSymmetric)):
    s = flat_row(cls, [200, 200, 50, 200, 200, 200, 100, 200, 200])
    f, spread, cons = run_row(s, ticks=40000)
    print(f"   {name:14s} -> spread={spread:.1f} (mean {sum(f)/len(f):.0f})  cons={cons:.3f}")

print("\n[3] MIN_FLOW confounder — does the fix work with the ORIGINAL deadzone 0.5?")
for mf in (0.01, 0.5):
    s = flat_row(SimFix, [62, 62, 11, 62, 62], min_flow=mf)
    f, spread, cons = run_row(s)
    print(f"   MIN_FLOW={mf:<4} -> {[int(v) for v in f]}  spread={spread:.1f}")

print("\n[4] DRY GAP — water split by an air cell on a flat floor; should it merge?")
s = flat_row(SimFix, [124, 124, None, 124, 124])
f, spread, cons = run_row(s)
mid = s.fill[1][3]  # the gap cell
print(f"   final wet={[int(s.fill[1][x]) for x in range(1,6)]}  gap-cell-fill={mid:.0f}  cons={cons:.3f}")

print("\n[5] SPREAD INTO DRY — full block next to dry flat floor; should spread & level")
s = flat_row(SimFix, [255, 255, 255, None, None, None])
f, spread, cons = run_row(s)
print(f"   final={[('.' if s.mat[1][x]==AIR else int(s.fill[1][x])) for x in range(1,7)]}  cons={cons:.3f}")

print("\n[6] CHECKERBOARD — [250,10,250,10,250] single layer; oscillate or level?")
s = flat_row(SimFix, [250, 10, 250, 10, 250])
maxmove = 0.0
t0 = s.total()
for _ in range(3000):
    s.tick(); maxmove = max(maxmove, s._last_moved)
f = [s.fill[1][x] for x in range(1, 6)]
print(f"   final={[int(v) for v in f]}  spread={max(f)-min(f):.1f}  max_move/tick={maxmove:.2f}  cons={abs(s.total()-t0):.3f}")

print("\n[7] WIDE layer (25 cells) single dip — does leveling propagate across?")
fills = [200] * 25; fills[12] = 0
s = flat_row(SimFix, fills)
f, spread, cons = run_row(s, ticks=60000)
print(f"   spread={spread:.1f} (mean {sum(f)/len(f):.0f})  cons={cons:.3f}")
