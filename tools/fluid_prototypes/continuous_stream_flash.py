"""Bench investigation: in-engine, with use_surface_leveling OFF (LevelSurface fully disabled),
the user still saw water "flashing"/"sucking back into itself" while a stream settles onto a
puddle -- confirmed WORSE without LevelSurface, so this is a baseline projection bug, unrelated
to LevelSurface (see LEVELSURFACE_TRANSIENT_HANDOFF.md's resolution banner).

Hypothesis: unlike every prior bench (a one-time drop, then let it settle), a real falling stream
adds fill to the top of a column EVERY TICK while it's still falling. Each tick's BuildRHS/STIFF
divergence source is computed from whatever partial-fill snapshot exists that instant -- if the
stream's landing cell fluctuates in fill tick-to-tick (natural, since it's mid-transport), the
solved P and corrected flux could be oscillating rather than converging, i.e. genuine instability
under a continuously-changing RHS, not just slow convergence. This bench uses ONLY the base
Sim (no LevelSurface at all) and simulates a continuous stream (re-adds fill to the top of the
landing column every tick, mimicking new water arriving) to see if fill/flux at the landing site
oscillate without damping.
"""
import importlib.util, pathlib
_here = pathlib.Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
base = _load("utube_projection")
AIR, LIQUID, SOLID, FC = base.AIR, base.LIQUID, base.SOLID, base.FILL_CAP


def build_puddle_with_headspace(W=30, H=20):
    s = base.Sim(W, H)
    for x in range(1, W - 1):
        for y in range(1, 15):
            s.carve(x, y, AIR)
    for x in range(1, W - 1):
        s.carve(x, 1, LIQUID, 60.0)
    return s


def run_continuous_stream(stream_fill_per_tick=200.0, source_y=10, ticks=200):
    """Every tick, before solving, top up the source column's TOP-most air cell to a fixed fill
    -- mimicking a continuous stream depositing water from above, rather than a one-time drop."""
    s = build_puddle_with_headspace()
    cx = s.W // 2
    t0_added = 0.0
    fills_at_landing = []
    fy_at_landing = []
    for i in range(ticks):
        # inject stream: refill the source cell each tick (only if it has room), mimicking
        # a steady drip -- this is the "still falling" state a real stream is in.
        if s.mat[source_y][cx] != LIQUID:
            s.mat[source_y][cx] = LIQUID
        room = FC - s.fill[source_y][cx]
        add = min(room, stream_fill_per_tick)
        s.fill[source_y][cx] += add
        t0_added += add

        s.tick()
        fills_at_landing.append(s.fill[1][cx])
        fy_at_landing.append(s.Fy[1][cx])

    total_water = s.total()
    print(f"injected={t0_added:.1f}  total_in_domain={total_water:.1f}  "
          f"conservation_ok={abs(total_water - t0_added) < 1.0}")
    print("landing-cell fill trajectory (every 5 ticks):",
          [round(fills_at_landing[i]) for i in range(0, ticks, 5)])
    print("landing-cell Fy trajectory (every 5 ticks):",
          [round(fy_at_landing[i]) for i in range(0, ticks, 5)])

    # oscillation check: does fy at the landing cell flip sign repeatedly in the tail (steady
    # stream, well past initial transient) instead of settling to a small/steady value?
    tail = fy_at_landing[-40:]
    sign_flips = sum(1 for a, b in zip(tail, tail[1:]) if a * b < 0)
    print(f"sign flips in Fy over last 40 ticks: {sign_flips} "
          f"({'OSCILLATING' if sign_flips > 15 else 'settled/monotone'})")
    return fills_at_landing, fy_at_landing


if __name__ == "__main__":
    print("=== continuous stream landing on puddle (base Sim, NO LevelSurface) ===")
    run_continuous_stream()
