"""
diagnose_utube.py — quantitative root-cause dump for the U-tube failure.

Shows, at t=0 / mid-fill / settled, per-arm mass + surface y, and — the key
evidence — head() and the X flux lanes along the bottom connector. On the base
flux system every saturated mid-connector voxel reads head()==255 (rock/glass
above, so the straight-up column scan sees only the voxel's own fill), the
head *difference* between adjacent connector cells is 0, and therefore the
connector X-flux is 0 (or decaying residue) no matter how lopsided the arms
are. That zero gradient is the structural proof from
docs/CONNECTED_VESSEL_HANDOFF.md §3, reproduced numerically.

Optionally pass "phi" to run the same dump under PhiFloodFluidSim (the fix),
which should show a nonzero, monotonic gradient along the connector instead.

Run:  python diagnose_utube.py [phi]
"""

import sys

from fluid_engine import VoxelWorld, FluidSim, WATER
import scenes
from run_tests import arm_mass, arm_surface, CHUNK_SIZE, NC


def dump_connector(world, sim, meta, label):
    """Print head + phi (if available) + X flux lane for every voxel along the
    connector center line (y = base_y, z = meta z)."""
    y = meta["base_y"]
    z = meta["z"]
    x0 = meta["arm_a_x"] - meta["arm_half_xz"]
    x1 = meta["arm_b_x"] + meta["arm_half_xz"]

    mA = arm_mass(world, meta, "A")
    mB = arm_mass(world, meta, "B")
    sA = arm_surface(world, meta, "A")
    sB = arm_surface(world, meta, "B")
    print("\n--- %s ---" % label)
    print("arm A: mass=%6d surface_y=%2d | arm B: mass=%6d surface_y=%2d | mass gap=%d"
          % (mA, sA, mB, sB, abs(mA - mB)))

    phi = getattr(sim, "last_phi", None)

    print("connector scan (y=%d, z=%d, x=%d..%d):" % (y, z, x0, x1))
    print("  %4s %5s %6s %8s %8s" % ("x", "fill", "head", "phi", "fluxX"))
    prev_head = None
    grad_nonzero = 0
    for x in range(x0, x1 + 1):
        gpos = (x, y, z)
        v = world.read_a(gpos)
        mat = v & 0xFF
        fill = (v >> 8) & 0xFF
        if mat != WATER:
            print("  %4d %5s %6s %8s %8s" % (x, "-", "-", "-", "-"))
            prev_head = None
            continue
        h = sim.head(gpos, WATER)
        lane = sim.flux_lane(gpos, 0)  # axis 0 = X, the along-connector axis
        fx = sim.flux_buf[lane] if lane >= 0 else 0
        p = phi.get(gpos, "-") if phi is not None else "-"
        print("  %4d %5d %6d %8s %8d" % (x, fill, h, p, fx))
        if prev_head is not None and h != prev_head:
            grad_nonzero += 1
        prev_head = h
    print("  adjacent-pair HEAD gradients nonzero: %d  (0 == structurally blind connector)"
          % grad_nonzero)


def main():
    use_phi = len(sys.argv) > 1 and sys.argv[1] == "phi"
    world = VoxelWorld(CHUNK_SIZE, NC)
    if use_phi:
        from phi_flood import PhiFloodFluidSim
        sim = PhiFloodFluidSim(world)
        print("=== U-TUBE DIAGNOSTIC (PhiFloodFluidSim) ===")
    else:
        sim = FluidSim(world)
        print("=== U-TUBE DIAGNOSTIC (base flux replica) ===")
    meta = scenes.build_utube(world, origin=(0, 0, 0))
    sim.bootstrap_active_from_world()

    dump_connector(world, sim, meta, "t=0 (just poured, connector still empty)")

    for _ in range(60):
        sim.step()
    dump_connector(world, sim, meta, "t=60 (mid-fill: connector saturated, arms lopsided)")

    for _ in range(140):
        sim.step()
    dump_connector(world, sim, meta, "t=200 (base flux: falsely settled / fix: equalized)")


if __name__ == "__main__":
    main()
