"""
bench_large.py — larger-rig sanity checks for PhiFloodFluidSim (T4).

The small run_tests.py rig (CHUNK_SIZE=8, arm_h=8, gap=4) is fast but small;
algorithm correctness must hold at engine-like resolution too. Two rigs:

  1. CHUNK_SIZE=16, default U-tube (arm_h=16, gap=8) — taller arms, chunk
     boundaries crossed vertically AND horizontally by the liquid body.
  2. CHUNK_SIZE=8, LONG connector (gap=24) — the connector spans 4+ chunks;
     stresses the EPS_H steady-state error bound (EPS_H * path_len / 255
     voxels) and cross-chunk flood propagation.

Run:  python bench_large.py            (phi fix)
      python bench_large.py base       (baseline, for contrast)
"""

import sys
import time

from fluid_engine import VoxelWorld, FluidSim, WATER
import scenes
import run_tests as rt


def utube_run(name, chunk_size, nc, utube_kwargs, sim_cls, ticks1, ticks2):
    print("\n########## %s (%s, cs=%d) ##########" % (name, sim_cls.__name__, chunk_size))
    t0 = time.time()
    world = VoxelWorld(chunk_size, nc)
    sim = sim_cls(world)
    meta = scenes.build_utube(world, origin=(0, 0, 0), **utube_kwargs)
    sim.bootstrap_active_from_world()

    max_cons = 0
    for _ in range(ticks1):
        sim.step()
        max_cons = max(max_cons, abs(sim.last_conservation_delta))
    sA1 = rt.arm_surface(world, meta, "A"); sB1 = rt.arm_surface(world, meta, "B")
    mA1 = rt.arm_mass(world, meta, "A");    mB1 = rt.arm_mass(world, meta, "B")
    poured_fill = meta["water_poured"] * 255
    eq = abs(sA1 - sB1) <= 1 and abs(mA1 - mB1) <= max(1, poured_fill // 10)
    print("first fill (%d ticks): A y=%d m=%d | B y=%d m=%d | gap y=%d m=%d -> EQUALIZED: %s"
          % (ticks1, sA1, mA1, sB1, mB1, abs(sA1 - sB1), abs(mA1 - mB1),
             "PASS" if eq else "FAIL"))

    side = meta["arm_half_xz"]
    add_center = (meta["arm_a_x"], sA1 + 3, meta["z"])
    world.edit_box(add_center, (side, 2, side), "ADD", WATER)
    sim.add_active((add_center[0] // chunk_size,
                    add_center[1] // chunk_size,
                    add_center[2] // chunk_size))
    for _ in range(ticks2):
        sim.step()
        max_cons = max(max_cons, abs(sim.last_conservation_delta))
    sA2 = rt.arm_surface(world, meta, "A"); sB2 = rt.arm_surface(world, meta, "B")
    reacted = sB2 > sB1
    sc = rt.staircase_count(world, meta, "A") + rt.staircase_count(world, meta, "B")
    print("second pour (%d ticks): A y=%d | B y=%d (was %d) -> REACTIVITY: %s"
          % (ticks2, sA2, sB2, sB1, "PASS" if reacted else "FAIL"))
    print("staircase=%d  max|cons|=%d  elapsed=%.1fs" % (sc, max_cons, time.time() - t0))
    return eq and reacted and sc == 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "base":
        sim_cls = FluidSim
    else:
        from phi_flood import PhiFloodFluidSim
        sim_cls = PhiFloodFluidSim

    ok1 = utube_run("RIG 1: tall arms, cs=16", 16, (6, 6, 6),
                    {}, sim_cls, ticks1=350, ticks2=160)
    # arm_h/water_h sized so the poured volume (13 layers * 9 = 117 voxels)
    # exceeds the 27*3-voxel connector — otherwise all the water fits in the
    # connector and the arms are trivially "equal" at zero mass.
    ok2 = utube_run("RIG 2: long connector (gap=24), cs=8", 8, (10, 8, 8),
                    {"gap": 24, "arm_h": 16, "water_h": 12}, sim_cls,
                    ticks1=350, ticks2=160)
    print("\nLARGE-RIG RESULT: %s" % ("ALL PASS" if (ok1 and ok2) else "FAIL"))


if __name__ == "__main__":
    main()
