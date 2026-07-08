"""
probe_staircase.py — is the post-second-pour staircase count transient or a
locked false-rest? Runs the U-tube reactivity sequence under PhiFloodFluidSim
and prints staircase count + arm masses every 40 ticks well past the harness's
cutoff, plus a full column profile at the end.
"""

from fluid_engine import VoxelWorld, WATER
from phi_flood import PhiFloodFluidSim
import scenes
import run_tests as rt


def profile(world, meta, side):
    ax = meta["arm_a_x"] if side == "A" else meta["arm_b_x"]
    z = meta["z"]
    out = []
    for y in range(meta["base_y"], meta["top_y"] + 1):
        v = world.read_a((ax, y, z))
        if (v & 0xFF) == WATER:
            out.append((v >> 8) & 0xFF)
        else:
            out.append(None)
    return out


def main():
    world = VoxelWorld(rt.CHUNK_SIZE, rt.NC)
    sim = PhiFloodFluidSim(world)
    meta = scenes.build_utube(world, origin=(0, 0, 0))
    sim.bootstrap_active_from_world()

    for _ in range(200):
        sim.step()
    sA1 = rt.arm_surface(world, meta, "A")
    side = meta["arm_half_xz"]
    world.edit_box((meta["arm_a_x"], sA1 + 3, meta["z"]), (side, 3, side), "ADD", WATER)
    sim.add_active((meta["arm_a_x"] // world.chunk_size,
                    (sA1 + 3) // world.chunk_size,
                    meta["z"] // world.chunk_size))

    for t in range(40, 521, 40):
        for _ in range(40):
            sim.step()
        sc = rt.staircase_count(world, meta, "A") + rt.staircase_count(world, meta, "B")
        mA = rt.arm_mass(world, meta, "A"); mB = rt.arm_mass(world, meta, "B")
        sA = rt.arm_surface(world, meta, "A"); sB = rt.arm_surface(world, meta, "B")
        print("t=+%3d  staircase=%2d  A(y=%d m=%d)  B(y=%d m=%d)  active=%d"
              % (t, sc, sA, mA, sB, mB, len(sim._active_chunks)))

    print("\ncenter column profiles bottom->top (None = not water):")
    print("  A:", profile(world, meta, "A"))
    print("  B:", profile(world, meta, "B"))


if __name__ == "__main__":
    main()
