"""
test_pile_ripple.py — regression test for the "flat lake ripple" bug found
2026-07-08 in-editor: placing one full water voxel directly on top of an
already-settled, full flat lake caused a spurious pressure wave to radiate
across the ENTIRE lake in a single tick, draining a wide crater (fill 1-5
"ring") from one added voxel's worth of mass. Root cause and fix are in
phi_flood.py's module docstring (TRUE-SURFACE PROTECTION).

This test builds a flat lake, lets it settle and sleep, places one extra
full voxel on top of one column, and checks that the disturbance stays
LOCAL (bounded radius, bounded mass movement) rather than radiating across
the whole lake.

Run:  python test_pile_ripple.py [base]   (base = old, unfixed behavior check)
"""

import sys

from fluid_engine import VoxelWorld, FluidSim, WATER, pack_voxel
import scenes


def max_disturbed_radius(world, meta, center_xz, before_fill):
    """Max Chebyshev (x,z) distance from center_xz at which any column's
    total fill differs from its pre-pile value by more than a small
    tolerance (a few units of settle noise)."""
    cs = world.chunk_size
    cols = {}
    for slot, arr in world.voxels.items():
        for local, u in enumerate(arr):
            if (u & 0xFF) != WATER:
                continue
            lx = local % cs
            ly = (local // cs) % cs
            lz = local // (cs * cs)
            cp = world.slot_to_chunk[slot]
            gx, gz = cp[0] * cs + lx, cp[2] * cs + lz
            cols[(gx, gz)] = cols.get((gx, gz), 0) + ((u >> 8) & 0xFF)
    max_r = 0
    for (gx, gz), fill in cols.items():
        before = before_fill.get((gx, gz), 0)
        if abs(fill - before) > 10:  # tolerance: settle noise, not real disturbance
            r = max(abs(gx - center_xz[0]), abs(gz - center_xz[1]))
            max_r = max(max_r, r)
    return max_r


def column_fill_map(world, meta):
    cs = world.chunk_size
    cols = {}
    for slot, arr in world.voxels.items():
        for local, u in enumerate(arr):
            if (u & 0xFF) != WATER:
                continue
            lx = local % cs
            ly = (local // cs) % cs
            lz = local // (cs * cs)
            cp = world.slot_to_chunk[slot]
            gx, gz = cp[0] * cs + lx, cp[2] * cs + lz
            cols[(gx, gz)] = cols.get((gx, gz), 0) + ((u >> 8) & 0xFF)
    return cols


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "base":
        sim_cls = FluidSim
    else:
        from phi_flood import PhiFloodFluidSim
        sim_cls = PhiFloodFluidSim
    print("sim class: %s" % sim_cls.__name__)

    world = VoxelWorld(8, (8, 4, 8))
    sim = sim_cls(world)
    meta = scenes.build_flat_lake(world, origin=(0, 0, 0), width=12, depth=4, water_h=4)
    sim.bootstrap_active_from_world()

    for _ in range(150):
        sim.step()
    print("settled: active=%d" % len(sim._active_chunks))
    before_fill = column_fill_map(world, meta)
    total_before = sum(before_fill.values())

    # Pile one extra full voxel directly on top of one column's existing top.
    # Scan for the ACTUAL surface rather than computing from water_h/floor_y
    # metadata (build_flat_lake's center+half box is asymmetric for even
    # water_h, off-by-one from the naive floor_y+water_h-1 computation).
    pile_x, pile_z = meta["origin"][0], meta["origin"][2]
    top_y = meta["floor_y"] - 5
    for y in range(meta["floor_y"] - 5, meta["floor_y"] + 20):
        v = world.read_a((pile_x, y, pile_z))
        if (v & 0xFF) == WATER:
            top_y = y
    world.write_a((pile_x, top_y + 1, pile_z), pack_voxel(WATER, 255))
    sim.add_active(world.world_to_chunk((pile_x, top_y + 1, pile_z)))
    total_after_pour = total_before + 255

    ticks_done = 0
    for block in (10, 20, 30, 60, 130):
        for _ in range(block):
            sim.step()
        ticks_done += block
        cols_now = column_fill_map(world, meta)
        radius = max_disturbed_radius(world, meta, (pile_x, pile_z), before_fill)
        total_now = sum(cols_now.values())
        print("t=+%3d  disturbed_radius=%d  total_fill=%d (was %d, poured %d)  active=%d"
              % (ticks_done, radius, total_now, total_before, total_after_pour,
                 len(sim._active_chunks)))

    final_radius = max_disturbed_radius(world, meta, (pile_x, pile_z), before_fill)
    # A single added voxel should disturb at most a small neighbourhood (a
    # handful of columns around the pile), never radiate across the whole
    # ~12-wide lake. Bar: radius <= 3 (generous vs. the expected 1-2 hop
    # spread of one voxel's worth of mass).
    passed = final_radius <= 3
    print("\nPILE RIPPLE CONTAINMENT: %s (final disturbed radius=%d, lake half-width=%d)"
          % ("PASS" if passed else "FAIL (spurious wide-radius ripple)", final_radius, 6))


if __name__ == "__main__":
    main()
