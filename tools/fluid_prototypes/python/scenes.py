"""
scenes.py — test-scene builders for the fluid engine.

Each builder mutates a VoxelWorld (and optionally returns metadata the harness
needs, e.g. arm boundaries for the U-tube). They mirror the constructs used by
the in-engine rigs (world.gd build_utube_test_scene, edit_box ADD/REMOVE).

The U-tube is parameterised by chunk_size so the identical topology (two vertical
shafts + a fully-saturated horizontal connector, sealed) can be exercised at any
resolution. The base flux system is structurally blind to the connector gradient,
so this rig is the primary acceptance test.
"""

from typing import Dict, List, Tuple

from fluid_engine import (
    VoxelWorld, AIR, ROCK, GLASS, WATER, OIL, pack_voxel,
)


def _load_box(world: VoxelWorld, center: Tuple[int, int, int],
              half: Tuple[int, int, int]) -> None:
    cs = world.chunk_size
    for x in range(center[0] - half[0], center[0] + half[0] + 2):
        for y in range(center[1] - half[1], center[1] + half[1] + 2):
            for z in range(center[2] - half[2], center[2] + half[2] + 2):
                cp = (x // cs, y // cs, z // cs)
                world.load_chunk(cp)


def build_utube(world: VoxelWorld, origin: Tuple[int, int, int],
                arm_h: int = None, gap: int = None, arm_half_xz: int = 1,
                wall: int = 2, water_h: int = None,
                shell_mat: int = GLASS) -> Dict:
    """Build a sealed U-tube (two vertical shafts joined by a horizontal passage).

    Returns metadata: arm A/B world-x, base_y, top_y, water surface scan planes, etc.
    The rig is placed so the connector sits at y = base_y and both arms rise to
    top_y. Scaled to chunk_size by default.
    """
    cs = world.chunk_size
    ah = arm_h if arm_h is not None else cs
    gp = gap if gap is not None else max(4, cs // 2)
    wh = water_h if water_h is not None else max(2, ah // 2)

    arm_a_x = origin[0]
    arm_b_x = origin[0] + gp
    base_y = origin[1]
    top_y = base_y + ah
    z = origin[2]

    # --- load every chunk the rig touches ---
    shell_center = ((arm_a_x + arm_b_x) // 2, (base_y + top_y) // 2, z)
    shell_half = (
        (arm_b_x - arm_a_x) // 2 + arm_half_xz + wall,
        (top_y - base_y) // 2 + wall,
        arm_half_xz + wall,
    )
    _load_box(world, shell_center, shell_half)

    # --- solid shell (sealed container) ---
    world.edit_box(shell_center, shell_half, "ADD", shell_mat)

    # --- carve arm A shaft ---
    arm_a_center = (arm_a_x, (base_y + top_y) // 2, z)
    arm_half = (arm_half_xz, (top_y - base_y) // 2, arm_half_xz)
    world.edit_box(arm_a_center, arm_half, "REMOVE")

    # --- carve arm B shaft ---
    arm_b_center = (arm_b_x, (base_y + top_y) // 2, z)
    world.edit_box(arm_b_center, arm_half, "REMOVE")

    # --- carve bottom connector (joins both shafts) ---
    connector_center = ((arm_a_x + arm_b_x) // 2, base_y, z)
    connector_half = ((arm_b_x - arm_a_x) // 2 + arm_half_xz, 0, arm_half_xz)
    world.edit_box(connector_center, connector_half, "REMOVE")

    # --- pre-poured water block near the top of arm A (exact known volume) ---
    water_top_y = top_y - 2
    water_center = (arm_a_x, water_top_y - wh // 2, z)
    water_half = (arm_half_xz, wh // 2, arm_half_xz)
    world.edit_box(water_center, water_half, "ADD", WATER)

    return {
        "arm_a_x": arm_a_x, "arm_b_x": arm_b_x,
        "base_y": base_y, "top_y": top_y, "z": z,
        "arm_half_xz": arm_half_xz, "gap": gp, "wall": wall,
        # Actual poured volume: cross-section * slab height (2*half+1 per axis).
        "water_poured": (2 * arm_half_xz + 1) ** 2 * (2 * (wh // 2) + 1),
    }


def build_flat_lake(world: VoxelWorld, origin: Tuple[int, int, int],
                    width: int, depth: int, water_h: int) -> Dict:
    """A rock bowl (open top) with a flat water slab — tests settling / flat lake."""
    cs = world.chunk_size
    cx, cy, cz = origin
    # Rock floor + walls (a box shell with open top).
    bowl_center = (cx, cy + depth // 2, cz)
    bowl_half = (width // 2 + 1, depth // 2 + 1, width // 2 + 1)
    _load_box(world, bowl_center, bowl_half)
    world.edit_box(bowl_center, bowl_half, "ADD", ROCK)
    # Carve the interior, leaving a floor one voxel thick at the bottom.
    inner_center = (cx, cy + depth // 2 + 1, cz)
    inner_half = (width // 2, depth // 2, width // 2)
    world.edit_box(inner_center, inner_half, "REMOVE")
    # Water slab on the floor.
    water_center = (cx, cy + water_h // 2, cz)
    water_half = (width // 2, water_h // 2, width // 2)
    world.edit_box(water_center, water_half, "ADD", WATER)
    return {"origin": origin, "width": width, "depth": depth, "water_h": water_h,
            "floor_y": cy}


def build_falling_block(world: VoxelWorld, origin: Tuple[int, int, int],
                        size: int, mat: int = WATER, fall_dist: int = 20) -> Dict:
    """A solid block of liquid suspended in air inside a sealed rock basin — tests
    free fall + conservation. The basin (floor + 4 walls, open top) fully contains
    the water so mass can't "fall off the edge" of unloaded chunks."""
    cs = world.chunk_size
    floor_y = origin[1] - fall_dist
    # Rock basin: shell box with interior carved, like build_flat_lake.
    basin_center = (origin[0], (origin[1] + floor_y) // 2, origin[2])
    basin_half = (size + 3, fall_dist // 2 + 2, size + 3)
    _load_box(world, basin_center, basin_half)
    world.edit_box(basin_center, basin_half, "ADD", ROCK)
    inner = (size + 2, fall_dist // 2 + 1, size + 2)
    world.edit_box(basin_center, inner, "REMOVE")
    center = origin
    half = (size // 2, size // 2, size // 2)
    world.edit_box(center, half, "ADD", mat)
    return {"origin": origin, "size": size, "mat": mat, "floor_y": floor_y,
            "basin_half": basin_half}


def build_drain_column(world: VoxelWorld, origin: Tuple[int, int, int],
                       height: int, mat: int = WATER) -> Dict:
    """A vertical column of liquid with empty space below — tests drawdown."""
    cs = world.chunk_size
    _load_box(world, (origin[0], origin[1] + height // 2, origin[2]),
              (2, height // 2 + 4, 2))
    center = (origin[0], origin[1] + height // 2, origin[2])
    half = (1, height // 2, 1)
    world.edit_box(center, half, "ADD", mat)
    return {"origin": origin, "height": height, "mat": mat}
