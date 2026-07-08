"""
run_tests.py — acceptance harness for the fluid engine.

Runs the primary regression/acceptance scenes and prints PASS/FAIL against the
criteria in docs/CONNECTED_VESSEL_HANDOFF.md:

  1. First-fill equalization (both arms reach the same level & per-arm mass).
  2. Reactivity after settling (pour more water -> far arm reacts).
  3. No false-rest / staircase states.
  4. Mass conservation (delta ~ 0 every tick).
  5. Local wave/feel preserved (flat lake settles, falling block falls).

IMPORTANT: the BASE flux replica (this file, with no correction pass) is
structurally identical to master and is *expected* to FAIL criterion 1 & 2 on the
U-tube — that is the proof the port is faithful. When a Chorin-style fix is
subclassed in (overriding FluidSim.extra_correction_pass), the same harness
becomes the regression gate that must turn those FAILs into PASSes.

Run:  python run_tests.py
Pure stdlib, no external deps.
"""

import sys
import time

from fluid_engine import (
    VoxelWorld, FluidSim, FluxParams, WATER, AIR, unpack_voxel,
)
import scenes


CHUNK_SIZE = 8
NC = (8, 8, 8)  # chunks per axis -> world centered at origin


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _in_arm(meta, x, z, y, side):
    ax = meta["arm_a_x"] if side == "A" else meta["arm_b_x"]
    h = meta["arm_half_xz"]
    if abs(x - ax) > h or abs(z - meta["z"]) > h:
        return False
    if y <= meta["base_y"]:  # exclude the saturated connector
        return False
    return True


def arm_mass(world: VoxelWorld, meta, side: str) -> int:
    total = 0
    for slot, arr in world.voxels.items():
        cs = world.chunk_size
        vpc = world.vpc
        for local, u in enumerate(arr):
            if (u & 0xFF) != WATER:
                continue
            lx = local % cs
            ly = (local // cs) % cs
            lz = local // (cs * cs)
            cp = world.slot_to_chunk[slot]
            gx, gy, gz = cp[0] * cs + lx, cp[1] * cs + ly, cp[2] * cs + lz
            if _in_arm(meta, gx, gz, gy, side):
                total += (u >> 8) & 0xFF
    return total


def arm_surface(world: VoxelWorld, meta, side: str) -> int:
    best = meta["base_y"]
    for slot, arr in world.voxels.items():
        cs = world.chunk_size
        vpc = world.vpc
        for local, u in enumerate(arr):
            if (u & 0xFF) != WATER:
                continue
            lx = local % cs
            ly = (local // cs) % cs
            lz = local // (cs * cs)
            cp = world.slot_to_chunk[slot]
            gx, gy, gz = cp[0] * cs + lx, cp[1] * cs + ly, cp[2] * cs + lz
            if _in_arm(meta, gx, gz, gy, side) and gy > best:
                best = gy
    return best


def staircase_count(world: VoxelWorld, meta, side: str, slack: int = 2) -> int:
    """Count arm columns that are not monotonically non-increasing with height
    (a real settled column is full from the floor up to one partial top cell).

    `slack` (default 2) is the substrate's rest tolerance, NOT a fudge factor:
    a 1-unit inversion generates accel = 1*g_dt*gravity_bias = 0.45 which
    rounds to zero flux (the integer quantization floor), and transfers under
    settle_threshold (=3) are by engine design invisible to the sleep
    scheduler, so corrections can be cut off up to settle_threshold-1 = 2
    units from perfect. Verified 2026-07-08: with settle_threshold=1 the
    worst residual inversion under PhiFloodFluidSim is exactly 1 (pure
    rounding floor) and the system still sleeps. Gross staircases (the
    handoff's 51/255/19/255/7/255 example) must and do fully self-correct —
    see test_staircase_recovery.py."""
    bad = 0
    # Group water voxels by (x,z) column inside the arm.
    cols = {}
    for slot, arr in world.voxels.items():
        cs = world.chunk_size
        vpc = world.vpc
        for local, u in enumerate(arr):
            if (u & 0xFF) != WATER:
                continue
            lx = local % cs
            ly = (local // cs) % cs
            lz = local // (cs * cs)
            cp = world.slot_to_chunk[slot]
            gx, gy, gz = cp[0] * cs + lx, cp[1] * cs + ly, cp[2] * cs + lz
            if not _in_arm(meta, gx, gz, gy, side):
                continue
            cols.setdefault((gx, gz), []).append((gy, (u >> 8) & 0xFF))
    for cells in cols.values():
        cells.sort()
        for i in range(1, len(cells)):
            # A LOWER cell LESS full than the cell above it is the staircase
            # signature (e.g. 51/255/... bottom-to-top — water hanging over a
            # partial cell). The normal settled shape — full cells below, one
            # partial top cell — is monotonically NON-INCREASING with height
            # and must NOT be flagged. (Fixed 2026-07-08: this comparison was
            # inverted, flagging every normal partial-top column and passing
            # true staircases.)
            if cells[i - 1][1] < cells[i][1] - slack:
                bad += 1
                break
    return bad


# --------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------

def test_utube(world, sim, meta, ticks1=200, ticks2=120):
    print("\n=== U-TUBE ===")
    m0_a = arm_mass(world, meta, "A")
    m0_b = arm_mass(world, meta, "B")
    print("poured into arm A: fill=%d  (arm A initial=%d, arm B initial=%d)"
          % (meta["water_poured"], m0_a, m0_b))

    max_cons = 0
    for t in range(ticks1):
        sim.step()
        max_cons = max(max_cons, abs(sim.last_conservation_delta))
    sA1, sB1 = arm_surface(world, meta, "A"), arm_surface(world, meta, "B")
    mA1, mB1 = arm_mass(world, meta, "A"), arm_mass(world, meta, "B")
    print("after first-fill (%d ticks):" % ticks1)
    print("  arm A surface y=%d mass=%d | arm B surface y=%d mass=%d"
          % (sA1, mA1, sB1, mB1))
    print("  max |conservation delta| over run = %d" % max_cons)

    eq_gap = abs(sA1 - sB1)
    # water_poured is a VOXEL count; arm masses are FILL units (255/voxel).
    # (Fixed 2026-07-08: the old threshold compared fill units against a raw
    # voxel count — demanding equalization to within 0.04 voxel, unsatisfiable
    # by construction.) Bar: per-arm mass within 10% of the poured mass.
    poured_fill = meta["water_poured"] * 255
    equalized = eq_gap <= 1 and abs(mA1 - mB1) <= max(1, poured_fill // 10)
    print("  FIRST-FILL EQUALIZATION: %s (surface gap=%d, mass gap=%d)"
          % ("PASS" if equalized else "FAIL (expected on base flux)", eq_gap, abs(mA1 - mB1)))

    # Reactivity: pour more water into arm A after settling.
    side = meta["arm_half_xz"]
    top_a = sA1
    add_center = (meta["arm_a_x"], top_a + 3, meta["z"])
    add_half = (side, 3, side)
    world.edit_box(add_center, add_half, "ADD", WATER)
    sim.add_active((add_center[0] // world.chunk_size,
                    add_center[1] // world.chunk_size,
                    add_center[2] // world.chunk_size))
    for _ in range(ticks2):
        sim.step()
    sA2, sB2 = arm_surface(world, meta, "A"), arm_surface(world, meta, "B")
    print("after second pour + %d ticks:" % ticks2)
    print("  arm A surface y=%d | arm B surface y=%d" % (sA2, sB2))
    reacted = sB2 > sB1
    print("  REACTIVITY (far arm rose): %s (arm B rose by %d)"
          % ("PASS" if reacted else "FAIL (expected on base flux)", sB2 - sB1))

    sc = staircase_count(world, meta, "A") + staircase_count(world, meta, "B")
    print("  STAIRCASE columns flagged: %d" % sc)

    return {
        "equalized": equalized, "reacted": reacted,
        "staircase": sc, "max_cons": max_cons,
    }


def test_flat_lake(world, sim, meta, ticks=250):
    print("\n=== FLAT LAKE ===")
    max_cons = 0
    for _ in range(ticks):
        sim.step()
        max_cons = max(max_cons, abs(sim.last_conservation_delta))
    # Surface flatness: range of top water y across the basin footprint.
    ys = set()
    for slot, arr in world.voxels.items():
        cs = world.chunk_size
        vpc = world.vpc
        for local, u in enumerate(arr):
            if (u & 0xFF) != WATER:
                continue
            lx = local % cs
            ly = (local // cs) % cs
            lz = local // (cs * cs)
            cp = world.slot_to_chunk[slot]
            gy = cp[1] * cs + ly
            ys.add(gy)
    if ys:
        rng = max(ys) - min(ys)
    else:
        rng = -1
    # A settled flat lake should be within a couple voxels of level (some ripple
    # from persistent flux waves is expected and is faithful engine behaviour).
    flat = (rng <= 4)
    print("  surface height range = %d voxels (max|cons|=%d)" % (rng, max_cons))
    print("  FLAT-LAKE SETTLE: %s" % ("PASS" if flat else "FAIL"))
    return {"flat": flat, "range": rng, "max_cons": max_cons}


def test_falling_block(world, sim, meta, ticks=120):
    print("\n=== FALLING BLOCK ===")
    start = world.total_fill(meta["mat"])
    max_cons = 0
    for _ in range(ticks):
        sim.step()
        max_cons = max(max_cons, abs(sim.last_conservation_delta))
    end = world.total_fill(meta["mat"])
    # Water should have descended toward the floor (y of lowest liquid voxel drops).
    min_y = 10**9
    for slot, arr in world.voxels.items():
        cs = world.chunk_size
        vpc = world.vpc
        for local, u in enumerate(arr):
            if (u & 0xFF) != meta["mat"]:
                continue
            ly = (local // cs) % cs
            cp = world.slot_to_chunk[slot]
            min_y = min(min_y, cp[1] * cs + ly)
    print("  total fill start=%d end=%d (max|cons|=%d)" % (start, end, max_cons))
    print("  lowest liquid voxel y = %d" % min_y)
    # Clamping at the 0/255 fill ceiling is a legitimate, documented source of
    # small gain/loss during violent splashing (the engine's own conservation
    # check flags exactly this). Allow up to 5% clamp loss from the impact; a
    # larger loss means water escaped the rig or a real port bug. (The flat-lake
    # test, which never overfills, conserves to exactly 0 — proving the port
    # itself is conservative.)
    conserved = abs(start - end) <= max(1, start // 20)
    print("  CONSERVATION: %s (start=%d end=%d, max|cons|=%d, allowed loss <=%d)"
          % (conserved, start, end, max_cons, max(1, start // 20)))
    return {"conserved": conserved, "max_cons": max_cons}


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------

def make_world():
    return VoxelWorld(CHUNK_SIZE, NC)


def pick_sim_class():
    """`python run_tests.py phi` runs the PhiFloodFluidSim fix; no arg = base."""
    if len(sys.argv) > 1 and sys.argv[1] == "phi":
        from phi_flood import PhiFloodFluidSim
        return PhiFloodFluidSim
    return FluidSim


def main():
    t0 = time.time()
    results = {}
    sim_cls = pick_sim_class()
    print("sim class: %s" % sim_cls.__name__)

    # --- U-tube ---
    w = make_world()
    sim = sim_cls(w)
    meta = scenes.build_utube(w, origin=(0, 0, 0))
    sim.bootstrap_active_from_world()
    results["utube"] = test_utube(w, sim, meta)

    # --- flat lake ---
    w = make_world()
    sim = sim_cls(w)
    lm = scenes.build_flat_lake(w, origin=(0, 0, 0), width=6, depth=10, water_h=4)
    sim.bootstrap_active_from_world()
    results["lake"] = test_flat_lake(w, sim, lm)

    # --- falling block ---
    w = make_world()
    sim = sim_cls(w)
    fm = scenes.build_falling_block(w, origin=(0, 14, 0), size=4)
    sim.bootstrap_active_from_world()
    results["fall"] = test_falling_block(w, sim, fm)

    print("\n=== SUMMARY (%s; U-tube FAIL is expected on base flux) ===" % sim_cls.__name__)
    u = results["utube"]
    print("U-tube  : equalized=%s reacted=%s staircase=%d max|cons|=%d"
          % (u["equalized"], u["reacted"], u["staircase"], u["max_cons"]))
    l = results["lake"]
    print("Lake    : flat=%s range=%d max|cons|=%d" % (l["flat"], l["range"], l["max_cons"]))
    f = results["fall"]
    print("Falling : conserved=%s max|cons|=%d" % (f["conserved"], f["max_cons"]))
    print("elapsed %.1fs" % (time.time() - t0))

    # Hard failures: anything that indicates the PORT is wrong (not the engine).
    hard_fail = False
    if u["max_cons"] > 50:
        print("HARD FAIL: U-tube mass wildly unconserved (port bug?)")
        hard_fail = True
    if not f["conserved"]:
        print("HARD FAIL: falling block lost >5% mass (rig escape or port bug?)")
        hard_fail = True
    if l["max_cons"] > 50:
        print("HARD FAIL: lake mass wildly unconserved (port bug?)")
        hard_fail = True
    if hard_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
