"""
fluid_engine.py — faithful 1:1 Python port of the master-branch voxel fluid sim.

This is NOT a conceptual reimplementation. It mirrors, line-for-line where it
matters, the real engine:

  * shaders/fluid_flux.comp        — the two-pass persistent-flux solver
                                      (Accelerate -> Transport) that ships on
                                      master.
  * simulations/fluid_simulation.gd — the CPU-side dispatcher: sparse flux
                                      paging, dispatch-range (active + 6 face
                                      neighbours) construction, per-tick
                                      page (un)map/restore, idle-guard eviction,
                                      conservation check, dirty-bitmask sleep.

Fidelity rules (see tools/fluid_prototypes/README.md):
  * 3D, not 2D.
  * Integer math is exact: GLSL `int(x)` truncates toward zero -> Python `int()`.
    GLSL `round()` tie breaking is implementation-defined; we use round-half-away
    from zero (`gl_round`) so .5 lands deterministically.
  * The same dispatch discipline: a chunk only participates if it is in this
    tick's flux_list; flux_page_table reflects *this tick's* membership, not
    "ever had a page". The page<0 guard in compute_s_out/compute_s_in and the
    Accelerate neighbour-page check are reproduced exactly (these are the
    dispatch-range leak fixes from 2026-07-02).
  * head() / eff_flux() / compute_s_out() / compute_s_in() are copied verbatim
    from the GLSL semantics.

This file is the base the Chorin pressure-projection fixes will later plug into
(see `FluidSim.extra_correction_pass`, a no-op hook between Accelerate and
Transport). It does NOT implement a granular solver (per the request) — SAND
exists as a static solid; only LIQUID materials (water, oil) flow.

Chunk size is a parameter. Tests use small chunk sizes for tractable run times;
the algorithm is identical at chunk_size = 32 (the real engine default).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants — mirror the engine.
# ---------------------------------------------------------------------------

# Material ids (from material_db.gd defaults).
AIR = 0
ROCK = 1
SAND = 2
WATER = 3
GLASS = 4
FIRE = 5
SMOKE = 6
OIL = 7

# SimulationClass enum values (material_def.gd).
SIM_NONE = 0
SIM_GRANULAR = 1
SIM_LIQUID = 2
SIM_GAS = 3
SIM_EFFECT = 4

FILL_CAP = 255
HEAD_MAX_DEPTH = 24
MIN_FLOW = 1
MAX_CHANGELOG = 524288

AXIS_DIR = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

# Default flux tuning (fluid_simulation.gd exports).
DEFAULT_DAMP = 0.97
DEFAULT_G_DT = 0.15
DEFAULT_GRAVITY_BIAS = 3.0
DEFAULT_SETTLE_THRESHOLD = 3
DEFAULT_STEPS_PER_TICK = 1
FLUX_PAGE_INITIAL_CAPACITY = 64


# ---------------------------------------------------------------------------
# Small numeric helpers — exact integer semantics.
# ---------------------------------------------------------------------------

def gl_round(x: float) -> int:
    """Round half away from zero (deterministic stand-in for GLSL round())."""
    if x >= 0.0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def pack_voxel(mat: int, fill: int, temp: int = 0, flags: int = 0) -> int:
    return (mat & 0xFF) | ((fill & 0xFF) << 8) | ((temp & 0xFF) << 16) | ((flags & 0xFF) << 24)


def unpack_voxel(u: int) -> Tuple[int, int, int, int]:
    return (u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF, (u >> 24) & 0xFF)


# ---------------------------------------------------------------------------
# VoxelWorld — the chunked voxel store + material sim table.
# ---------------------------------------------------------------------------

class VoxelWorld:
    """Mirrors WorldOrchestrator's voxel buffer / chunk table, minus rendering."""

    def __init__(self, chunk_size: int, nc: Tuple[int, int, int]):
        self.chunk_size = chunk_size
        self.vpc = chunk_size * chunk_size * chunk_size
        # World is centered at origin: chunks run -nc/2 .. nc/2-1 per axis.
        self.grid_min = (-nc[0] // 2, -nc[1] // 2, -nc[2] // 2)
        self.grid_span = nc

        # chunk_pos (tuple) -> slot (int).  Reverse map too.
        self.chunk_to_slot: Dict[Tuple[int, int, int], int] = {}
        self.slot_to_chunk: Dict[int, Tuple[int, int, int]] = {}
        self._next_slot = 0

        # slot -> list of uint32 voxels (length vpc).
        self.voxels: Dict[int, List[int]] = {}

        # material_id -> SimulationClass (256 entries, like material_sim buf).
        self.material_sim = [SIM_NONE] * 256
        self.material_sim[AIR] = SIM_NONE
        self.material_sim[ROCK] = SIM_NONE
        self.material_sim[SAND] = SIM_GRANULAR
        self.material_sim[WATER] = SIM_LIQUID
        self.material_sim[GLASS] = SIM_NONE
        self.material_sim[FIRE] = SIM_EFFECT
        self.material_sim[SMOKE] = SIM_GAS
        self.material_sim[OIL] = SIM_LIQUID

    # --- chunk management (mirrors WorldOrchestrator._add_chunk_to_grid) ---

    def is_loaded(self, cp: Tuple[int, int, int]) -> bool:
        return cp in self.chunk_to_slot

    def load_chunk(self, cp: Tuple[int, int, int]) -> int:
        if cp in self.chunk_to_slot:
            return self.chunk_to_slot[cp]
        slot = self._next_slot
        self._next_slot += 1
        self.chunk_to_slot[cp] = slot
        self.slot_to_chunk[slot] = cp
        self.voxels[slot] = [0] * self.vpc
        return slot

    def unload_chunk(self, cp: Tuple[int, int, int]) -> None:
        slot = self.chunk_to_slot.pop(cp, None)
        if slot is None:
            return
        self.slot_to_chunk.pop(slot, None)
        self.voxels.pop(slot, None)

    # --- coordinate math (mirrors fluid_flux.comp world_to_chunk / world_to_flat) ---

    def world_to_chunk(self, gpos: Tuple[int, int, int]) -> Tuple[int, int, int]:
        cs = self.chunk_size
        return (int(math.floor(gpos[0] / cs)),
                int(math.floor(gpos[1] / cs)),
                int(math.floor(gpos[2] / cs)))

    def chunk_table_idx(self, cp: Tuple[int, int, int]) -> int:
        lx = cp[0] - self.grid_min[0]
        ly = cp[1] - self.grid_min[1]
        lz = cp[2] - self.grid_min[2]
        if (lx < 0 or lx >= self.grid_span[0] or
                ly < 0 or ly >= self.grid_span[1] or
                lz < 0 or lz >= self.grid_span[2]):
            return -1
        return lx + ly * self.grid_span[0] + lz * self.grid_span[0] * self.grid_span[1]

    def world_to_flat(self, gpos: Tuple[int, int, int]) -> int:
        cp = self.world_to_chunk(gpos)
        if cp not in self.chunk_to_slot:
            return -1
        slot = self.chunk_to_slot[cp]
        cs = self.chunk_size
        lp = (gpos[0] - cp[0] * cs, gpos[1] - cp[1] * cs, gpos[2] - cp[2] * cs)
        local = lp[2] * cs * cs + lp[1] * cs + lp[0]
        return slot * self.vpc + local

    # --- voxel access (mirrors fluid_flux.comp read_a) ---

    def read_a(self, gpos: Tuple[int, int, int]) -> int:
        idx = self.world_to_flat(gpos)
        if idx < 0:
            return 0
        return self.voxels[idx // self.vpc][idx % self.vpc]

    def write_a(self, gpos: Tuple[int, int, int], u: int) -> None:
        idx = self.world_to_flat(gpos)
        if idx < 0:
            return
        self.voxels[idx // self.vpc][idx % self.vpc] = u & 0xFFFFFFFF

    # --- scene editing (mirrors edit_box ADD/REMOVE) ---

    def edit_box(self, center: Tuple[int, int, int], half: Tuple[int, int, int],
                 op: str, mat: int = AIR, fill: int = 255) -> None:
        for x in range(center[0] - half[0], center[0] + half[0] + 1):
            for y in range(center[1] - half[1], center[1] + half[1] + 1):
                for z in range(center[2] - half[2], center[2] + half[2] + 1):
                    if op == "REMOVE":
                        self.write_a((x, y, z), 0)
                    else:  # ADD
                        self.write_a((x, y, z), pack_voxel(mat, fill))

    def edit_sphere(self, center: Tuple[int, int, int], radius: int,
                    mat: int, fill: int = 255) -> None:
        r2 = radius * radius
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                for z in range(center[2] - radius, center[2] + radius + 1):
                    dx, dy, dz = x - center[0], y - center[1], z - center[2]
                    if dx * dx + dy * dy + dz * dz <= r2:
                        self.write_a((x, y, z), pack_voxel(mat, fill))

    # --- metrics ---

    def total_fill(self, mat: int) -> int:
        total = 0
        for slot in self.voxels:
            for u in self.voxels[slot]:
                if (u & 0xFF) == mat:
                    total += (u >> 8) & 0xFF
        return total

    def count_voxels(self, mat: int) -> int:
        c = 0
        for slot in self.voxels:
            for u in self.voxels[slot]:
                if (u & 0xFF) == mat:
                    c += 1
        return c


# ---------------------------------------------------------------------------
# FluidSim — faithful port of fluid_simulation.gd + fluid_flux.comp.
# ---------------------------------------------------------------------------

@dataclass
class FluxParams:
    damp: float = DEFAULT_DAMP
    g_dt: float = DEFAULT_G_DT
    gravity_bias: float = DEFAULT_GRAVITY_BIAS
    settle_threshold: int = DEFAULT_SETTLE_THRESHOLD
    steps_per_tick: int = DEFAULT_STEPS_PER_TICK


class FluidSim:
    def __init__(self, world: VoxelWorld, params: Optional[FluxParams] = None):
        self.world = world
        self.params = params or FluxParams()

        # Sparse flux paging (mirrors _flux_chunk_table / _flux_buf / _flux_page_table).
        self._flux_chunk_table: Dict[Tuple[int, int, int], int] = {}  # chunk_pos -> page
        self._flux_page_idle: Dict[Tuple[int, int, int], int] = {}
        self._flux_freelist: List[int] = []
        self._flux_capacity_pages = 0
        self.flux_buf: List[int] = []
        # Logical per-tick flux page table: chunk_pos -> page or -1.
        self.flux_page_table: Dict[Tuple[int, int, int], int] = {}

        self._active_chunks: Dict[Tuple[int, int, int], int] = {}  # chunk_pos -> idle
        self._frame = 0

        # Per-tick diagnostics.
        self.last_conservation_delta = 0
        self.last_gain = (0, 0)  # (flat_idx, amount)
        self.last_dirty_chunks: List[Tuple[int, int, int]] = []

        # Hook for Chorin-style correction passes (no-op in base flux replica).
        self.correction_enabled = False

        self._ensure_flux_buf_capacity(FLUX_PAGE_INITIAL_CAPACITY)

    # --- page management (mirrors fluid_simulation.gd) ---

    def _ensure_flux_buf_capacity(self, min_pages: int) -> None:
        if min_pages <= self._flux_capacity_pages:
            return
        new_cap = self._flux_capacity_pages
        if new_cap == 0:
            new_cap = max(FLUX_PAGE_INITIAL_CAPACITY, min_pages)
        while new_cap < min_pages:
            new_cap *= 2
        vpc3 = self.world.vpc * 3
        new_buf = [0] * (new_cap * vpc3)
        new_buf[: self._flux_capacity_pages * vpc3] = self.flux_buf
        self.flux_buf = new_buf
        for i in range(new_cap - 1, self._flux_capacity_pages - 1, -1):
            self._flux_freelist.append(i)
        self._flux_capacity_pages = new_cap

    def _allocate_flux_page(self, cp: Tuple[int, int, int]) -> None:
        if cp in self._flux_chunk_table:
            return
        if not self._flux_freelist:
            self._ensure_flux_buf_capacity(self._flux_capacity_pages + 1)
        if not self._flux_freelist:
            return
        page = self._flux_freelist.pop()
        self._flux_chunk_table[cp] = page

    def _free_flux_page(self, cp: Tuple[int, int, int]) -> None:
        page = self._flux_chunk_table.pop(cp, None)
        if page is None:
            return
        self._flux_freelist.append(page)
        self.flux_page_table[cp] = -1
        vpc3 = self.world.vpc * 3
        base = page * vpc3
        for i in range(base, base + vpc3):
            self.flux_buf[i] = 0

    # --- dispatch list (mirrors _build_flux_chunk_list) ---

    def build_flux_chunk_list(self, active_list) -> List[Tuple[int, int, int]]:
        seen: Set[Tuple[int, int, int]] = set()
        out: List[Tuple[int, int, int]] = []
        offsets = [
            (0, 0, 0),
            (1, 0, 0), (-1, 0, 0),
            (0, 0, 1), (0, 0, -1),
            (0, 1, 0), (0, -1, 0),
        ]
        for cp in active_list:
            for off in offsets:
                nb = (cp[0] + off[0], cp[1] + off[1], cp[2] + off[2])
                if nb not in seen and self.world.is_loaded(nb):
                    seen.add(nb)
                    out.append(nb)
        return out

    # --- activation ---

    def bootstrap_active_from_world(self) -> None:
        """Mark every loaded chunk containing a liquid voxel as active."""
        for cp, slot in self.world.chunk_to_slot.items():
            for u in self.world.voxels[slot]:
                m = u & 0xFF
                if m != 0 and self.world.material_sim[m] == SIM_LIQUID:
                    self._active_chunks[cp] = 0
                    break

    def add_active(self, cp: Tuple[int, int, int]) -> None:
        self._active_chunks[cp] = 0
        self._allocate_flux_page(cp)

    # --- flux lane resolution (mirrors fluid_flux.comp flux_lane /
    #     resolve_page_local, using chunk_pos membership + per-tick page table) ---

    def _resolve_page_local(self, gpos: Tuple[int, int, int]):
        cp = self.world.world_to_chunk(gpos)
        if not self.world.is_loaded(cp):
            return (-1, -1)
        page = self.flux_page_table.get(cp, -1)
        if page < 0:
            return (-1, -1)
        cs = self.world.chunk_size
        lp = (gpos[0] - cp[0] * cs, gpos[1] - cp[1] * cs, gpos[2] - cp[2] * cs)
        local = lp[2] * cs * cs + lp[1] * cs + lp[0]
        return (page, local)

    def flux_lane(self, gpos: Tuple[int, int, int], axis: int) -> int:
        cp = self.world.world_to_chunk(gpos)
        if not self.world.is_loaded(cp):
            return -1
        page = self.flux_page_table.get(cp, -1)
        if page < 0:
            return -1
        cs = self.world.chunk_size
        lp = (gpos[0] - cp[0] * cs, gpos[1] - cp[1] * cs, gpos[2] - cp[2] * cs)
        local = lp[2] * cs * cs + lp[1] * cs + lp[0]
        return page * self.world.vpc * 3 + local * 3 + axis

    # --- core flux primitives (mirrors fluid_flux.comp verbatim) ---

    def head(self, gpos: Tuple[int, int, int], mat_id: int) -> int:
        v = self.world.read_a(gpos)
        if (v & 0xFF) != mat_id:
            return 0
        f = (v >> 8) & 0xFF
        total = int(f)
        if f < 255:
            return total
        above = gpos
        for _ in range(HEAD_MAX_DEPTH):
            above = (above[0], above[1] + 1, above[2])
            av = self.world.read_a(above)
            if (av & 0xFF) != mat_id:
                break
            af = (av >> 8) & 0xFF
            total += int(af)
            if af < 255:
                break
        return total

    @staticmethod
    def eff_flux(raw: int) -> int:
        return raw if abs(raw) >= MIN_FLOW else 0

    def compute_s_out(self, gpos: Tuple[int, int, int], cs: int, vpc: int) -> float:
        page, local = self._resolve_page_local(gpos)
        if page < 0:
            return 0.0  # dispatch-range fix: un-dispatched chunk has no capacity
        v = self.world.read_a(gpos)
        fill = (v >> 8) & 0xFF
        total_out = 0
        for axis in range(3):
            lane = page * vpc * 3 + local * 3 + axis
            f = self.eff_flux(self.flux_buf[lane])
            if f > 0:
                total_out += f
            nb = (gpos[0] - AXIS_DIR[axis][0], gpos[1] - AXIS_DIR[axis][1], gpos[2] - AXIS_DIR[axis][2])
            nlane = self.flux_lane(nb, axis)
            if nlane >= 0:
                nf = self.eff_flux(self.flux_buf[nlane])
                if nf < 0:
                    total_out += -nf
        if total_out == 0:
            return 1.0
        return min(1.0, fill / total_out)

    def compute_s_in(self, gpos: Tuple[int, int, int], cs: int, vpc: int) -> float:
        page, local = self._resolve_page_local(gpos)
        if page < 0:
            return 0.0
        v = self.world.read_a(gpos)
        fill = (v >> 8) & 0xFF
        total_in = 0.0
        for axis in range(3):
            lane = page * vpc * 3 + local * 3 + axis
            f = self.eff_flux(self.flux_buf[lane])
            if f < 0:
                donor = (gpos[0] + AXIS_DIR[axis][0], gpos[1] + AXIS_DIR[axis][1], gpos[2] + AXIS_DIR[axis][2])
                total_in += float(-f) * self.compute_s_out(donor, cs, vpc)
            nb = (gpos[0] - AXIS_DIR[axis][0], gpos[1] - AXIS_DIR[axis][1], gpos[2] - AXIS_DIR[axis][2])
            nlane = self.flux_lane(nb, axis)
            if nlane >= 0:
                nf = self.eff_flux(self.flux_buf[nlane])
                if nf > 0:
                    total_in += float(nf) * self.compute_s_out(nb, cs, vpc)
        if total_in <= 0.0:
            return 1.0
        capacity = 255 - int(fill)
        if capacity < 0:
            capacity = 0
        return min(1.0, capacity / total_in)

    # --- the extension hook for Chorin-style fixes ---

    def extra_correction_pass(self) -> None:
        """Override (subclass) to inject a pressure-projection / phi-flood
        correction between Accelerate and Transport. No-op here."""
        pass

    # --- one simulation tick (mirrors FluidSimulation.step) ---

    def step(self) -> int:
        if not self._active_chunks:
            return 0
        world = self.world
        cs = world.chunk_size
        vpc = world.vpc

        active_list = list(self._active_chunks.keys())
        flux_list = self.build_flux_chunk_list(active_list)

        # Allocate pages for every chunk in this tick's dispatch range.
        for cp in flux_list:
            self._allocate_flux_page(cp)

        # Restore per-tick page table entries for dispatched chunks.
        for cp in flux_list:
            page = self._flux_chunk_table.get(cp, -1)
            if page >= 0:
                self.flux_page_table[cp] = page

        # Unmap + zero pages for paged chunks NOT in this tick's dispatch range.
        in_range = set(flux_list)
        for cp in list(self._flux_chunk_table.keys()):
            if cp not in in_range:
                self._free_flux_page(cp)  # zeroes page region + sets flux_page_table=-1

        merged_dirty: Set[int] = set()

        for _s in range(self.params.steps_per_tick):
            self._frame += 1
            if not flux_list:
                continue

            # ---- Pass A: Accelerate ----
            self._pass_accelerate(flux_list, cs, vpc)
            if self.correction_enabled:
                self.extra_correction_pass()

            # ---- Pass B: Transport (fused conservation guard + apply) ----
            dirty_this_step, cons_delta, gain = self._pass_transport(flux_list, cs, vpc)
            merged_dirty |= dirty_this_step
            self.last_conservation_delta = cons_delta
            self.last_gain = gain

        # Decode dirty slots -> chunk positions (mirrors _decode_dirty_bitmask).
        changed: List[Tuple[int, int, int]] = []
        for slot in merged_dirty:
            cp = world.slot_to_chunk.get(slot)
            if cp is not None:
                changed.append(cp)
        self.last_dirty_chunks = changed

        # Update active set (mirrors the idle-guard in step()).
        changed_set = set(changed)
        still_active: Dict[Tuple[int, int, int], int] = {}
        for cp, idle in self._active_chunks.items():
            if cp in changed_set:
                still_active[cp] = 0
            else:
                idle += 1
                if idle < 4:
                    still_active[cp] = idle
        for cp in changed_set:
            if cp not in still_active:
                still_active[cp] = 0

        for cp in list(self._active_chunks.keys()):
            if cp not in still_active:
                self._free_flux_page(cp)
        self._active_chunks = still_active

        # Evict pages allocated purely as flux-dispatch neighbours (never active).
        still_page_idle: Dict[Tuple[int, int, int], int] = {}
        for cp in list(self._flux_chunk_table.keys()):
            if cp in self._active_chunks:
                continue
            idle = self._flux_page_idle.get(cp, 0) + 1
            if idle < 4:
                still_page_idle[cp] = idle
            else:
                self._free_flux_page(cp)
        self._flux_page_idle = still_page_idle

        return len(changed)

    # --- Pass A: Accelerate (mirrors fluid_flux.comp pass == 0) ---

    def _pass_accelerate(self, flux_list, cs: int, vpc: int) -> None:
        world = self.world
        mat_sim = world.material_sim
        damp = self.params.damp
        g_dt = self.params.g_dt
        gravity_bias = self.params.gravity_bias

        for cp in flux_list:
            page = self._flux_chunk_table.get(cp, -1)
            if page < 0:
                continue
            slot = world.chunk_to_slot[cp]
            arr = world.voxels[slot]
            for lz in range(cs):
                for ly in range(cs):
                    for lx in range(cs):
                        local = lz * cs * cs + ly * cs + lx
                        gpos = (cp[0] * cs + lx, cp[1] * cs + ly, cp[2] * cs + lz)
                        u = arr[local]
                        mat_id = u & 0xFF
                        my_fill = (u >> 8) & 0xFF

                        my_is_liquid = (mat_id != 0 and mat_sim[mat_id] == SIM_LIQUID)
                        my_is_air = (mat_id == 0)

                        for axis in range(3):
                            d = AXIS_DIR[axis]
                            nb = (gpos[0] + d[0], gpos[1] + d[1], gpos[2] + d[2])
                            nv = world.read_a(nb)
                            nm = nv & 0xFF

                            lane = page * vpc * 3 + local * 3 + axis

                            nb_is_liquid = (nm != 0 and mat_sim[nm] == SIM_LIQUID)
                            nb_is_air = (nm == 0)

                            # Neighbour-page check (dispatch-range fix, 2026-07-02).
                            nb_cp = world.world_to_chunk(nb)
                            nb_page = self.flux_page_table.get(nb_cp, -1) if world.is_loaded(nb_cp) else -1
                            if nb_page < 0:
                                self.flux_buf[lane] = 0
                                continue

                            if (not my_is_liquid and not my_is_air) or (not nb_is_liquid and not nb_is_air):
                                self.flux_buf[lane] = 0
                                continue
                            if not my_is_liquid and not nb_is_liquid:
                                self.flux_buf[lane] = 0
                                continue
                            if my_is_liquid and nb_is_liquid and mat_id != nm:
                                self.flux_buf[lane] = 0
                                continue

                            active_mat = mat_id if my_is_liquid else nm
                            dh = self.head(gpos, active_mat) - self.head(nb, active_mat)
                            if axis == 1:
                                accel = (float(dh) - float(FILL_CAP)) * g_dt * gravity_bias
                                axis_cap = float(FILL_CAP) * gravity_bias
                            else:
                                accel = float(dh) * g_dt
                                axis_cap = float(FILL_CAP)

                            old_f = float(self.flux_buf[lane])
                            decayed = int(old_f * damp)
                            new_f = clamp(decayed + accel, -axis_cap, axis_cap)
                            self.flux_buf[lane] = gl_round(new_f)

    # --- Pass B: Transport (mirrors fluid_flux.comp pass == 1) ---

    def _pass_transport(self, flux_list, cs: int, vpc: int) -> Tuple[Set[int], int, Tuple[int, int]]:
        world = self.world
        mat_sim = world.material_sim
        settle_threshold = self.params.settle_threshold

        changelog: Dict[int, int] = {}  # flat -> new uint32 voxel
        cons_delta = 0
        gain_flat = -1
        gain_amount = 0
        dirty: Set[int] = set()

        for cp in flux_list:
            page = self._flux_chunk_table.get(cp, -1)
            if page < 0:
                continue
            slot = world.chunk_to_slot[cp]
            arr = world.voxels[slot]
            for lz in range(cs):
                for ly in range(cs):
                    for lx in range(cs):
                        local = lz * cs * cs + ly * cs + lx
                        gpos = (cp[0] * cs + lx, cp[1] * cs + ly, cp[2] * cs + lz)
                        u = arr[local]
                        mat_id = u & 0xFF
                        my_fill = (u >> 8) & 0xFF

                        if page >= 0:
                            my_s_out = self.compute_s_out(gpos, cs, vpc)
                            my_s_in = self.compute_s_in(gpos, cs, vpc)
                        else:
                            my_s_out = 0.0
                            my_s_in = 0.0

                        delta_f = 0.0
                        for axis in range(3):
                            d = AXIS_DIR[axis]
                            if page >= 0:
                                lane = page * vpc * 3 + local * 3 + axis
                                f = self.eff_flux(self.flux_buf[lane])
                                if f > 0:
                                    receiver = (gpos[0] + d[0], gpos[1] + d[1], gpos[2] + d[2])
                                    delta_f -= float(f) * my_s_out * self.compute_s_in(receiver, cs, vpc)
                                elif f < 0:
                                    donor = (gpos[0] + d[0], gpos[1] + d[1], gpos[2] + d[2])
                                    delta_f -= float(f) * self.compute_s_out(donor, cs, vpc) * my_s_in
                            nb = (gpos[0] - d[0], gpos[1] - d[1], gpos[2] - d[2])
                            nlane = self.flux_lane(nb, axis)
                            if nlane >= 0:
                                w = self.eff_flux(self.flux_buf[nlane])
                                if w > 0:
                                    delta_f += float(w) * self.compute_s_out(nb, cs, vpc) * my_s_in
                                elif w < 0:
                                    delta_f += float(w) * my_s_out * self.compute_s_in(nb, cs, vpc)

                        delta = gl_round(delta_f)
                        if delta == 0:
                            continue

                        is_liquid = (mat_id != 0 and mat_sim[mat_id] == SIM_LIQUID)
                        is_air = (mat_id == 0)
                        if not is_liquid and not is_air:
                            continue
                        if is_air and delta <= 0:
                            continue
                        if is_liquid and my_fill == 0 and delta <= 0:
                            continue

                        fill_mat = mat_id
                        if is_air:
                            found = False
                            for axis2 in range(3):
                                d2 = AXIS_DIR[axis2]
                                for sgn in (1, -1):
                                    sn = (gpos[0] + sgn * d2[0], gpos[1] + sgn * d2[1], gpos[2] + sgn * d2[2])
                                    sv = world.read_a(sn)
                                    sm = sv & 0xFF
                                    if sm != 0 and mat_sim[sm] == SIM_LIQUID:
                                        fill_mat = sm
                                        found = True
                                        break
                                if found:
                                    break
                            if not found:
                                continue

                        old_fill = int(my_fill) if is_liquid else 0
                        new_fill = int(clamp(old_fill + delta, 0, 255))
                        if new_fill == old_fill:
                            continue

                        applied = new_fill - old_fill
                        cons_delta += applied
                        if applied > gain_amount:
                            gain_amount = applied
                            gain_flat = slot * vpc + local

                        if new_fill == 0:
                            new_voxel = 0
                        elif is_air:
                            new_voxel = pack_voxel(fill_mat, new_fill)
                        else:
                            new_voxel = (u & 0xFFFF00FF) | (new_fill << 8)

                        flat = slot * vpc + local
                        changelog[flat] = new_voxel

                        if abs(applied) >= settle_threshold:
                            dirty.add(slot)

        # Apply changelog (mirrors WorldOrchestrator.apply_changelog writing voxels_a).
        for flat, new_voxel in changelog.items():
            slot = flat // vpc
            local = flat % vpc
            world.voxels[slot][local] = new_voxel & 0xFFFFFFFF

        return dirty, cons_delta, (gain_flat, gain_amount)

    # --- convenience runner ---

    def run(self, ticks: int) -> List[int]:
        history = []
        for _ in range(ticks):
            self.step()
            history.append(self.last_conservation_delta)
        return history
