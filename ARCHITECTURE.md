# Voxel Engine — Architecture

## 1. Why We're Changing

The current architecture grew organically and accumulated coupling between systems that should be independent. Four foundational problems:

### 1.1 Voxel Storage Is Lossy

CPU `block_data` stores 1 byte per voxel (material_id only). Mass, health, and flags live only in GPU buffers. The CPU doesn't know if water is half-full or full. Collision rebuilds, explosion logic, and any future CPU-side system are flying blind.

### 1.2 Simulation Is Tangled Into Storage

`GPUSandSimulation` owns the GPU voxel buffers that the renderer reads from. The renderer cannot function without the simulation node. This means:
- Removing simulation breaks rendering
- Changing voxel format requires updating both systems simultaneously
- The renderer has no independence — it's a slave to the simulation controller

### 1.3 Material Definition Contains Simulation State

`MaterialDef` has `initial_health` and `burn_time` — simulation concerns baked into the material definition. The material database should describe what a material *is*, not how it behaves under fire. This violates the separation between definition and behavior.

### 1.4 No Single Writer

Fluid, Fire, Player, and Explosion all write to the same voxel bytes simultaneously. This caused the CAS deadlocks, oscillation bugs, and order-dependent races that plagued the previous simulation. There is no arbitration layer — whoever writes last wins.

### 1.5 Dead Code Accumulation

Three shader files exist but are never dispatched:
- `mark_receives.comp` — superseded by mass-based flux
- `pressure_propagation.comp` — never integrated
- `fire_sim.comp` — stub, never instantiated

`DebugOverlay.gd` is coupled to `GPUSandSimulation` for a debug visual that adds no value during active development.

---

## 2. Foundational Pillars

**The Kernel (Pillars 0 & 1)** provides chunk storage and rendering. It does not permit systems to mutate voxels arbitrarily. Pillars 2–4 sit on top of the Kernel and define how simulation systems interact with it safely — preventing the race conditions, CAS deadlocks, and order-dependent bugs from the previous architecture.

### Pillar 0 — Chunks (Done)

A chunk is a 32³ region of world space. Each chunk is a `StaticBody3D` for collision. Chunks are generated in the background and rebuilt incrementally. This layer is stable and doesn't need changes.

### Pillar 1 — Voxel Storage

A voxel is the smallest addressable volume of world space. The voxel storage system has exactly one responsibility: **store the state of every addressable volume in the chunk.**

Every voxel stores only state information that is unique to that location. Byte interpretation varies by material phase:

**SOLID / GRANULAR voxels:**

| Byte | Field | Range | Description |
|------|-------|-------|-------------|
| 0 | material_id | 0–255 | Maps to Material Database. 0 = AIR. |
| 1 | fill | 0–255 | Always 255 for solids. Reserved for future sub-voxel solid density. |
| 2 | temperature | −128..127 | Signed. Controlled by thermal simulation (Phase 8). |
| 3 | flags | bit flags | Engine flags only (see Pillar 3). |

**LIQUID / GAS voxels (Phase 7d+):**

| Byte | Field | Range | Description |
|------|-------|-------|-------------|
| 0 | material_id | 0–255 | Water, oil, lava, smoke, etc. |
| 1 | pressure | 0–255 | Amount of fluid in this cell. 0 = empty (becomes AIR). Drives flux physics. |
| 2 | vx | −127..127 | Signed horizontal velocity in X. Packed as u8 (`vx + 127`). |
| 3 | vz | −127..127 | Signed horizontal velocity in Z. |

Rationale for LIQUID/GAS byte repurposing: per-liquid temperature is not meaningful (thermal sim treats all water at uniform conductivity); liquids carry no engine flags; velocity enables momentum, waves, and splash behavior. The thermal pass skips byte 2 for LIQUID/GAS voxels.

**AIR = 0u.** All air voxels are stored as exactly zero. This is critical for CAS operations in granular simulation and for the particle rebake path.

**A voxel does NOT store material properties.** A voxel is a location with information — not a material.

**AIR = 0u.** All air voxels are stored as exactly zero. This is critical for CAS operations in simulation.

#### Pillar 1.1 — Material Database

`MaterialID` maps to a Material Database entry that describes what a material *is* — not how it behaves under any specific simulation.

**`MaterialDef` fields (current):**

| Field | Type | Description |
|-------|------|-------------|
| id | int | 0-255, unique identifier |
| name | String | Display name |
| phase | Phase | Physical state: SOLID / LIQUID / GAS / EFFECT |
| simulation | SimulationClass | How the engine moves it: STATIC / GRANULAR / LIQUID / GAS / EFFECT |
| gravity | GravityMode | NONE / DOWN / UP / NEUTRAL |
| color | Color | RGBA (alpha = transparency) |

**Enums:**

```gdscript
enum Phase           { SOLID=0, LIQUID=1, GAS=2, EFFECT=3 }
enum SimulationClass { NONE=0, GRANULAR=1, LIQUID=2, GAS=3, EFFECT=4 }
enum GravityMode     { NONE=0, DOWN=1, UP=2, NEUTRAL=3 }

const SIMULATION_COUNT := 5  # size of _simulation_counts array in Chunk
const SIM_BITS: Array[int] = [0, 1, 2, 4, 8]  # indexed by SimulationClass value
```

`SimulationClass` values are **sequential** (not power-of-two). `SIM_BITS` is the precomputed bitmask for each class — `SIM_BITS[GRANULAR]=1`, `SIM_BITS[LIQUID]=2`, `SIM_BITS[GAS]=4`, `SIM_BITS[EFFECT]=8`. Using a lookup array avoids `1 << (sim - 1)` bit-shifts, which GDScript's parser rejects when the operand may be zero. `NONE=0` means "no simulation" — rock, wood, and air all use `NONE`.

**Collidability** is derived from `SimulationClass`: `NONE` (non-air, mat_id != 0) and `GRANULAR` → collidable; everything else → not.

**GPU buffer layout** (32 bytes/material):

| Offset | Field |
|--------|-------|
| 0 | packed RGBA8 color (u32) |
| 4 | simulation class (u32) |
| 8 | phase (u32) |
| 12 | gravity mode (u32) |
| 16-31 | reserved |

**No simulation state.** Properties like density, friction, flammability, and melting point are deferred to later phases when simulation systems are rebuilt. The definition stores only what the renderer and collision system need right now.

**Default materials:**

| ID | Name | Phase | Simulation | Gravity |
|----|------|-------|------------|---------|
| 0 | AIR | GAS | NONE | NONE |
| 1 | ROCK | SOLID | NONE | NONE |
| 2 | SAND | SOLID | GRANULAR | DOWN |
| 3 | WATER | LIQUID | LIQUID | DOWN |
| 4 | WOOD | SOLID | NONE | NONE |
| 5 | FIRE | EFFECT | EFFECT | UP |
| 6 | SMOKE | GAS | GAS | UP |
| 7 | OIL | LIQUID | LIQUID | DOWN |

---

### Pillar 2 — The Orchestrator (Single Authority)

The Orchestrator is the only code allowed to write to `Chunk.block_data`. No simulation system (Fluid, Fire, Physics, Player) may directly call `chunk.set_voxel()`.

**Why:** In the previous architecture, Fluid, Fire, and the Player all wrote to the same bytes simultaneously. The Orchestrator eliminates this by acting as a transaction manager.

#### 2.1 The Mutation Request

Instead of direct writes, systems submit a `VoxelMutation`:

```gdscript
class VoxelMutation:
	var chunk_pos: Vector3i
	var local_idx: int
	var material_id: int   # -1 means "no change"
	var fill: int          # -1 means "no change"
	var temperature: int   # -1 means "no change"
	var flags: int         # -1 means "no change"
	var priority: int      # HIGH, NORMAL, LOW (used for conflict resolution)
```

#### 2.2 The Commit Phase

Once per frame (after all systems have finished their `compute()` phase), the Orchestrator executes:

1. **Validation:** Clamps fill to 0–255, ensures material_id is valid, bounds-checks the chunk.
2. **Conflict Resolution:** If two systems (e.g., Fluid and Fire) try to write to the same voxel in the same frame, the Orchestrator decides who wins based on priority (e.g., Fire spreading into water → water wins, cooling it).
3. **Commit:** Writes the final resolved values to `block_data`.
4. **Post-Commit:** Marks the chunk as `DIRTY_MESH` (so the renderer re-uploads 128KB) and triggers the chunk_table slot update if the material changed to/from AIR.

#### 2.3 Why This Fixes the Old Bugs

The oscillation, CAS deadlocks, and order-dependent races were all caused by multiple writers competing over the same buffer. The Orchestrator eliminates this by:

- **Single writer:** Only the Orchestrator touches `block_data`. Period.
- **Deferred writes:** Systems declare intent (mutations), not actions. The Orchestrator batches and resolves.
- **Conflict resolution:** Priority-based arbitration prevents two systems from fighting over the same voxel.
- **Deterministic order:** The Orchestrator enforces a strict execution order: Temperature → Fluids → Fire → Reactions.

#### 2.4 block_data Ownership and the Sync Budget

`Chunk.block_data` is a **CPU cache of GPU state**, not a real-time mirror. The GPU (`_voxel_buf_a`) is always authoritative at runtime. block_data is populated once at generation and is otherwise stale during active gameplay.

**Sole writer rule:** Only `WorldOrchestrator` writes to `block_data`. No simulation, editor tool, or player system may write directly. The legal paths are:

| Path | When |
|------|------|
| `apply_mutation()` at chunk generation | One-time init — sets initial block_data from the generation loop |
| `request_chunk_sync(chunk_pos)` | On-demand full sync: downloads the 128KB chunk slot from `_voxel_buf_a`, writes block_data, calls `rebuild_counts_and_mask()`. Budget-gated. |
| `sync_block_data_from_buf(entries)` | Incremental: applies changelog `(flat_idx, new_value)` entries to block_data without a full download. O(changelog_size) writes, typically <<100 entries per active chunk. |

**Download budget:** Both sync paths count against `_sync_budget_remaining` (reset each frame). Default: 4 full syncs OR 8 incremental syncs per frame. Excess requests are deferred to the next frame. `sync_all_for_save()` bypasses the budget — it runs once at save time over all loaded chunks.

**Who requests syncs at runtime:** Nothing. `block_data` is no longer used for collision — `CollisionSystem` downloads only surface voxels via `extract_surface_voxels()` (GPU compute, compact). All other systems — rendering, simulation, cursor targeting, editing — read from GPU buffers or GPU readback results. `block_data` is only read for: initial GPU upload at generation, save-time full-chunk sync, and `query_voxel_material()` (single-voxel cursor hit, kept accurate for player edits only). No runtime system may access `block_data` expecting current simulation state.

---

### Pillar 3 — Simulation Runtime

**How does the world change?** The runtime manages simulation lifecycles. Simulations own their own work state and propose mutations. The orchestrator commits.

#### Ownership table

| System | Owns | Does NOT own |
|---|---|---|
| **ChunkManager** | Persistent voxel storage, chunk lifecycle | Simulation logic, mutation authority |
| **SimulationRuntime** | Discovery (chunk scan on external commits), active simulation set, frame dispatch | Simulation state, geometry, mutation commit |
| **SimulationContext** | Read-only world interface, mutation writer | Persistent state, candidate lists |
| **Simulation** | Algorithm, all persistent algorithm state (frontiers, queues, graphs) | World state, scheduling decisions |
| **WorldOrchestrator** | Mutation authority, conflict resolution, commit | Scheduling, simulation logic |

#### Scheduler invariants

1. Every unstable voxel eventually gets simulated.
2. Stable worlds become idle — `_active` converges to empty with no timers.
3. No simulation can lose work — once active, it stays until it reports done.
4. Simulations own their notion of "unfinished" — the runtime only asks `has_work()`.
5. The scheduler never understands physics.
6. Discovery and residual work are separate: external commits trigger scanning; simulation-generated commits don't.

#### Discovery vs. residual

There are two kinds of work:

**Discovery** — triggered by external mutations (player edit, explosion, world load). The runtime scans the affected chunks once, builds per-simulation candidate lists, and calls `add_candidates()` on each simulation with non-empty results.

**Residual** — the simulation moved something; its frontier already contains the next positions to check. The simulation propagates this itself, without any re-scanning. `chunks_committed` fires for simulation-generated mutations too, but `GranularSimulation`'s frontier uses Dictionary set semantics — duplicate positions are a no-op, so residual re-discovery is harmless and free.

```
External mutation (player places sand)
      ↓ chunks_committed fires
      ↓ runtime scans affected chunks → finds granular voxels
      ↓ sim.add_candidates(positions)
      ↓ sim added to _active

Sand falls (simulation mutation)
      ↓ chunks_committed fires
      ↓ runtime scans → finds same sand (now at new position)
      ↓ sim.add_candidates(positions) → already in Dictionary frontier → no-op
      ↓ sim already in _active; frontier propagates naturally
```

#### SimulationRuntime — frame loop

**On `chunks_committed`** (any commit — external or simulation):
1. Scan each affected chunk (skip if `active_simulation_mask() == 0`)
2. Build per-simulation candidate arrays from non-NONE voxels
3. For each simulation with non-empty candidates: call `add_candidates()`, add to `_active` if not already there

**Each frame (`run()`):**
1. For each simulation in `_active`: call `sim.step(ctx)`, collect mutations
2. After step: if `sim.has_work()` is false, remove from `_active`
3. Return all mutations to WorldOrchestrator

No regions. No AABBs. No TTLs. No geometry in the runtime.

```
chunks_committed
      ↓
SimulationRuntime scans once (active_simulation_mask fast path)
      ↓
sim.add_candidates(positions)     ← only for non-empty candidate lists
      ↓
_active set grows

Each frame:
  for sim in _active:
      sim.step(ctx) → mutations
      if not sim.has_work(): remove from _active
```

#### VoxelSimulation interface

```gdscript
func get_simulation_class() -> int          # declares which SimulationClass this handles
func add_candidates(chunk_positions: Array[Vector3i]) -> void  # runtime found relevant chunks
func step(ctx: SimulationContext) -> void   # advance the simulation one frame
func has_work() -> bool                     # false = runtime removes from active set
```

The simulation owns all persistent state. Example — `GranularSimulation` maintains a `Dictionary` frontier (Vector3i → true, set semantics). `add_candidates` merges positions in. `step` advances the frontier one level. `has_work` checks emptiness. No task IDs, no lifecycle ceremony, no coverage tracking.

```gdscript
func step(ctx: SimulationContext) -> void:
    var next: Dictionary = {}
    for pos in _frontier:
        var below = pos + Vector3i.DOWN
        if ctx.get_material_id(below) == 0:
            ctx.emit_move(pos, below)
            next[below] = true
    _frontier = next
```

Adding a new simulation: extend `VoxelSimulation`, override `get_simulation_class()`, implement `add_candidates` / `step` / `has_work`, add as child of `SimulationRuntime`. Zero runtime changes.

#### SimulationContext — lightweight world view

Created fresh each frame, owns no persistent data.

```gdscript
ctx.get_material_id(world_pos)  # single-field read, resolves chunk internally
ctx.get_material(material_id)   # → MaterialDef
ctx.emit_move(from, to)         # queues atomic move as two mutations
ctx.flush_mutations()           # returns queued mutations, clears buffer
ctx.random() / ctx.frame()
```

`positions()` is removed — simulations own their frontiers.

#### Mutation commit

`WorldOrchestrator.commit()` is unchanged: conflict resolution (highest priority wins per world_pos), then `chunk.apply_mutation()` for each resolved mutation.

```gdscript
func commit():
    for m in best.values():
        var chunk = _chunk_manager.chunks.get(m.chunk_pos)
        if not chunk: continue
        var old_sim := _material_simulation[chunk.block_data[m.local_idx]]
        var new_sim := _material_simulation[m.material_id]
        if chunk.apply_mutation(m.local_idx, m.material_id, m.fill, m.temp, m.flags, old_sim, new_sim):
            affected_chunks[m.chunk_pos] = true
    chunks_committed.emit(positions)
```

Active regions are removed — `_expand_or_create_region` no longer called.

#### The three foundational pillars

- **Pillar 1 — What exists?** Voxels stored in chunks. Each voxel is an addressable volume containing a material id and local state (fill, temperature, flags). The world is persistent.
- **Pillar 2 — What can exist?** Materials define physical identity and simulation class. The material database is the single source of truth for what a material is — not how it behaves.
- **Pillar 3 — How does the world change?** External mutations trigger a chunk scan; the runtime delivers candidates to simulations. Simulations own their internal work state and propagate residuals without re-scanning. The orchestrator is the sole authority that commits mutations.

---

### Pillar 4 — GPU Buffer Architecture

The voxel data lives in two places: `Chunk.block_data` (CPU, authoritative for collision/save/load) and `_voxel_buf` (GPU, authoritative for rendering and simulation). These must stay in sync through a controlled pipeline, not ad-hoc uploads.

#### 4.1 Buffer Ownership (target state after Phase 5)

| Buffer | Owner | Purpose |
|--------|-------|---------|
| `_voxel_buf_a` | WorldOrchestrator | Canonical GPU state. Renderer always binds this. Simulation reads this. Changelog sync writes back to this. |
| `_voxel_buf_b` | WorldOrchestrator | Simulation scratch: GPU sim writes here only. Never read by renderer or CPU. Never swapped. |
| `_chunk_table` | WorldOrchestrator | Maps chunk positions → slot offsets in the voxel buffers. Shared by renderer and simulation. |
| `_changelog_buf` | WorldOrchestrator | GPU-written list of `(global_voxel_idx, new_u32)` pairs. CPU downloads to patch `_voxel_buf_a` (via `patch_apply.comp`). |
| `_surface_out_buf` | WorldOrchestrator | Output buffer for `surface_extract.comp`. Reused across calls. Layout: `u32 count` then `u32 entries[]` packed as `(local_flat_idx & 0xFFFF) \| (mat_id << 16)`. Max 65536 entries (64³). |
| `_surface_params_buf` | WorldOrchestrator | Persistent params buffer for `surface_extract.comp`: `chunk_size`, `slot`, `max_entries`, `_pad`. Updated per call. |
| `_sim_active_buf` | WorldOrchestrator | One `u32` per voxel slot (same indexing as `_voxel_buf_a`). Non-zero = grain moved this sim tick = treat as particle. Zeroed by `zero_slots.comp`, written to `1u` by `granular_sim.comp` `try_move()`, read by `surface_extract.comp` binding 4. Grown alongside voxel bufs on capacity expansion. |
| `block_data` | Chunk (holds) / Orchestrator (sole writer) | CPU cache of GPU state. Populated once at chunk generation. NOT updated at runtime. Read only at save time and for `query_voxel_material()` (single-voxel cursor, kept current for player edits only). |

**Why no swap:** The simulation shader dispatches only over active chunks. Non-active chunks are never written to `_voxel_buf_b`, so after a swap the renderer would see stale/zero data for them. Instead, `apply_changelog()` applies each changed voxel to `block_data` and immediately calls `_update_chunk_in_grid()` to upload the affected chunks to `_voxel_buf_a`. A stays canonical at all times; B is a permanent write-only scratch buffer.

**Current state (Phase 5 Step 5 complete):** `WorldOrchestrator` owns `_voxel_buf_a`, `_voxel_buf_b`, `_chunk_table`, `_changelog_buf`, and `_material_sim_buf`. The renderer is a pure consumer. GPU granular simulation is running. `swap_voxel_bufs()` exists but is not called — retained for potential future use in a full-world sim pass.

#### 4.2 The Frame Pipeline (Phase 5+)

```
Simulate:  GranularSimulation dispatches compute shader
               reads  _voxel_buf_a   (canonical state)
               writes _voxel_buf_b   (scratch — only active chunks written)
               appends changelog     (what changed, compact)

Sync:      CPU downloads changelog (~KB, not ~MB)
               dispatches patch_apply.comp → writes entries to _voxel_buf_a
               decodes dirty bitmask → affected chunk positions
               updates active_simulation_mask() per chunk via new_value
               fires chunks_committed → CollisionSystem dirty set
               block_data NOT updated here — synced on demand by CollisionSystem

Render:    RaymarchRenderer binds _voxel_buf_a
               always canonical — no swap, no stale non-active chunks
```

Player mutations (place/remove voxel) still go through `WorldOrchestrator.commit()` — written to `block_data` and uploaded to `_voxel_buf_a` directly. They don't use the scratch-buffer path.

#### 4.3 Write Conflict Resolution — Checkerboard

GPU simulation runs thousands of threads in parallel. Two grains cannot both write to the same empty cell in the same pass. Solution: **checkerboard update**.

- Even frames: only process voxels where `(x + z + frame) % 2 == 0`
- Odd frames: only process voxels where `(x + z + frame) % 2 == 1`

Adjacent cells never update in the same pass. No atomics needed. The cost is that grains update every other frame, which is invisible at 60fps.

#### 4.4 GPU Discovery Scan

The CPU `_on_chunks_committed` scan (32,768 voxels per chunk per frame) is replaced by a GPU compute pass that scans `_voxel_buf_a` for non-NONE simulation voxels and writes candidate positions to an output buffer. The CPU reads back the list and calls `add_candidates`. This lands for free once `_voxel_buf` is no longer owned by the renderer.

#### 4.5 Changelog Buffer

Pre-allocated buffer holding `(global_voxel_idx: u32, new_value: u32)` pairs. An atomic counter at offset 0 tracks how many entries were written this frame. The simulation shader appends to this buffer using `atomicAdd`. At the end of the frame the CPU downloads the counter, then downloads exactly that many entries. Entries are applied to `block_data` in order.

The changelog replaces `VoxelMutation` for simulation-generated changes. Player mutations remain CPU-authored and skip the changelog path entirely.

---

## 3. The Plan

### ~~Phase 0+1 — Kernel Pass~~ ✅ COMPLETE

**What was built:** Simulation infrastructure stripped, renderer became self-sufficient, voxel format expanded to 4 bytes/voxel, collision rebuild de-spiked. Rendering and physics are working and performant.

#### What was delivered

**Deleted:** `GPUSandSimulation.gd`, `DebugOverlay.gd`, `shaders/sand_sim.comp`, `shaders/mark_receives.comp`, `shaders/pressure_propagation.comp`, `shaders/fire_sim.comp`

**ChunkManager signals** replace the 6 `_gpu_sand.*` call sites:

```gdscript
signal slab_invalidated          # fires once after full batch load completes
signal voxel_changed(world_pos: Vector3i)
signal chunks_updated(chunk_positions: Array[Vector3i])
signal voxels_removed(world_positions: Array[Vector3i])
```

`slab_invalidated` fires after `_pending_results.clear()` — not per-chunk. This prevents the O(N²) rebuild that fired on every individual `chunk_added` during initial load.

**RaymarchRenderer** owns `_voxel_buf` and `_chunk_table`. Updated incrementally via `_add_chunk_to_grid`, `_update_chunk_in_grid`, `_remove_chunk_from_grid`. SlabBuilder is gone. Pre-allocates `_params_buf`, `_cam_buf`, `_vp_buf` as persistent RIDs updated in place each frame — no per-frame alloc/free.

**Sparse voxel buffer:** `_voxel_buf` is allocated for loaded chunks only. A free-slot list (`_free_slots`) recycles slots from removed chunks. If the list is empty and capacity exhausted, `compute_helper.buffer_grow()` doubles the buffer. Slot indices in `_chunk_to_slot` / `_slot_to_chunk` / `_chunk_table` are pool-allocated (NOT equal to the flat grid index). Chunk table stays dense (grid_span³ × 4 bytes). This design is intentional for Phase 11 streaming — slots are freed when chunks unload and reallocated when new chunks enter the window. See Phase 6.6 for the rationale against pre-allocation.

**Zero-copy GPU upload:** The 4-byte CPU voxel layout `[mat_id, fill, temp, flags]` is identical to the GPU `u32` layout on little-endian hardware. `block_data` is uploaded directly via `append_array` — no GDScript packing loop.

**MaterialDef** updated to `Phase` / `SimulationClass` / `GravityMode` / `color` (see Pillar 1.1). `SimulationClass` values are sequential (`NONE=0`…`EFFECT=4`); `SIM_BITS` provides the per-class bitmask. `STATIC` removed — static is the absence of simulation. Collidability: `NONE` (non-air) and `GRANULAR` → collidable. **MaterialDB** GPU buffer: packed RGBA at offset 0, simulation at 4, phase at 8, gravity at 12.

**Voxel accessor API** on `Chunk`:

```gdscript
const VOXEL_STRIDE := 4
func get_material_id(local_idx) / get_fill / get_temperature / get_flags
func set_voxel(local_idx, mat_id, fill=255, temp=0, flags=0)
```

~~`build_collision_boxes()` reads material_id at stride-4 intervals.~~ **Removed** — see Collision section below.

**Collision box build — optimized then removed:** `build_collision_boxes` + `_greedy_rectangles` + `_shape_pool` + `_dirty_worker` + `_dirty_collision_queue` were all deleted. See §Collision & Rigid Bodies below for the replacement plan.

**Shader:** chunk-slot caching in DDA loop was already present (~6 table lookups/ray instead of ~200). Debug pixel output updated for new voxel field layout. On-screen perf overlay added to renderer (render ms, frame ms, fps, chunk count).

**Buffer ownership (Phase 5 Step 2, current):**
- `WorldOrchestrator` owns `_voxel_buf_a`, `_voxel_buf_b` (ping-pong), `_chunk_table`, `_free_slots`
- Renderer is a pure consumer — calls `ensure_grid()` + `get_voxel_buf()` each frame; owns nothing

---

### ~~Phase 2 — The Orchestrator Kernel~~ ✅ COMPLETE

**What was built:** `WorldOrchestrator.gd` is the single writer for all voxel mutations. All mutation paths (player add/remove, sphere spawning, explosions) submit through it. A single `chunks_committed` signal replaces the previous per-operation signals.

#### What was delivered

**`WorldOrchestrator.gd`** — new node attached to root `world`. Owns the mutation queue and Active Region list. Called each frame from `_process`.

**`VoxelMutation`** inner class: `world_pos`, `chunk_pos`, `local_idx`, `material_id`, `fill`, `temp`, `flags`, `priority`.

**`submit_mutation(world_pos, mat_id, fill, temp, flags, priority)`** — public API. Called by all systems. Appends to `_pending_mutations`.

**`commit()`** — runs each frame:
1. Resolve conflicts: highest `priority` wins per `world_pos`
2. Write resolved mutations via `chunk.apply_mutation()` (atomic write + O(1) simulation count update)
3. Expand Active Regions for each mutation (`_expand_or_create_region`)
4. Merge overlapping regions (`_merge_regions`) — runs once after all expansions
5. Emit `chunks_committed(positions: Array[Vector3i])`

**Single signal — `chunks_committed`** replaces all previous ChunkManager mutation signals (`voxel_changed`, `chunks_updated`, `voxels_removed`). Two consumers:
- **RaymarchRenderer** → `_on_chunks_updated()` — 128KB reupload per affected chunk
- **ChunkManager** → `_on_chunks_committed()` — appends to `_dirty_collision_queue`

**Active Region system** replaces per-chunk dirty tracking for simulation scheduling:

```gdscript
class ActiveRegion:
	var aabb: AABB       # world-space voxel coordinates
	var settled_frames: int = 0
	var max_cooldown: int = 60
	var id: int
```

- `_expand_or_create_region`: finds any region whose `aabb.grow(1.0)` contains the mutation, expands it and resets its cooldown; otherwise creates a new region. Capped at 256 regions.
- `_merge_regions`: O(N²) pass after all expansions, merges any pair of intersecting AABBs.
- `_update_regions`: increments `settled_frames` each frame, removes regions that exceed `max_cooldown`. Phase 4 will replace this timed cooldown with changelog-driven shrink (AABB clamped to the set of voxels that actually changed).

**ChunkManager** — all 4 mutation methods delegate to the Orchestrator. `_dirty_chunks: Dictionary` replaced with `_dirty_collision_queue: Array[Vector3i]`, sorted by player distance before each rebuild pull (1 chunk/frame). Keeps `slab_invalidated` for the initial batch load.

**Bulk mutation API** — `spawn_sand_sphere` and `explode_at` use chunk-first iteration (AABB → chunk loop → local coords) eliminating per-voxel `_floor_div`/`_pos_mod`. Mutations are built directly (`WorldOrchestrator.VoxelMutation.new()` with pre-computed `chunk_pos`/`local_idx`) and submitted via `submit_batch()` in chunks of 8192 (`BATCH_SIZE_LIMIT`) to keep memory flat for large radii. `submit_mutation_direct(chunk_pos, local_idx, world_pos, ...)` is available for callers with pre-computed coordinates.

**Truly async collision rebuild** — `_dirty_worker` is now `void`. After computing boxes it calls `chunk.call_deferred("_apply_collision_boxes", boxes)` — no `wait_to_finish` in `_process`. `Chunk._apply_collision_boxes` runs on the main thread, applies the boxes, then calls `cm._on_collision_rebuild_complete()` to clear the thread reference. `_process` only ever starts threads; it never waits on them.

**Collision radius** — collision is only for the player (`CharacterBody3D`); projectiles detect hits by sampling `block_data` directly and have no physics collision dependency. An `@export var collision_radius: int = 3` (chunk units) on `ChunkManager` limits collision to a Chebyshev-distance cube around the player:
- Initial load and `regenerate_chunks`: `generate_collision()` only called for in-radius chunks.
- `_on_chunks_committed`: dirty rebuilds only enqueued for in-radius chunks.
- `_process`: tracks `_player_chunk`; on chunk boundary crossing, newly in-range chunks are enqueued for rebuild and newly out-of-range chunks have shapes cleared via `Chunk.clear_collision()`.
- Dirty queue pop: stale entries that left the radius while queued are dropped.

**Signal flow:**
```
Player → ChunkManager.add_voxel()
		   └─ WorldOrchestrator.submit_mutation()

spawn_sand_sphere / explode_at → WorldOrchestrator.submit_batch()

_process (WorldOrchestrator):
  _update_regions()    # cooldown tick, remove settled regions
  commit()
	├─ chunk.apply_mutation()
	├─ _expand_or_create_region() × N
	├─ _merge_regions()
	└─ emit chunks_committed(positions)
		 ├─ RaymarchRenderer._on_chunks_updated()        # GPU reupload
		 └─ ChunkManager._on_chunks_committed()          # append to collision queue (in-radius only)

_process (ChunkManager):
  Phase 1: initial chunk load → slab_invalidated → RaymarchRenderer._init_grid()
  Phase 2: player chunk tracking → enqueue/clear radius boundary chunks
  Phase 3: start next collision rebuild thread (player-distance sorted, 1/frame, in-radius only)
			 worker finishes → call_deferred → Chunk._apply_collision_boxes()
											 → cm._on_collision_rebuild_complete()
```

**Phase 5 integration point:** When GPU simulation arrives, the Orchestrator copies intersecting chunks into staging buffers per active region, dispatches compute shaders, reads back changelogs, and uses changelog AABBs to shrink (not expire) regions. The `chunks_committed` signal and all consumers remain unchanged.

---

### ~~Phase 3 — Simulation Foundation~~ ✅ COMPLETE

**What was built:** The simulation scheduling infrastructure. No physics yet — just the wiring that makes adding a new simulation a zero-orchestrator-change operation.

#### What was delivered

**`Chunk` simulation counts and mask** — per-chunk O(1) tracking of which `SimulationClass` values are present:
- `_simulation_counts: PackedInt32Array` — one entry per non-NONE SimulationClass (index = sim_class - 1)
- `_simulation_mask: int` — private bitmask; bit set iff count > 0 for that class
- `has_simulation(sim_class)`, `active_simulation_mask()` — public accessors; mask never exposed directly
- `apply_mutation(local_idx, mat_id, fill, temp, flags, old_sim, new_sim) -> bool` — atomic write + count update + early-out
- `_transition_simulation(old_sim, new_sim)` — O(1) count dec/inc and mask bit management
- `rebuild_counts_and_mask(material_simulation: PackedInt32Array)` — one-time init after generation; also a debug drift verifier. Must `resize()` before `fill(0)` — GDScript's `fill()` is a no-op on a zero-length array, and background worker threads can call this before deferred init.

**Initialization order lessons:**
- MaterialDB._ready() runs after WorldOrchestrator._ready() due to scene order; `call_deferred("_build_material_cache")` in both WorldOrchestrator and SimulationRuntime ensures the cache is built after MaterialDB populates
- `GranularSimulation._ready()` calls `get_parent().register_simulation(self)` — parent-scan from `SimulationRuntime._ready()` returns empty children list because GDScript children are not yet initialized when parent's `_ready()` fires
- SimulationRuntime must be a child of WorldOrchestrator in world.tscn (`parent="WorldOrchestrator"`); GranularSimulation must be a child of SimulationRuntime (`parent="WorldOrchestrator/SimulationRuntime"`) — these are scene-relative paths

---

### ~~Phase 4 — Granular Solver (falling sand)~~ ✅ COMPLETE

**What was built:** `GranularSimulation` is running and sand visibly falls. The end-to-end pipeline — discovery scan → `add_candidates` → `step()` → mutations → commit — is confirmed working. The region-based scheduler was replaced with a minimal lifecycle-driven scheduler.

#### What was delivered

**`WorldOrchestrator`** — active region system removed entirely:
- `ActiveRegion`, `_regions`, `_region_id_counter`, `MAX_REGIONS` gone
- `_expand_or_create_region`, `_merge_regions`, `_update_regions` gone
- `_process` calls `_runtime.run()` (no args) then `commit()`
- `commit()` emits `chunks_committed` for all affected chunks unconditionally

**`simulations/voxel_simulation.gd`** — new interface:
```gdscript
func get_simulation_class() -> int
func add_candidates(positions: Array[Vector3i]) -> void  # runtime delivers discovered voxels
func step(ctx: SimulationContext) -> void
func has_work() -> bool                                  # false = runtime removes from active set
```

**`simulations/simulation_runtime.gd`** — full rewrite:
- Subscribes to `chunks_committed` directly; on each commit: scans affected chunks once (mask fast-path), builds per-simulation candidate arrays, calls `add_candidates()` for non-empty results
- Owns `_active: Array[VoxelSimulation]` — simulations currently with work
- `run()`: calls `step()` for each active simulation, removes those where `has_work()` is false
- No regions, no AABBs, no candidate dictionaries, no `run(regions)` parameter

**`simulations/granular_simulation.gd`** — full rewrite:
- `_frontier: Dictionary` (Vector3i → true, set semantics — no duplicate positions)
- `add_candidates`: merges positions into frontier (idempotent — duplicate positions from residual re-scan are harmless)
- `step`: for each frontier position — try DOWN, then four diagonal-down directions in randomised order; successful destination enters `next`; grains that cannot move exit the frontier naturally (no explicit stable flag needed)
- `has_work`: checks `_frontier.is_empty()`

**Stability is implicit in the frontier:** a grain that cannot move is simply never added to `next`. It is skipped every subsequent frame until a neighbour changes, which fires `chunks_committed`, rescans the chunk, and re-adds the grain via `add_candidates`. No explicit "stable" flag, no separate sleeping list.

**Diagonal movement:** when blocked directly below, the simulation tries `(below + dx, 0, 0)` and `(below − dx, 0, 0)`, then `(below, 0, dz)` and `(below, 0, −dz)` where `dx` and `dz` are chosen randomly each step. This prevents directional bias and produces a natural angle of repose in all four horizontal directions.

**`simulations/simulation_context.gd`** — `positions()` removed; `_init` no longer takes a candidates array.

**Signal flow (current):**
```
_process (WorldOrchestrator):
  _runtime.run() → mutations
	└─ for each sim in _active:
		 sim.step(ctx) → ctx.flush_mutations()
		 if not sim.has_work(): remove from _active
  _pending_mutations.append_array(mutations)
  commit()
	├─ conflict resolution (priority wins per world_pos)
	├─ chunk.apply_mutation() ← atomic: write + O(1) count/mask update
	└─ emit chunks_committed(positions)
		 ├─ RaymarchRenderer._on_chunks_updated()        # GPU reupload
		 ├─ ChunkManager._on_chunks_committed()          # collision queue
		 └─ SimulationRuntime._on_chunks_committed()     # ← NEW: discovery scan
			  └─ _scan_chunk() × affected chunks with mask != 0
			  └─ sim.add_candidates(positions)
			  └─ _active grows if sim wasn't already active
```

**Why the frontier Dictionary sidesteps re-discovery:**
When sand falls from chunk A to chunk B, `chunks_committed` fires for chunk B. The runtime scans it, finds sand at position P (the destination), and calls `add_candidates([P])`. `GranularSimulation.add_candidates` does `_frontier[P] = true`. But P is already in `_frontier` because `step()` put it there. The Dictionary set insert is a no-op — no duplicate work, no new task, no coverage tracking needed.

**Discovery scan — deferred to Phase 5:** The `_on_chunks_committed` scan currently walks 32,768 voxels per affected chunk on the main thread. A GPU compute pass could replace this (the data is already in `_voxel_buf`), but the GPU scan requires exposing buffer ownership from `RaymarchRenderer` — the same restructure Phase 5 needs anyway. Doing it now would be touching the infrastructure twice. Short-term mitigation: only scan chunks where `active_simulation_mask()` actually changed (new simulation type entered the chunk), skipping residual sand-moves within already-tracked chunks.

---

### ~~Phase 5 — GPU Simulation Pipeline~~ ✅ COMPLETE

**Goal:** Move simulation onto the GPU. Granular is the test case. A working GPU granular solver proves the pipeline before any new simulation types are added.

Fluid (water), fire, and gas come after this phase — they are new simulations on top of proven GPU infrastructure, not part of building it.

#### ~~Step 1 — Buffer ownership restructure~~ ✅ COMPLETE

`_voxel_buf`, `_chunk_table`, and `_free_slots` moved from `RaymarchRenderer` to `WorldOrchestrator`. The renderer holds a `_orchestrator: WorldOrchestrator` reference and calls `ensure_grid()` + `get_voxel_buf()` / `get_chunk_table()` at the top of each `_render()` call. It owns no buffers. The orchestrator connects to `slab_invalidated` directly and calls `_update_chunk_in_grid()` inside `commit()` before emitting `chunks_committed` — GPU state is in sync before any listener runs. `_on_chunks_updated` is removed from the renderer entirely.

#### ~~Step 2 — Ping-pong buffers~~ ✅ COMPLETE

`WorldOrchestrator` allocates `_voxel_buf_b` alongside `_voxel_buf_a` at grid init, with identical layout and capacity. Both buffers are grown together whenever capacity is exhausted. CPU mutations (`_update_chunk_in_grid`) write only to `_voxel_buf_a` — the simulation shader's A→B copy handles non-simulated voxels when it dispatches. `swap_voxel_bufs()` swaps the two RIDs (O(1)). Public API: `get_voxel_buf()` → `_voxel_buf_a`, `get_voxel_buf_b()` → `_voxel_buf_b`, `swap_voxel_bufs()`. The renderer is unchanged — it always calls `get_voxel_buf()` and gets whichever buffer is currently canonical.

#### ~~Step 3 — Changelog buffer~~ ✅ COMPLETE

Pre-allocated GPU buffer at `_changelog_buf`. Layout: `u32 count` at offset 0, then `(flat_voxel_idx: u32, new_value: u32)` pairs (8 bytes each). `MAX_CHANGELOG_ENTRIES = 65536` → 512 KB total.

`flat_voxel_idx = slot * voxels_per_chunk + local_voxel_word_idx`. A `_slot_to_chunk: Dictionary` (slot → Vector3i) is maintained alongside `_chunk_to_slot` to make this reversible on the CPU side.

Public API on `WorldOrchestrator`:
- `get_changelog_buf() -> RID` — bind this as a storage buffer in the simulation shader
- `reset_changelog()` — writes 0 to offset 0; call before each simulation dispatch
- `apply_changelog() -> Array[Vector3i]` — downloads the counter, then the entries, calls `chunk.apply_mutation()` for each, returns affected chunk positions

`apply_changelog` uses `apply_mutation` (not raw byte writes) so `active_simulation_mask` and `_simulation_counts` stay accurate. The returned positions are used by the caller to emit `chunks_committed` after the swap.

#### ~~Step 4 — Granular compute shader (`shaders/granular_sim.comp`)~~ ✅ COMPLETE

`local_size = (8, 8, 8)`. Dispatched `(4, 4, 4)` workgroups per chunk → 32³ threads per dispatch. Per thread:

1. **Copy A→B** unconditionally at top of `main()` — ensures all non-moving voxels are preserved in B without a separate copy pass.
2. **Skip** if `mat_id == 0` or `material_sim[mat_id] != SIM_GRANULAR`.
3. **Checkerboard skip**: `(local_pos.x + local_pos.z) % 2 != frame % 2` — alternates which (x+z) parity updates each frame, preventing diagonally adjacent grains from competing for the same destination in the same pass.
4. **Try fall down**: reads `voxels_a` at `gpos + (0,-1,0)` via `world_to_flat()` (handles cross-chunk via `chunk_table`). If air: `write_b_log(dest, sand)` + `write_b_log(source, air)` → both go to B and are appended to changelog via `atomicAdd`.
5. **Try 4 diagonal-down directions**: randomised axis/direction via Wang hash seeded by `frame * 1013904223 + my_flat`. Same write pattern.
6. **Stable** → default A→B copy from step 1 is the final result.

`world_to_chunk` uses `floor(float(pos) / float(chunk_size))` — matches Godot's floor division for negative coordinates, consistent with `raymarch.comp`.

`_material_sim_buf` added to `WorldOrchestrator`: 256 u32s (1 KB) mapping material_id → SimulationClass. Uploaded in `_build_material_cache()`, exposed via `get_material_sim_buf()`. Bound as binding 4 in the simulation shader.

#### ~~Step 5 — GranularSimulation becomes a shader dispatcher~~ ✅ COMPLETE

`_frontier: Dictionary` replaced by `_active_chunks: Dictionary` (chunk_pos → true). `add_candidates` converts world positions to chunk positions via `voxel_to_chunk_position`. `has_work` checks `_active_chunks.is_empty()`.

`step(ctx)` GPU path:
1. `reset_changelog()` — zero counter
2. Build uniforms once (voxel_buf_a/b, chunk_table, params_buf, material_sim_buf, changelog_buf)
3. For each active chunk: `_upload_params(chunk_pos, ...)` + `_compute.run(..., 4, 4, 4)`
4. `submit()` + `sync()` — wait for all dispatches to complete
5. `apply_changelog()` → updates block_data + re-uploads changed chunks to `_voxel_buf_a`; returns affected chunk positions
6. `chunks_committed.emit(changed)` — triggers collision queue + discovery scan
7. Drop chunks with no changelog entries (stable); rest remain active next frame

**No swap:** `_voxel_buf_b` is write-only scratch. `apply_changelog()` writes results back to `_voxel_buf_a` directly (see §4.1).

**Checkerboard stability fix:** A chunk is only evicted from `_active_chunks` after **two** consecutive zero-entry dispatch frames. The checkerboard skips one parity per frame (grains where `(x+z) % 2 != frame % 2`). Two stacked grains at the same `x, z` share the same parity — one idle frame means "wrong parity this frame," not "truly stable." Without the two-frame guard, same-parity stacks froze after one uneventful dispatch. `_active_chunks` now stores `int` (consecutive idle frames) instead of `bool`. Movement resets the counter to 0; `idle >= 2` drops the chunk.

Stability at chunk granularity: if a chunk produces zero changelog entries, all its granular grains are stuck and it exits `_active_chunks`. It re-enters when a neighbour change fires `chunks_committed` and discovery finds granular voxels again — same wake-up semantics as the CPU frontier, just coarser.

`SimulationRuntime.run()` now snapshots `_active` before iterating (`to_run = _active.duplicate()`). GPU sims emit `chunks_committed` inside `step()`, which re-enters `_on_chunks_committed → _active.append(...)`. The snapshot prevents iterator invalidation; a post-loop merge picks up any sims added mid-frame.

`WorldOrchestrator.is_grid_valid()` added — returns true when `_voxel_buf_a`, `_chunk_table`, and `_material_sim_buf` are all valid RIDs. Guards the dispatcher against dispatching before the first `ensure_grid()` completes.

**Active chunk cap:** `GranularSimulation.add_candidates()` limits `_active_chunks` to `MAX_ACTIVE_CHUNKS = 64` (raised from 20 in Phase 6.6 — see that section). When at capacity, new chunks are skipped until an existing chunk goes idle via the idle guard.

#### ~~Step 6 — Fast discovery: chunk-mask path~~ ✅ COMPLETE

**Problem (observed in profiler):** `SimulationRuntime._on_chunks_committed` currently calls `_scan_chunk` which walks all 32,768 voxels of each committed chunk — measured at ~20ms for 6 chunks. `add_candidates` then calls `voxel_to_chunk_position` ~931 times to convert the resulting voxel positions back to chunk coordinates. Total: ~65ms across `_on_chunks_committed` + `_scan_chunk` + `add_candidates` + `voxel_to_chunk_position`.

**Fix:** Replace the per-voxel scan with a chunk-mask check. Each `Chunk` already maintains `_simulation_mask` — a bitmask set whenever a simulation-class voxel is written. `_on_chunks_committed` can check the mask (O(1) per chunk) and pass the chunk position directly to `add_candidates`, skipping all per-voxel iteration. Extend to vertical neighbours (chunk_pos ± Y) because granular voxels fall between chunks.

```gdscript
func _on_chunks_committed(chunk_positions: Array[Vector3i]) -> void:
    var granular_bit := MaterialDef.SIM_BITS[MaterialDef.SimulationClass.GRANULAR]
    if not _simulations.has(granular_bit):
        return
    var sim: VoxelSimulation = _simulations[granular_bit]
    var candidates: Array[Vector3i] = []
    for cp in chunk_positions:
        for neighbour in [cp, cp + Vector3i(0, 1, 0), cp + Vector3i(0, -1, 0)]:
            var chunk: Chunk = _chunk_manager.chunks.get(neighbour)
            if chunk and (chunk.active_simulation_mask() & granular_bit):
                candidates.append(neighbour)  # chunk position, not voxel
    if not candidates.is_empty():
        sim.add_candidates(candidates)
        if not _active.has(sim):
            _active.append(sim)
```

`GranularSimulation.add_candidates` is updated to accept chunk positions directly (not voxel positions), storing them in `_active_chunks` with no conversion needed.

**Expected impact:** Collapses ~65ms of per-voxel GDScript into ~0.1ms of O(chunks) bitmask checks. The `_scan_chunk`, `voxel_to_chunk_position`, and `_floor_div` hot-path entries disappear from the profiler.

**Note on discovery completeness:** The mask approach is sufficient for granular simulation because we dispatch per chunk, not per voxel. Fluid and gas simulations follow the same pattern. If a future simulation needs per-voxel candidate filtering (e.g. fire spread to specific material types), a GPU scan pass (Step 8) handles it.

#### ~~Step 7 — Single changelog download round-trip~~ ✅ COMPLETE

**Problem (observed in profiler):** `apply_changelog()` currently makes two GPU round-trips: `download_range(offset=0, size=4)` for the count, then `download_range(offset=4, size=count*8)` for the entries. Each round-trip shows up as ~1.5ms `download_range` + ~1ms `sync` in the profiler.

**Fix:** Download the full changelog buffer in one call (`download_range(0, 4 + MAX_ENTRIES * 8)`), read the count from byte 0, then decode entries from the same byte array. One `sync`, one download.

```gdscript
func apply_changelog() -> Array[Vector3i]:
    var raw := _compute.download_range(_changelog_buf, 0, 4 + MAX_CHANGELOG_ENTRIES * 8)
    var count := raw.decode_u32(0)
    if count == 0: return []
    count = min(count, MAX_CHANGELOG_ENTRIES)
    for i in range(count):
        var off := 4 + i * 8
        var flat_idx := raw.decode_u32(off)
        var new_value := raw.decode_u32(off + 4)
        # ... apply as before
```

**Expected impact:** Halves GPU sync overhead — `download` + `sync` each appear once in the profiler instead of twice.

**Tradeoff:** Downloads the full 512 KB buffer every frame the sim runs, regardless of entry count. For sparse active frames (few moving grains), the old approach was cheaper. Acceptable because the 512 KB transfer is dominated by the `sync` latency anyway — the actual data volume is negligible vs. PCIe bandwidth.

#### Step 8 — GPU discovery scan (optional)

A small compute scan shader dispatched over committed chunk slots, writing simulation-class voxel positions to an output buffer. The CPU reads back the list and calls `add_candidates`. Replaces Step 6's CPU mask check with a true GPU scan.

**When to implement:** Only if Step 6 proves insufficient — e.g., when fire or thermal simulation needs per-voxel granularity, or when `_on_chunks_committed` is still measured as a bottleneck after Step 6. Step 6 should reduce discovery to near-zero; Step 8 is a contingency.

#### ~~Step 9 — CPU simulation path removed~~ ✅ COMPLETE

`SimulationContext` deleted. `step(ctx)` → `step()` across `VoxelSimulation`, `GranularSimulation`, and `SimulationRuntime`. `run()` is now `void` — no mutations collected or returned. `GranularSimulation` owns its own `_frame: int` counter (incremented at the top of `step()`) replacing the former `ctx.frame()` call. `WorldOrchestrator._process()` calls `_runtime.run()` as a void statement.

#### ~~Step 10 — Targeted voxel buffer writes (GPU patch shader)~~ ✅ COMPLETE

**Problem:** `apply_changelog()` called `_update_chunk_in_grid()` for every affected chunk, re-uploading the full 128 KB chunk even when only a handful of voxels changed.

**What was tried and rejected:**
- Per-entry 4-byte `buffer_update` calls: ~28 000 GDScript call-overhead hits were slower than 25 full-chunk uploads.
- Dirty-range slices: `PackedByteArray.slice()` allocations per chunk exceeded the bandwidth saving.

**What was built — `shaders/patch_apply.comp`:**
A 64-thread compute shader that reads the simulation changelog `(flat_idx u32, new_value u32)[]` and writes each entry directly into `_voxel_buf_a`. Dispatched with `ceil(count / 64)` workgroups after each sim pass -- 1 dispatch replaces N chunk re-uploads.

`apply_changelog()` dispatches `patch_apply.comp` before the CPU block_data sync loop. `sync_block_data_from_buf()` (extracted from `apply_changelog`) is a shared helper also called by `EditDispatcher._drain_edit_changelog()` -- no `_update_chunk_in_grid` calls anywhere in the sim path.

**GLSL gotchas discovered:**
- `flat` is a reserved GLSL interpolation keyword -- never use it as a variable name in shaders.
- Non-ASCII characters (em dashes, etc.) in GLSL comments can silently corrupt tokenization on some drivers. Use ASCII-only shader source.

#### ~~Step 11 — Multi-chunk dispatch~~ ✅ COMPLETE

Replaced the per-chunk `_params_buf` (44 bytes, uploaded and dispatched N times) with:
- `_global_params_buf` (32 bytes: chunk_size, grid_min/span, frame) — uploaded once per step
- `_chunk_list_buf` (N × 12 bytes: chunk_pos.xyz per active chunk) — uploaded once per step, grows on demand

Dispatch is now a single `_compute.run(_pipeline, uniforms, active_count * 4, 4, 4)`. The shader derives its chunk via `gl_WorkGroupID.x / 4` and local X workgroup via `gl_WorkGroupID.x % 4`. N GDScript→GPU round-trips collapsed to 1 regardless of active chunk count.

#### ~~Step 12 — Fixed simulation timestep~~ ✅ COMPLETE

`SimulationRuntime.run(delta)` now accumulates `delta` and fires `_tick()` at a fixed `SIM_HZ = 60.0` ticks/second with a `MAX_CATCHUP = 1` guard (reduced from 3 to prevent spiral-of-death under heavy load — at low FPS, running multiple catch-up ticks compounds the slowdown). `WorldOrchestrator._process(delta)` passes `delta` through. Checkerboard parity derives from simulation tick count, not render frames — sand falls at the same physical rate at 30fps or 160fps.

#### ~~Step 13 — Renderer uniform set caching~~ ✅ COMPLETE

`RaymarchRenderer` caches `_render_uniform_set: RID` and the RIDs it was built from (`_cached_voxel_rid`, `_cached_chunk_table_rid`). The set is rebuilt only when those RIDs change (grid rebuild or window resize). `compute_helper` gained `create_uniform_set()` and `run_with_set()` to support caller-owned sets. Every steady-state render frame skips 8 `RDUniform` allocations and the `uniform_set_create` driver call.

---

---

### ~~Phase 6.5 — Chunk Boundary "Pop" Bug Fix~~ ✅ COMPLETE

**Problem:** After tightening active chunk logic (`MAX_ACTIVE_CHUNKS` cap, `MAX_CATCHUP=1`, edit queue budget), sand dropped from height would freeze at chunk boundaries then suddenly "pop" down.

**Root cause — all-air chunks have no GPU slot:**

`WorldOrchestrator._init_grid()` only called `_add_chunk_to_grid()` for chunks in `cm.chunks` — so all-air grid positions had `slot = -1` in the chunk table. The shader's `world_to_flat()` returns `-1` for positions in those chunks (`if (slot < 0) return -1;`), and sand at the bottom of a real chunk above an all-air void **cannot fall**. The sand accumulates frame after frame, the chunk produces no changelog entries, gets evicted after the idle guard, and when the pile finally reaches a chunk with a valid slot, it drops suddenly.

**Fix — generate_chunks creates Chunk nodes for all grid positions (including all-air):**

`ChunkManager.generate_chunks()` already generates a Chunk node for every grid position — including all-air chunks. This means `cm.chunks` has entries for all 512 positions at init time, so `_init_grid` naturally allocates a slot for each one. Sand falls into all-air regions because those regions have a valid (zero-filled) slot in the voxel buffer.

**Fix B — Per-class masked dict:**

`_extract_changed_chunks` used `masked: Dictionary` (chunk_pos → bool) to deduplicate `mark_simulation_active` calls — but it evaluated the simulation class from only the first changelog entry for each chunk. If the first entry was a clear (air, sim=NONE) but later entries had sand (sim=GRANULAR), the mask was never set. Replaced with `masked_bits: Dictionary` (chunk_pos → int bitmask) so each simulation class is marked independently per chunk per tick.

---

### Phase 6.6 — Architecture Corrections ✅ COMPLETE

**Pre-allocation reverted to sparse:**

An intermediate version replaced the sparse slot pool (`_free_slots` + `buffer_grow`) with full grid pre-allocation — one contiguous buffer sized for ALL grid positions regardless of how many are loaded. This was reverted because:

- Pre-allocation contradicts the streaming design (Phase 11). A sliding-window streaming world cannot pre-allocate for the entire possible world extent.
- For a 256×256×256 world at chunk_size=32: 512 chunks × 32³ × 4 bytes × 2 buffers = 128 MB committed at startup, even for sparse terrain.
- The free-slot pool infrastructure (`_free_slots`, `buffer_grow`) is exactly what Phase 11 needs to recycle slots as chunks load and unload. Removing it created unnecessary debt.

Current state: sparse allocation with `_free_slots: Array[int]` pool and `buffer_grow` growth. Buffer starts at `max(64, next_power_of_two(loaded_count))` chunks and doubles when exhausted. Slot indices in `_chunk_to_slot` / `_slot_to_chunk` / `_chunk_table` are pool-allocated, NOT identical to the flat grid index.

**`_gpu_dirty` removed:**

`_gpu_dirty: Dictionary` tracked which chunks had GPU state ahead of `block_data`, intended for save-time sync. It was removed because:

- It grew monotonically during gameplay (never cleared at runtime) — every chunk touched by simulation was permanently marked dirty.
- This caused every `query_voxel_material` call to route through a GPU `download_range` sync when we had a "GPU-primary, check dirty first" read path.
- For save time, `sync_chunk_to_cpu()` can download all loaded chunks directly without the dirty index — or the save system can simply download all chunk slots. The dirty set optimization is not needed until save is implemented.

**`MAX_ACTIVE_CHUNKS` raised to 64:**

The cap at 20 prevented large sand spheres from activating all their chunks simultaneously. A radius-20 sphere spans ~27 chunk positions; at cap 20, the outermost 7 chunks never simulated. Raised to 64, which covers any sphere radius up to ~50 voxels without hitting the cap. At 64 active chunks the single multi-chunk dispatch (Step 11) still fires once per tick — GPU overhead scales with active_count workgroups, not GDScript calls.

**`MAX_CHANGELOG_ENTRIES` raised to 131072 (1 MB):**

The original 65536 cap overflowed at ~70k entries when simulating a large sand sphere, clamping and discarding state. Profiling showed the engine can handle ~102k changelog entries per tick before dropping to 20fps. Raised to 131072 to eliminate the overflow warning at realistic sphere sizes.

**`downloads_per_frame` floor is 1:**

`Voxel/downloads_per_frame` shows 1 at idle — this is the cursor readback in `RaymarchRenderer._render()` (`download_range(_cursor_buf, 0, 32)`). It fires every rendered frame regardless of simulation activity. This is correct: 32 bytes for player crosshair targeting. Not a leak or bug. Simulation changelog downloads appear as transient spikes above 1 when active.

**RaymarchRenderer uniform set free_rid error fixed:**

When `_init_grid` calls `_free_voxel_buffers`, Godot's RD auto-invalidates any uniform set that references the freed voxel buffer. The renderer's `_render_uniform_set` still held a non-zero RID (`is_valid()` returns true), but the underlying resource was gone — calling `free_rid` on it produced "Attempted to free invalid ID." Fix: only call `free_rid(_render_uniform_set)` when the buffer RIDs haven't changed (e.g. window resize). When the voxel buffer was rebuilt, the RD already freed it — just drop the reference with `_render_uniform_set = RID()`.

**Micro-voxel direction:**

The visual size of voxels is controlled by `voxel_scale` (currently 0.5m/voxel), not by `chunk_size`. To achieve a micro-voxel aesthetic (fine-grained world), reduce `voxel_scale` to 0.05–0.1 and scale the player proportionally larger relative to the world — or keep player size and treat the finer grid as higher-detail terrain. `chunk_size = 32` stays optimal for streaming granularity. Increasing chunk_size would reduce chunk count but make each simulation dispatch, collision rebuild, and slot upload 8× heavier (64³ vs 32³) with no visual benefit.

---

### Current Performance Profile (post-Phase 6.6)

Measured at ~102k changelog entries/tick (large sand sphere, 64 active chunks):

| Function | Time | Notes |
|---|---|---|
| `WorldOrchestrator._process` | 59ms | dominated by simulation |
| `GranularSimulation.step` | 59ms | includes shader dispatch + changelog |
| `WorldOrchestrator.apply_changelog` | 57ms | changelog download + extract + patch |
| `WorldOrchestrator._extract_changed_chunks` | **53ms** | **dominant bottleneck — eliminated in Phase 6.7** |
| `sync` | 2.7ms | 3 calls total (sim + patch + renderer) |
| `download_range` | 2.4ms | 3 calls (2-phase changelog + cursor) |
| `download` | 2.1ms | renderer color+depth readback |
| `_render` | 5ms | raymarcher |

**Key finding:** `_extract_changed_chunks` consumes 53ms of a 59ms frame budget — a pure GDScript loop iterating over up to 131k `(flat_idx, new_value)` pairs doing dictionary lookups per entry to resolve slot→chunk_pos and update simulation masks. The GPU itself is cheap; the CPU post-processing is the bottleneck. Eliminated in Phase 6.7 by the GPU dirty-chunk bitmask.

---

### Phase 6.7 — GPU Dirty-Chunk Bitmask ✅ COMPLETE

**Problem:** `_extract_changed_chunks` iterated every changelog entry in GDScript (up to 131k per tick) to build the changed-chunk list. At ~102k entries this consumed 53ms out of a 59ms frame budget.

**Solution:** Added a 64-byte `dirty_chunks` buffer (binding 7) to `granular_sim.comp`. When a voxel successfully moves, two `atomicOr` calls mark both the source and destination chunk slots in the bitmask:

```glsl
void mark_dirty(int flat_idx, int vpc) {
	int chunk_slot = flat_idx / vpc;
	atomicOr(dirty_chunks.data[uint(chunk_slot) >> 5u], 1u << (uint(chunk_slot) & 31u));
}
```

After `submit()+sync()`, `GranularSimulation` downloads the 64-byte bitmask (no additional GPU stall — GPU already synced), decodes it by iterating 16 u32s, and maps each set bit to a chunk_pos via `_slot_to_chunk`. This replaces the entire `_extract_changed_chunks` call.

`WorldOrchestrator.apply_changelog()` now returns void and no longer calls `_extract_changed_chunks`. The `extract_changed_chunks()` public alias remains for `EditDispatcher` (edit changelog has small counts, not a bottleneck).

**Per-tick CPU work after Phase 6.7:**
- 4 bytes downloaded (changelog count)
- 64 bytes downloaded (dirty bitmask, post-sync memcopy)
- 16 u32 iterations to decode bitmask (≤64 set bits)
- 0 per-voxel GDScript iterations

**Dirty bitmask sizing:** 64 bytes covers 512 slots (current fixed world capacity). If `buffer_grow` ever pushes capacity past 512, `_dirty_chunks_buf` and the shader binding must grow accordingly (tracked as a future TODO in `granular_simulation.gd`).

**Simulation mask updates:** The previous `_extract_changed_chunks` called `chunk.mark_simulation_active(sim)` per new material seen. With the bitmask, GranularSimulation is self-scheduling — `add_candidates` via `SimulationRuntime._on_chunks_committed` still fires and merges into the active set, but GranularSimulation no longer depends on it for correctness. Sand falling into an empty chunk propagates naturally: the destination chunk appears in the bitmask → included in next tick's active set → its sand appears in next tick's bitmask → chain continues until all grains settle.

**Opt-2 implemented:** `atomicCompSwap` on `voxels_b` eliminates the same-parity destination race. A pre-dispatch `zero_slots.comp` pass zeros all active chunk slots in voxels_b so `0u` reliably means "unclaimed." Each grain calls `try_move(df, my_voxel, my_flat, vpc)` which does `atomicCompSwap(voxels_b[df], 0u, my_voxel)` — only the winning thread logs both entries and zeros the source. Losers fall through to the next direction. Known edge case: cross-chunk diagonal destinations in non-active chunks are not zeroed and may have stale B; those races are rare (require same-parity grains both at y=0 of an active chunk targeting the same diagonal cell in a non-active chunk below) and cause at most 1-grain losses rather than the catastrophic in-chunk losses fixed here.

---

### Planned Optimizations

#### ~~Opt-1 — GPU dirty-chunk bitmask~~ ✅ COMPLETE (Phase 6.7)

#### ~~Opt-2 — Atomic destination claiming via `atomicCompSwap`~~ ✅ COMPLETE

**Problem:** Two same-parity grains 2 apart (e.g., at (x,z) and (x+2,z)) could both read A[dest]=air and both call `write_b_log(dest, sand)` in the same pass. Last write wins; both sources were zeroed → one grain lost per collision. The checkerboard prevents adjacent-cell races but NOT same-parity-2-apart races.

**Fix:** `zero_slots.comp` zeros all active chunk slots in `voxels_b` before each dispatch (separate submit+sync). `granular_sim.comp` replaces the blind dest write with `atomicCompSwap(voxels_b[df], 0u, my_voxel)`. Only the winning thread logs both entries and zeros the source. Losing threads fall through to the next direction — the grain does not disappear, it just tries a different path.

**Fluid compatibility:** Fluid will use `atomicAdd` on `voxels_b` for fill accumulation, running in a separate dispatch after granular settles. Both use `coherent buffer VoxelBufB` with different atomic operations on non-overlapping slots at different sim phases — no conflict.

**Remaining edge case:** Diagonal moves crossing into non-active chunk slots are not zeroed by `zero_slots`, so CAS may incorrectly fail for those destinations if B has stale data. Rare in practice (requires specific geometry at chunk boundaries) and causes at most 1-grain losses rather than mass loss.

#### Opt-3 — Async changelog: overlap GPU sim with CPU patch-apply

Current pipeline: dispatch sim → submit → sync (stall) → download count → patch GPU → submit → sync (stall). Two serial stalls per tick. Alternative: double-buffer the changelog so while GPU runs frame N, CPU processes frame N-1's already-downloaded data. Eliminates both stalls from the critical path with a one-frame lag in `chunks_committed` propagation (acceptable — discovery only needs "which chunks changed," not exact values). Worthwhile when changelog counts grow further with fluid.

#### Opt-4 — GPU-primary chunk discovery

After Opt-1's bitmask is available, a small GPU pass could also write simulation-mask candidates to an output buffer so `_on_chunks_committed` never needs to touch `block_data`. Combined with Opt-2+3, this would make the entire sim loop GPU-resident with only ~68 bytes of CPU downloads per tick. Scales naturally to fluid/gas where changelog counts will be 4–8× higher than granular.

---

### Near-Term Roadmap (pre-Fluid)

The following work is prioritised before Phase 7 (Fluid). The rationale: the world needs to feel like a place you can walk around and interact with before adding a complex second simulation. Discovering editing-system bugs while debugging fluid is the worst time to find them.

**Simulation correctness status:** Granular simulation is stable. `atomicCompSwap` (Opt-2) eliminates in-chunk destination races; `zero_slots.comp` pre-pass zeros active chunks AND their below-neighbors so cross-chunk falls also use CAS correctly. `MAX_ACTIVE_CHUNKS` cap removed — the actual limit is the dirty bitmask (512 slots = full world). Sand loss should now be negligible.

#### Step 1 — Noise Terrain + Chunk Streaming

Restore noise-based terrain generation and wire it into a streaming chunk loader centred on the player. This replaces the current fixed-grid full-world load.

**Terrain generation:**
- 3D OpenSimplex / FastNoiseLite noise for cave/overhang structure, 2D heightmap for surface
- `TerrainGetter` (already exists as `@tool` script) feeds chunk-request callbacks
- Chunks generate on background threads (existing `_dirty_worker` path), upload to GPU when ready

**Streaming window:**
- Fixed render window: e.g. 16×8×16 chunks
- As the player moves, `_grid_min` shifts and the chunk table updates incrementally (see Phase 11 design — streaming is now moved up in priority)
- Chunks that leave the render window are saved to disk and their GPU slots freed

**Player scale:** `voxel_scale` drops from 0.5 m to ~0.05–0.1 m/voxel. The player model scales up so they're ~18 voxels tall. At this scale sand looks like sand rather than a Minecraft block. All existing GPU buffer math is scale-agnostic — no shader changes needed.

#### Step 2 — Simulation Distance

Define a simulation radius centred on the player (e.g. radius 2 chunks = 5×5×5 = 125 chunks max). Only chunks within this radius are:
- Ticked by `GranularSimulation` (and future fluid/fire)
- Eligible for collision rebuilds
- Included in `chunks_committed` propagation to `SimulationRuntime`

Chunks outside the radius are rendered normally but frozen — their GPU data is live, they just don't tick. `GranularSimulation.add_candidates()` already gates on chunk position being loaded; the streaming system filters candidates against the sim radius before passing them in. `ChunkManager` emits a `chunk_unloaded` signal so simulations can evict positions that leave the window.

#### ~~Step 3 — CollisionSystem~~ ✅ COMPLETE (GPU-primary, see §Collision & Rigid Bodies)

The per-slice / block_data / changelog approach in the original plan was superseded by GPU surface extraction. See the Collision section above for what was built. Brief summary of what changed from the original plan:
- block_data is NOT synced for collision — `extract_surface_voxels()` downloads only surface voxels (~2KB vs 128KB)
- Shapes are NOT per-slice `shape_owner` nodes — PhysicsServer3D RIDs directly on `Chunk.get_rid()`, no scene-tree nodes
- Shapes persist once built — not cleared on interest expiry
- Settle gate is NOT frame-count-based — a chunk rebuilds the first frame it's dirty, in-interest, and not committed this frame
- Open issue: falling granular columns still get collision shapes (see §Open problem)

#### Step 3b — CollisionSystem (original plan, kept for reference)

**The constraint:** Disabling collision entirely was measured to triple performance. The collision rebuild AND the Godot physics broadphase cost of active shapes both contribute. The solution is not to rebuild less often — it is to rebuild fewer slices AND to have active shapes only where something physical is actually present.

**Ownership model:**

| Concern | Owner |
|---------|-------|
| `StaticBody3D`, shape storage | `Chunk` |
| Greedy merge mechanics | `Chunk` |
| Rebuild decision (when, for which slices) | `CollisionSystem` |
| block_data sync authorization and budget | `WorldOrchestrator.request_chunk_sync()` |

**`CollisionSystem`** — new thin node (`collision_system.gd`):
- `_voxel_dirty: Dictionary[Vector3i, bool]` — set when `chunks_committed` fires; cleared after rebuild
- `_physics_interest: Dictionary[Vector3i, bool]` — cleared each frame; filled by Player + projectiles + rigid bodies calling `register_interest(chunk_pos)` each frame
- Each frame: for up to `SYNCS_PER_FRAME = 4` chunks in `_voxel_dirty ∩ _physics_interest`:
  1. Call `_orchestrator.sync_chunk_slices_from_changelog(chunk_pos)` — applies cached changelog entries for this chunk to `block_data` (O(entries), not 128KB). Returns the set of affected Y indices.
  2. Call `chunk.rebuild_collision_slice(y)` for each affected Y index only.
  3. Clear `_voxel_dirty[chunk_pos]`.
- When a chunk leaves `_physics_interest`, call `chunk.clear_collision_slices()` — clears all shapes, removing it from the broadphase.
- Profiler monitors: `Voxel/collision_dirty`, `Voxel/collision_interest`, `Voxel/collision_rebuilds_frame`

**`Chunk.rebuild_collision_slice(y: int)`** — new method:
Chunk holds 32 shape owners, one per Y-slice, created at init via `shape_owner_create()` and stored in `_slice_owners: Array[int]`. Rebuilding slice y:
1. `shape_owner_clear_shapes(_slice_owners[y])` — removes only this slice's physics shapes
2. Greedy rectangle merge over just that Y-slice of `block_data` (O(cs²) = O(1024) per slice)
3. `shape_owner_add_shape` + `shape_owner_set_transform` for each resulting box

**Why per-slice instead of per-chunk:** A patch of 5 moving sand voxels across 2 Y-slices costs 2 slice rebuilds (~2000 voxel checks + shape updates) instead of a full chunk rebuild (32,768 checks + full shape pool replacement). Total active shapes in the physics world also shrink proportionally — a flat terrain surface becomes 1 box per slice instead of N individual voxel boxes.

**Why not rebuild on every `chunks_committed`:** `chunks_committed` fires every sim tick for every active chunk — 10-60 per frame. Without gating, this collapses to "rebuild the world every frame." The dirty ∩ interest gate reduces rebuilds to: only chunks where (a) voxels actually changed AND (b) something physical is actually nearby. A sand pile 10 chunks from the player generates zero collision rebuilds.

**Interest radius:** CollisionSystem pre-computes `_interesting_chunks` from the player's chunk position each frame (3×3×3 Chebyshev cube = 27 chunks max). Only these chunks are ever registered as interesting. Start with radius 1 and measure; expand if the player clips terrain. Simulation radius and collision interest radius are independent — sand can simulate far outside the collision zone.

#### Step 4 — Player Character (Pillar A)

With collision shapes in place, `CharacterBody3D` physics works out of the box against the heightmap shapes. Add:
- Gravity, jump, step-up (1-voxel height)
- Camera height scaled to new `voxel_scale`
- Optional: supplement Godot physics with a `block_data` sweep for sub-voxel precision if needed (defer until it's actually a problem)

Without a working character you can only test the world from one fixed position. Everything downstream (fluid, editing UX, save/load) is easier to test when you can walk around it.

#### Step 5 — Editing Tools (Pillar B)

`EditDispatcher` already supports BOX, SPHERE, CYLINDER, CAPSULE on the GPU. The gap is player-facing controls:
- **Rectangle/box brush:** hold + drag to define a box region, confirm to place/remove. Useful for building containment walls for liquid testing.
- **Material HUD:** show selected material name/colour and hotbar slots (1-4 keys work, no visual feedback currently)
- **Fill-plane stamp:** place a flat N×N layer in one action (floor, wall)
- **Undo:** queue the inverse `EditRequest` through `EditDispatcher`; single level is enough for testing

These exist so when you test fluid you can build a pond in 3 clicks rather than 10 minutes of single-voxel placement.

#### Future: LOD Simulation

For chunks far from the player, full per-voxel granular simulation is unnecessary. Candidate approaches (to be designed when streaming is working):

- **Coarse sim:** simulate at 1/8th resolution (one representative voxel per 2³ block). Sand pile angle-of-repose heuristic rather than per-grain physics. Correct behaviour at boundaries when LOD chunks border full-res chunks.
- **Frozen state:** chunks beyond a threshold radius simply don't tick. Sand mid-fall is frozen until the player approaches. This is the simplest option and may be "good enough" for the sandbox feel — falling sand you can see is always in the full-res window anyway.
- **Compressed representation:** store dormant chunks as a run-length encoded heightmap instead of full 32³ voxel data. Decompress on demand when the chunk enters the sim window. Significant memory saving at large render distances.

The GPU dirty-bitmask architecture (Phase 6.7) is already LOD-ready: the simulation dispatch only covers the chunks in the chunk list. Excluding far chunks is a matter of not including them in that list.

---

### Collision & Player Movement

#### Current state — VoxelMover ✅

PhysicsServer3D collision shapes and `CollisionSystem` are completely removed. Player movement is driven by `VoxelMover` (`Player/voxel_mover.gd`), a CPU-side AABB mover that queries a GPU surface-extraction cache.

**No `move_and_slide()`, no `CharacterBody3D` physics, no greedy merge, no PhysicsServer3D shapes.** The player's `CollisionShape3D` node is removed. `Chunk` remains `StaticBody3D` in class only (GDScript inheritance) but no shapes are ever registered on it.

**`surface_extract.comp` / `extract_surface_voxels_batch`** — unchanged GPU infrastructure, now consumed by `VoxelMover` instead of `CollisionSystem`. See buffer sizing below.

**`VoxelMover` cache:**
- On `chunks_committed` or player crossing a chunk boundary: marks `_pending_refresh = true`
- In `_process`: extracts up to 8 uncached chunks per frame from the 3×3×3 neighborhood using `extract_surface_voxels_batch`; builds a `PackedByteArray` bitmap (cs³ bytes, 1 = solid surface) per chunk
- `is_solid(world_pos)` → O(1) bitmap lookup: `voxel_to_chunk_position` → `cache[cp][flat]`
- Evicts chunks more than 2 chunk-steps from the player center

**Movement (`VoxelMover.move`):**
- Per-axis: Y first (gravity/jump snap), then X with step-up, then Z with step-up
- `_bottom_solid` / `_top_solid`: 4-corner checks at feet / head (HALF_W = 0.4, HEIGHT = 2.5)
- `_column_solid`: 4 columns × 3 heights (SKIN, mid, top-SKIN) for lateral blocking
- Floor snap: `ceil(ny / vs) * vs` when landing; ceiling snap: `floor((ny + HEIGHT) / vs) * vs - HEIGHT`
- Step-up: when `_column_solid` at proposed pos but NOT at `pos.y + STEP_H (= 0.5 = 1 voxel)` → advance and raise Y

**Why surface voxels are sufficient:** The player can never touch an interior voxel (surrounded by solid on all 6 faces) without passing through a surface voxel first. Cave walls, floor, ceiling — all are surface-extracted. Interior-only voxels are unreachable during normal movement.

**In-flight grain filter:** `sim_active_buf` still correctly excludes actively-falling granular grains from surface extraction. The player walks through falling sand (it's not yet settled = not a solid collision face) and stands on settled sand. Correct behavior, zero extra cost.

**Buffer sizing:**
- `SURFACE_OUT_MAX_PER_CHUNK = 16384` entries per chunk
- `SURFACE_EXTRACT_MAX_BATCH = 8` — up to 8 chunks per GPU dispatch
- Output buffer: `8 × 16385 × 4 ≈ 512 KB`
- `_surface_slot_list_buf` — `int[SURFACE_EXTRACT_MAX_BATCH]`

#### Rigid bodies (future — Phase 11+)

Based on research into Teardown (Dennis Gustafsson) and Noita:

**Architecture: detached voxel clusters**

Rigid bodies in this engine are NOT physics-enabled voxels. They are separate `RigidBody3D` nodes spawned when a voxel cluster detaches from the world (e.g. a tree cut down, a wall blown apart). The workflow:

1. **Detach trigger:** `WorldOrchestrator` detects a structural disconnection (flood-fill connectivity check on the mutated chunk post-commit). Connected components that are no longer attached to static ground spawn as rigid bodies.
2. **Rigid body shape:** Each detached cluster gets a simplified collision shape — OBB (oriented bounding box) or convex hull of the cluster's voxel extent, NOT a voxel-accurate mesh. Shape is computed once at spawn, not updated as the body moves.
3. **Voxel representation:** The body carries its own `PackedByteArray` of voxel data (a copy of the cluster at detach time). It renders via the same raymarcher (submits to a separate voxel buffer slot), but its physics is the simplified shape.
4. **Interaction with simulation:** While airborne, the rigid body excludes its voxels from the world grid (they are zeroed at detach). The simulation world sees empty space where the body was. Grains and fluid flow into that space normally.
5. **Deposit (bake-back):** When the body comes to rest (velocity < threshold for N frames), it calls `WorldOrchestrator.submit_batch()` to write its voxels back into `block_data` at its current world position, then frees itself. The changelog pipeline handles the GPU sync.

**Key references:**
- Dennis Gustafsson (Teardown) blog: OBB broadphase + per-voxel contact normals, TGS solver for pile stability
- Noita (Nolla Games): flood-fill component detection, rigid pixel membership via cell flags, collapse triggers
- NVIDIA GPUGems Ch.29: GPU rigid body via particle proxies (relevant for future GPU-side solver)

**Connectivity detection** (the expensive part) uses a flood-fill over `block_data` triggered by `chunks_committed` when a mutation removes a voxel that was potentially load-bearing. This is the same flood-fill approach Noita uses. It runs on a background thread over the affected chunk and its neighbours; the result is a list of connected components. Components not attached to "anchor" voxels (bedrock, terrain) become detached clusters.

This system is **not planned until rigid body gameplay is needed** (falling trees, destructible structures). The voxel simulation and fluid systems come first.

---

### ~~Phase 5.5 — GPU Edit System~~ ✅ COMPLETE

**Goal:** Replace all player and scripted voxel mutations with GPU-dispatched edit operations. The CPU provides a shape descriptor; the GPU applies it directly to `_voxel_buf_a` and writes a changelog for `block_data` sync. Eliminates the GDScript mutation loop, makes Step 10b natural, and creates a universal edit API that scales from single-voxel clicks to spell volumes to procedural world-gen stamps.

**Why before Phase 7:** Fluid will stress the mutation path heavily — water placement, explosions, spell interactions, environmental changes. Designing the edit system after fluid exists means retrofitting two simulation systems instead of one. Getting this right now means fluid, fire, gas, and rigid-body deposit all use the same edit API from day one.

---

#### ~~Step 14 — EditRequest data model + EditDispatcher~~ ✅ COMPLETE

**`EditRequest`** inner class of `EditDispatcher` (not a standalone resource — avoids an extra file):

```gdscript
class EditRequest:
	var center:       Vector3i
	var shape_type:   int       # EditShape enum
	var shape_params: Vector4   # SPHERE:(r,0,0,0)  BOX:(hx,hy,hz,0)  CYLINDER:(r,h,0,0)  CAPSULE:(r,half_h,0,0)
	var operation:    int       # EditOp enum
	var material_id:  int
	var priority:     int = 0

enum EditShape { SPHERE=0, BOX=1, CYLINDER=2, CAPSULE=3 }
enum EditOp    { ADD=0, REMOVE=1, REPLACE=2 }
```

**`EditDispatcher`** node, instantiated as a child of `WorldOrchestrator` at the end of `_ready()`. `EditDispatcher._ready()` fires immediately after `add_child()` and grabs `_orchestrator = get_parent()`, so all WorldOrchestrator buffers are already valid.

- `apply_immediate(req)` — dispatches the GPU edit shader synchronously. Used for single-voxel player tools (add/remove) that must feel instant.
- `queue_edit(req)` — pushes to `_edit_queue`. Drained by `process_queue()` (called from `WorldOrchestrator._process()` before simulation) up to `EDIT_BUDGET_VOXELS = 32768` (32³) per frame. Sphere spawning and explosions use this path to prevent instant activation of many chunks.
- `_drain_edit_changelog()` downloads the edit changelog and calls `WorldOrchestrator.sync_block_data_from_buf()` -- no chunk re-uploads; the shader already wrote to `_voxel_buf_a`.

**Circular class_name dependency:** `WorldOrchestrator` references `EditDispatcher` in method bodies (fine -- resolved at runtime), but the field declaration uses `var _edit_dispatcher` (untyped) to avoid a parse-time circular dep. `ChunkManager` calls thin wrapper methods on `WorldOrchestrator` (`edit_add_voxel`, `edit_remove_voxel`, `edit_add_sphere`, `edit_remove_sphere`) instead of referencing `EditDispatcher` directly.

**What was replaced:**
- `ChunkManager.spawn_sand_sphere()` -- 200-line GDScript loop building VoxelMutations -- gone, replaced by `edit_add_sphere(center, r, mat)`
- `ChunkManager.explode_at()` -- same loop for REMOVE -- gone, replaced by `edit_remove_sphere(center, r)`
- `ChunkManager.add_voxel / remove_voxel` -- delegated through `edit_add_voxel / edit_remove_voxel`
- `_flush_batch`, `BATCH_SIZE_LIMIT`, `submit_batch` -- removed from chunk_manager
- `VoxelMutation`, `submit_mutation()`, `submit_mutation_direct()`, `_pending_mutations` -- still present for sim-internal moves (`commit()` handles granular sand), to be removed when fluid is wired up

---

#### ~~Step 15 — `edit_apply.comp` shader~~ ✅ COMPLETE

Dispatched over the edit AABB only. For a radius-30 sphere: 61^3 ~= 226K threads, single dispatch.

**Dispatch:** `ceil(aabb.x/8) x ceil(aabb.y/8) x ceil(aabb.z/8)` workgroups at `local_size=(8,8,8)`. Threads outside the AABB bounds-check via `aabb_size` and return early.

**Params buffer:** flat `int data[]` SSBO (96 bytes, 24 ints). Indexed by position -- avoids std430 struct padding surprises with mixed ivec3/vec4 members:

```
data[0..2]   center.xyz
data[3]      shape_type
data[4..7]   shape_params xyzw (stored as float bits, read with intBitsToFloat)
data[8]      operation
data[9]      material_id
data[10..12] aabb_min.xyz
data[13..15] aabb_size.xyz
data[16..18] grid_min.xyz
data[19..21] grid_span.xyz
data[22]     chunk_size
```

**SDF shapes:** SPHERE, BOX (exact SDF), CYLINDER, CAPSULE. SHAPE_MASK deferred (future mesh-to-voxel pipeline). Writes to `voxels_a` directly and appends `(vidx, new_value)` pairs to the edit changelog via `atomicAdd`.

**GLSL gotchas (documented here for future shaders):**
- `flat` is a reserved GLSL interpolation qualifier -- cannot be used as a variable name.
- Non-ASCII characters (em dashes etc.) in GLSL comments corrupt tokenization on some drivers. Use ASCII-only in all shader source.

---

#### ~~Step 16 — Wire player tools through EditDispatcher~~ ✅ COMPLETE

| Old path | New path |
|---|---|
| `ChunkManager.add_voxel(pos, mat)` | `_orchestrator.edit_add_voxel(pos, mat)` |
| `ChunkManager.remove_voxel(pos)` | `_orchestrator.edit_remove_voxel(pos)` |
| `ChunkManager.spawn_sand_sphere(center, r, mat)` | `_orchestrator.edit_add_sphere(center, r, mat)` |
| `ChunkManager.explode_at(center, radius)` | `_orchestrator.edit_remove_sphere(center, radius/voxel_scale)` |
| Future spells / procedural stamps | `EditDispatcher.queue_edit(request)` |

`world.gd` and `player_controller.gd` are unchanged -- they still call `chunk_manager.add_voxel` etc., which now delegate through the orchestrator wrappers. `commit()` is retained for simulation-internal mutations (granular sand moves); it is the next candidate for removal once fluid is validated and VoxelMutation is no longer needed.

---

### ~~Phase 6 — GPU-Primary Architecture~~ ✅ COMPLETE

**Goal:** Eliminate block_data as a real-time intermediary. The profiler showed that `sync_block_data_from_buf` — 607 `apply_mutation` calls — was consuming 132ms per avalanche frame. The GPU already holds authoritative voxel state; the CPU was paying a large tax just to maintain a cache that almost nothing reads at runtime. This phase removes that tax and makes block_data a lazy save-only cache.

**Why before Fluid:** Fluid generates far more changelog entries than granular (every water voxel moves every frame, fill values change continuously). Without this phase, fluid would immediately hit the same 132ms spike at much higher frequency.

**What reads block_data at runtime (audit):**

| Reader | Frequency | Resolution |
|---|---|---|
| `BlockRay._is_solid()` (DDA cursor) | 1-2×/frame | Replaced by GPU cursor readback (Step 17) |
| `active_simulation_mask()` in discovery | Every `chunks_committed` | Derived from changelog new_values (Step 19) |
| Collision rebuild | Removed | — |
| Chunk generation | Once at init, write-only | Unchanged |
| Save / load | On demand | Batch GPU download at save time (Step 18) |

After this phase: rendering, simulation, editing, and cursor targeting all run GPU-to-GPU. block_data is populated once at generation and re-synced only on save.

**GPU-Primary Invariants (enforced from Phase 6 onward):**

1. **GPU is the primary source of truth for dynamic voxel state.**
   `_voxel_buf_a` in `WorldOrchestrator` is the authoritative store for all voxels that may change at runtime (simulation, edits, explosions). Any system that needs to read current voxel state — rendering, physics, cursor targeting — reads from the GPU, not from `block_data`.

2. **`block_data` is a lazy CPU cache, not a runtime source of truth.**
   It is populated once at chunk generation and re-synced from GPU only on save. Code must never read `block_data` expecting current runtime state. Code that reads `block_data` at runtime (e.g. `_is_solid`, `apply_mutation` in hot paths) is a bug under this architecture.

3. **CPU never downloads voxel data during normal gameplay.**
   The only valid `buffer_get_data` calls on `_voxel_buf_a` during gameplay are: (a) edit changelog drain (a few KB of changed-entry pairs), (b) cursor readback (32 bytes), (c) debug overlay (32 bytes). Full-chunk downloads are deferred to save time.

4. **The simulation changelog drives all CPU-side bookkeeping.**
   Chunk dirty flags, simulation masks, and affected-chunk sets are derived from the changelog `(flat_idx, new_value)` pairs — not from scanning block_data. `new_value & 0xFF` gives the new material; that is sufficient to update masks and determine which chunks need collision rebuilds.

5. **The GPU cursor buffer replaces all GDScript DDA raycasting.**
   `RaymarchRenderer` exposes `get_cursor_remove_pos()` and `get_cursor_add_pos()`, populated each frame by the already-running raymarcher. No system should perform a software DDA against block_data for cursor targeting.

---

#### Step 17 — GPU cursor readback (replace BlockRay) ✅ COMPLETE

The raymarcher already DDA-traverses to the hit voxel for every rendered pixel. Piggyback on that to eliminate the GDScript DDA entirely.

**Changes to `raymarch.comp`:**
Add a binding for `_cursor_buf` (32 bytes: `ivec3 remove_pos` + int pad + `ivec3 add_pos` + int pad). When the shader processes the pixel at `mouse_uv`, write `hit_voxel_world_pos` and `hit_voxel_world_pos + face_normal` into the buffer. This happens once per frame at zero extra cost — the traversal is already running.

**Changes to `RaymarchRenderer.gd`:**
Allocate `_cursor_buf` alongside the existing output textures. After each `_render()` + sync, download the 32 bytes and expose via `get_cursor_remove_pos() -> Vector3i` and `get_cursor_add_pos() -> Vector3i`. Return `Vector3i(INT_MAX, ...)` as a sentinel when the ray misses (no hit voxel, e.g. looking at sky).

**Changes to `player_controller.gd`:**
Replace `raycast.get_ray_hit()` with `_renderer.get_cursor_remove_pos()` / `get_cursor_add_pos()`. Remove the `@onready var raycast` reference.

**Deleted:** `Player/ray_cast_3d.gd`, `Player/block_ray.gd` (and the `RayCast3D` node from the player scene).

**Unlocks:** Ghost-sphere brush preview — `get_cursor_add_pos()` gives the placement center every frame at zero extra cost. Future continuous-brush tools read the same value.

---

#### Step 18 — Lazy block_data (remove from hot paths) ✅ COMPLETE

Replace `sync_block_data_from_buf` in the sim hot path with a lightweight chunk-position extractor that never touches block_data:

```gdscript
func _extract_changed_chunks(buf: PackedByteArray, count: int) -> Array[Vector3i]:
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var seen: Dictionary = {}
	var masked_bits: Dictionary = {}
	var slot_to_chunk := _slot_to_chunk
	var chunk_manager := _chunk_manager
	var material_sim := _material_simulation
	for i in range(count):
		var off := 4 + i * 8
		var flat_idx := int(buf.decode_u32(off))
		var new_value := buf.decode_u32(off + 4)
		var slot: int = flat_idx / vpc
		var chunk_pos: Variant = slot_to_chunk.get(slot)
		if not chunk_pos:
			continue
		seen[chunk_pos] = true
		_gpu_dirty[chunk_pos] = true
		if not masked_bits.has(chunk_pos):
			masked_bits[chunk_pos] = 0
		var sim := material_sim[int(new_value & 0xFF)]
		var bit := MaterialDef.SIM_BITS[sim] if sim > 0 else 0
		if bit != 0 and (masked_bits[chunk_pos] & bit) == 0:
			masked_bits[chunk_pos] |= bit
			var chunk: Chunk = chunk_manager.chunks.get(chunk_pos)
			if chunk:
				chunk.mark_simulation_active(sim)
	var out: Array[Vector3i] = []
	for cp in seen: out.append(cp)
	return out
```

`apply_changelog()` calls this instead of `sync_block_data_from_buf`. No `apply_mutation`, no `_write_voxel`, no block_data writes. Cost is O(count) dictionary lookups.

Add `_gpu_dirty: Dictionary` (chunk_pos → true) to `WorldOrchestrator`. Every changelog drain and edit dispatcher drain marks affected chunks dirty.

Add `_sync_chunk_to_cpu(chunk_pos: Vector3i)` — downloads the 128KB slot from `_voxel_buf_a`, writes it to `chunk.block_data`, calls `chunk.rebuild_counts_and_mask()` to restore the simulation mask, then removes from `_gpu_dirty`. Called by the save system for every loaded chunk before writing to disk.

The edit dispatcher (`_drain_edit_changelog`) also goes lazy — with Step 17 done there is no runtime reader, so edits are deferred to save as well.

**Result:** 607 `apply_mutation` calls → 0 on the hot path. The entire `sync_block_data_from_buf` entry disappears from the profiler.

---

#### Step 19 — Simulation mask from changelog ✅ COMPLETE

Without block_data sync, `_simulation_counts` in each chunk goes stale after the first sim pass. The fix: update the mask incrementally from `new_value` in the changelog without needing `old_value`.

```gdscript
# In _extract_changed_chunks, also update masks:
var new_mat := int(new_value & 0xFF)
var new_sim := _material_simulation[new_mat]
if new_sim > 0:
	chunk._simulation_counts[new_sim - 1] += 1
	chunk._simulation_mask |= MaterialDef.SIM_BITS[new_sim]
```

A `masked_bits: Dictionary` (chunk_pos → int bitmask) guards against calling `mark_simulation_active` more than once per simulation class per chunk per tick — reducing the worst case from 21k GDScript method calls to ≤27 per tick when many entries affect the same chunks. Each simulation class bit is tracked independently so that a clear entry (air) doesn't prevent a later sand entry from setting the GRANULAR bit.

Counts are only incremented (adds), never decremented (removes require old_value we no longer have). This means counts drift high over time — but the mask is used only to detect presence (`count > 0`), not absence. False positives (mask says "has granular" when chunk is empty) cause at most 2 extra idle dispatch frames before the 2-frame eviction guard removes the chunk from `_active_chunks`. No correctness issue.

Counts are corrected on `_sync_chunk_to_cpu` (which calls `rebuild_counts_and_mask` from scratch after downloading the full chunk). So counts are always accurate after a save-sync.

**Discovery invariant:** New sand arriving in a previously-idle chunk C is detected because: (a) the changelog entry for C is in the sim pass output, (b) `_extract_changed_chunks` includes C in `affected`, (c) `chunks_committed` fires for C, (d) discovery checks C's mask. Because we increment the mask from new_values, C's mask reflects the incoming sand immediately — no block_data sync needed.

---

### Phase UX-1 — Sphere Edit Tool + Player Scale ✅ COMPLETE

**Goal:** Replace single-voxel click editing with a continuous sphere brush. Left click streams sand into the world; right click removes. Player capsule made significantly larger so voxels feel like a micro-scale world.

#### What was delivered

**Sphere brush (continuous):** `player_controller.gd` checks `Input.is_mouse_button_pressed` in `_process` each frame. While left button is held, emits `add_sphere(rem_pos, sphere_radius, selected_material)`; while right button is held, emits `remove_sphere(rem_pos, sphere_radius)`. Both route through `EditDispatcher.queue_edit` (rate-limited to `EDIT_BUDGET_VOXELS = 32768` voxels/frame), so the sphere streams in at a capped rate even at high radii.

**Signal changes:**
- `player.gd`: removed `add`, `remove`, `spawn_sphere` signals → replaced with `add_sphere(center, radius, material)` and `remove_sphere(center, radius)`
- `chunk_manager.gd`: added `remove_sphere(center, radius_voxels)` thin wrapper around `edit_remove_sphere`
- `world.gd`: updated connections to new signals

**Sphere indicator:** `_indicator` mesh changed from `BoxMesh` (single voxel) to `SphereMesh` with `radius = sphere_radius * VS`. Rendered with an unshaded wireframe semi-transparent shader so the player can see the edit footprint before committing. Updates position each frame to follow the cursor hit point.

**Player capsule:** `CapsuleShape3D` and `CapsuleMesh` radius→1.0, height→4.0 (was 0.5, 2.0). Player is now 8 voxels tall × 4 voxels wide. Camera moved to y=1.7 (eye level for 4m capsule). Makes voxels feel roughly half the visual size at the same `voxel_scale=0.5`.

**Projectile system removed entirely:** `Player/Projectile/` directory deleted (projectile.gd, projectile.tscn, projectile_launcher.gd, projectile_launcher.tscn, projectile_data.gd and their .uid files). `ProjectileLauncher` node removed from `player.tscn`. `ray_cast_3d.gd` also removed (was already unused since Phase 6 GPU cursor readback). B-key sphere shortcut removed from player controller.

---

### ~~Phase 7 — Fluid Solver (water)~~ ✅ COMPLETE (Phase 1 — Noita-style whole-voxel)

**Implemented:** `shaders/fluid_sim.comp` + `simulations/fluid_simulation.gd`. Water (id=3) and oil (id=7) are LIQUID materials and simulate automatically.

**Current movement rules (Noita-style, whole-voxel CAS):**
1. **Fall straight down** — if cell below is air, move there
2. **Fall diagonally down** — if below is blocked, try 4 diagonal-down options (randomised order)
3. **Spread horizontally** — if cannot fall, try 4 horizontal neighbours (randomised order)

Uses the same CAS + changelog infrastructure as `granular_sim.comp`. Identical binding layout (0–8), identical multi-chunk dispatch `(active_count*4, 4, 4)`. Checkerboard alternation on `(local_pos.x + local_pos.z) % 2` to prevent adjacent-cell races. `FluidSimulation` node lives under `WorldOrchestrator/SimulationRuntime` in `world.tscn`.

**Zero pass:** zero list includes all 10 potentially-writable neighbour chunk offsets (self + below + 4 horizontal + 4 diagonal-below) so cross-chunk horizontal flow always finds a zeroed B slot.

**Active chunk lifecycle:** identical to granular — 4 consecutive idle frames before a chunk sleeps. Wakes on `chunks_committed` for any neighbour.

#### Phase 7b — Granular-fluid displacement swap

**Status: ✅ COMPLETE — `try_displace` implemented in `granular_sim.comp`, including a 3-level water relocation (sideways at water's height → sideways at sand's old height → rise above sand as last resort).**

Granular and fluid run **sequentially** in the same tick (granular→apply_changelog→fluid). `try_displace` in `granular_sim.comp` swaps sand into a water cell and relocates the water to sand's old position:

1. Granular: `try_displace(df, sand_voxel, water_voxel, my_flat, vpc)` — CAS claims df for sand (same as `try_move`), then writes `water_voxel` unconditionally to `voxels_b[my_flat]`, logs both as changelog entries.
2. `apply_changelog`: A gets sand at df, water at my_flat.
3. Fluid: reads updated A — sees water at my_flat, simulates it normally.

No conflict: B is re-zeroed by `zero_slots` before each dispatch. `voxels_b[my_flat]` is this thread's exclusive source slot — no other thread can write it. The water's fill byte travels intact with the displaced voxel.

Fall path:
- Destination is **air**: `try_move` (existing, unchanged).
- Destination is **liquid**: `try_displace` (new).
- Destination is **solid/granular**: skip (existing, unchanged).

Same pattern applied to the 4 diagonal-down destinations.

#### Phase 7c — Fill semantics roadmap (Option B → C)

The current solver treats water as whole-voxels (fill = 255 always). Planned upgrade path:

**Step 1 — Sand-water swap (`try_displace`)** ✅ COMPLETE
Sand sinks through water; water rises to fill the vacated space. No fill changes yet.

**Step 2 — Render water fill as color tint** ✅ COMPLETE (renderer only)
`raymarch.comp` multiplies LIQUID voxel color by `tint = 0.4 + 0.6 * (fill/255.0)`. Full water (255) = normal colour; partial fill = darker. Works from all angles with zero sub-voxel geometry complexity. Height-clipping was attempted but abandoned — correct multi-angle clipping requires full sub-voxel ray intersection per entry face, disproportionate to the visual payoff before Step 5 equalization exists.

**Step 3 — Vertical pressure: water stacks fill from the bottom** ✅ COMPLETE
`try_compress` in `fluid_sim.comp`: when the cell below is the same liquid with fill < 255, transfer fill downward. `give = min(my_fill, 255 - their_fill)`. Destination fill tops up; source fill decreases. Source becomes air (0u) when fill reaches 0. Uses same CAS-from-zero pattern as `try_move` — zero_slots already zeroes the below B slot.

**Step 4 — Horizontal spread: whole-voxel (reverted)**
`try_split` was tried and reverted. Splitting fill on horizontal spread created geometric thinning (255→128→64→32) — water spread as a rug of near-invisible slivers. Partial fills will enter the system via the Step 5 equalization pass instead, where they emerge naturally from averaging rather than from spreading.

**Step 5 — Horizontal equalization (inline CAS)** ⚠️ PARTIALLY WORKING, FUNDAMENTAL BIAS UNRESOLVED

Inline step 4 in `fluid_sim.comp`: blocked liquid voxels find their lowest-fill horizontal neighbor and transfer half the fill difference via CAS-from-zero. Partial fills now enter the system via `try_split` (minimum give=64 to prevent thinning cascades). Connected-vessel behavior is visible.

**Root cause of remaining bias:** CAS is the wrong primitive for horizontal fluid movement. Within a GPU warp (32–64 threads linearized X-major), lower-x threads execute their `atomicCompSwap` before higher-x threads. When two cells compete for the same target slot, the lower-x cell always wins. This creates a systematic +X flow preference that no shuffle, checkerboard, or seed change can eliminate — the bias is in the race *outcome*, not the race *ordering*.

The checkerboard prevents adjacent cells competing in the same frame (fixing the bias) but halves horizontal throughput. Removing it restores throughput but reintroduces the bias. There is no setting of these two knobs that gives both correctness and speed.

**Conclusion: CAS-based discrete voxel moves are the right model for granular (sand is discrete) and the wrong model for fluid (water is continuous). The horizontal fluid system needs to be rebuilt on a different primitive.**

#### Phase 7d — Pressure-Driven Fluid (SUPERSEDED by Phase 7e — kept for history)

**Core insight:** Fluid doesn't *move* — it *flows*. The difference matters for the simulation primitive. A grain of sand occupies exactly one cell and relocates to another. Water distributes itself across cells as a pressure field; the physics is about equalizing that pressure, not relocating discrete objects.

##### Voxel layout change for LIQUID voxels

The `fill` byte (byte 1) becomes **pressure** — semantically the same value (0 = empty, 255 = full), renamed to reflect its role as the primary state driving flow. More importantly, bytes 2 and 3 are repurposed:

| Byte | SOLID / GRANULAR | LIQUID |
|------|-----------------|--------|
| 0 | mat_id | mat_id |
| 1 | fill (always 255) | pressure (0–255) |
| 2 | temperature (s8) | vx (s8, velocity in X) |
| 3 | flags | vz (s8, velocity in Z) |

Rationale:
- Per-liquid-voxel temperature is not meaningful — water is a uniform thermal conductor, not a heat source. The thermal pass (Phase 8) will treat all LIQUID cells as conducting at `water_conductivity` without reading byte 2.
- Liquids have no engine flags (no collision, no sim mask needed at voxel level).
- Velocity in vx/vz enables momentum: water picks up speed when flowing, carries through equilibrium, and naturally produces wave propagation and splash behavior.
- Vertical velocity (vy) is omitted — gravity drives vertical movement explicitly and uniformly; storing it per-voxel adds complexity without benefit.

##### Two-pass horizontal update: symmetric flux

**Why symmetric flux eliminates bias:**  
Each LIQUID voxel computes how much pressure to transfer to each horizontal neighbor. It atomicAdds `+delta` to the neighbor's flux slot and `−delta` to its own. Because the operation is symmetric (both contributions happen in one thread), no other thread's execution order can affect the net result — the sum is always correct regardless of warp ordering.

**`_flux_buf`:** one `s32` per voxel slot (same size as `_voxel_buf_a`). Zeroed before each horizontal pass. Holds signed flux deltas; positive = receiving pressure, negative = giving pressure.

**Pass A — Flux Accumulate** (`fluid_flux.comp` pass=0, read-only on voxels_a) ✅ IMPLEMENTED:

For each LIQUID voxel at position P with pressure `p` and velocity `(vx, vz)`:

```
for each horizontal neighbor N at position Q with pressure `q`:
    // Pressure gradient + velocity contribution
    gradient = p - q
    velocity_toward_N = dot(velocity, direction_to_N)
    raw_flux = (gradient + velocity_toward_N) / 4   // /4 caps total outflow to p
    flux = clamp(raw_flux, 0, p / 4)                // never give more than we have
    
    atomicAdd(flux_buf[P], -flux)
    atomicAdd(flux_buf[Q], +flux)
```

The `/4` divisor ensures a cell can give at most `p/4` to each of 4 neighbors (total outflow ≤ p). Conservation is exact: every `−flux` has a matching `+flux`.

**Pass B — Flux Apply** (`fluid_flux.comp` pass=1, reads voxels_a + flux_buf, writes changelog) ✅ IMPLEMENTED (velocity bytes deferred):

```
delta = flux_buf[my_flat]
new_pressure = clamp(pressure + delta, 0, 255)

if pressure > 0 and new_pressure == 0:
    write changelog: my_flat → 0u  (voxel becomes air)
elif pressure == 0 and new_pressure > 0:
    write changelog: my_flat → (source_mat_id | new_pressure << 8)  (air becomes liquid)
else:
    write changelog: my_flat → (voxel & 0xFFFF00FFu) | (new_pressure << 8)

// Velocity update: net flux direction becomes new velocity, decayed
net_flux_x = flux_from_+X - flux_from_-X   // computed from neighbor contributions
new_vx = clamp((vx * 7 / 8) + net_flux_x / 32, -127, 127)   // 7/8 decay + new impulse
new_vz = ... (same for Z)
```

**Pass B source mat_id problem:** when air receives fluid, we need to know which liquid material it is. Solution: Pass A stores the contributing neighbor's mat_id in the flux_buf entry alongside the delta (e.g., high 8 bits of the s32 = donor mat_id, low 24 bits = flux amount). Or use a separate `_flux_mat_buf` (u8 per slot). Simplest: use the high byte.

##### Vertical movement stays CAS-based

Vertical fall (straight down and diagonal) keeps the existing `try_move` / `try_compress` CAS approach. Rationale:
- Competition for the "below" slot is one-to-one (each cell has exactly one cell below), so warp ordering doesn't create systematic directional bias.
- Vertical fall is already fast and correct.
- Mixing pressure-based vertical with CAS-based is unnecessarily complex.

##### Full dispatch order (Phase 7d) ✅ IMPLEMENTED

```
1. zero_slots (granular B + all neighbors)
2. GranularSimulation CAS → apply_changelog
3. zero_slots (fluid B + all neighbors)    [horizontal offsets included]
4. FluidSimulation vertical CAS (fall + compress) → apply_changelog
5. zero flux_buf for all loaded slots (CPU buffer_update per slot)
6. FluidSimulation Pass 0: flux accumulate (atomicAdd to flux_buf, no changelog)
   [submit + sync — GPU barrier before Pass 1]
7. FluidSimulation Pass 1: flux apply → changelog → apply_changelog
```

Steps 6 and 7 require two separate dispatches with a submit/sync between them. This is 4 GPU sync points per tick per sub-step (fluid runs steps_per_tick sub-steps). Acceptable cost.

##### Why Phase 7d doesn't work — root cause

`_flux_buf` is zeroed and recomputed from scratch every tick (`_zero_flux_buf` before every Pass 0). This means the system has **no memory of motion between ticks** — it is pure diffusion, mathematically identical to fill-averaging, just spread over two passes instead of one. A diffuser can only relax toward flat; it can never overshoot equilibrium, so there is no slosh, no ripple, no wave, no momentum. Water *oozes* rather than *flows*.

This root cause produced every symptom patched during Phase 7d's tuning sessions:
- **Trace cells (fill 1–31) hanging in midair** — required manually capping donor outflow, snapping sub-threshold results to 0 in two different places (`fluid_sim.comp` `try_compress` and `fluid_flux.comp` Pass 1), and blocking sub-threshold cells from falling at all. None of this is physics — it's ad-hoc conservation enforcement standing in for a proper scaling pass.
- **No hydrostatic pressure** — two full (fill=255) cells push horizontally with identical force regardless of how many full cells are stacked above them. Deep water can't out-push shallow water. U-tubes and fountains are impossible.
- **Vertical fall is a separate, disconnected mechanism** (CAS-based `try_move`/`try_compress` in `fluid_sim.comp`) with no relationship to horizontal flux. Water that falls carries no momentum into its next horizontal move — a waterfall lands and instantly stops, rather than surging outward.
- **Sand-water displacement was never implemented** — sand's `try_move` in `granular_sim.comp` only CASes against air; it silently erases (teleports) water on contact rather than displacing it. The Phase 7d design below assumed this was solved; it isn't.

**Conclusion: recompute-from-scratch flux is the wrong primitive, for the same class of reason CAS was wrong for horizontal movement in Phase 7c. Both had no persistent state to carry motion across ticks. The fix is the same shape as the granular→pressure lesson: stop discarding information the system already computed.**

#### Phase 7e — Persistent Flux (Virtual Pipe Method) — PLANNED, supersedes 7d

**Core insight:** the missing ingredient in Phase 7d wasn't a smarter formula, it was **memory**. A flux value that persists tick-to-tick and only *decays* (rather than being erased and recomputed) turns the same physics into a damped oscillator: an imbalance overshoots equilibrium, rings, and settles — a wave, not an ooze. This is the "virtual pipe" method (Mei et al. 2007; Jákó & Tóth 2011) generalized from a 2.5D heightmap-column model to full 3D per-voxel-face flux.

##### What gets scrapped

| Scrapped | Replaced by |
|----------|-------------|
| `fluid_flux.comp` two-pass accumulate/apply (zeroed every tick) | 3-pass persistent flux: Accelerate → Scale → Transport |
| `_flux_buf` owned by `WorldOrchestrator`, dense (sized to full voxel buffer) | `_flux_buf` owned privately by `FluidSimulation`, sparse per-chunk pages |
| `WorldOrchestrator.get_flux_buf()` and all `_flux_buf` allocation/grow/free code in `WorldOrchestrator.gd` | `FluidSimulation`-internal allocation, page freelist, `_flux_chunk_table` |
| Vertical CAS fall (`try_move`/`try_compress` in `fluid_sim.comp`) for liquids | Y-axis flux face, driven by hydrostatic head — folded into the same 3-pass mechanism as X/Z |
| `fluid_sim.comp` entirely (the vertical-only CAS shader) | ✅ deleted (Step 7) — it was found to still be actively destroying fill via a pre-Phase-7e hack, see the resolved bug entry below |
| MIN_FILL trace-cell snapping hacks (Pass 1 snap-to-zero, `try_compress` remainder destruction, sub-32-fill fall guard) | Scale pass conservation guarantee — no trace cells are possible by construction, no thresholds needed |
| `steps_per_tick` sub-stepping workaround for stability | `damp` / `g·dt` tuning parameters, exposed via a debug UI slider (see Step 9) |

Granular sand (`granular_sim.comp`) is untouched — this is a fluid-only rebuild. Sand-water displacement (Step 8 below) is the one granular-adjacent change, and it is additive (a new destination check), not a rewrite.

##### New voxel-adjacent state

No changes to the 4-byte voxel word. `fill` (byte 1) remains 0–255 and is still the sole persistent liquid state in `_voxel_buf_a`. Flux is *not* voxel state — it is `FluidSimulation`-private, per the Pillar 3 ownership rule (simulation-private data never touches the Orchestrator's buffer table).

**`_flux_buf` (new, private):** one entry per liquid voxel slot, three signed 16-bit lanes packed into a u64 conceptually (GLSL: two u32 — `f_x | f_y << 16` in one, `f_z` in the other, or three separate i32 arrays if binding count allows). Each liquid voxel owns its `+X`, `+Y`, `+Z` face flux; the `−axis` value at any position is just the neighbor's owned value, negated (§ face ownership convention — kills double-accounting, makes seam behavior order-independent).

**Sparse paging:** flux storage is allocated per-chunk, not globally. `_flux_chunk_table: Dictionary` (chunk_pos → page slot) mirrors `WorldOrchestrator._chunk_table`'s pattern but is private to `FluidSimulation`. A chunk with no page is treated as a zero-flux boundary — not an error, not a special case in the shader (one table lookup before the physics).

##### Step-by-step implementation plan

**Step 1 — Ownership migration** ✅ COMPLETE (2026-07-01)
Removed `_flux_buf`, `get_flux_buf()`, and all allocate/grow/free calls from `WorldOrchestrator.gd`. Added private buffer management to `simulations/fluid_simulation.gd`: `_flux_buf` + `_flux_buf_capacity`, allocated/grown lazily via `_ensure_flux_buf()` at the top of `step()`, freed in `_exit_tree`. No shader changes — same buffer, same bindings, purely relocated ownership.

**Implementation note:** `WorldOrchestrator._ready()` runs *after* its children's `_ready()` (Godot calls `_ready()` bottom-up), so `SimulationRuntime`/`FluidSimulation` see `get_voxel_buf_capacity() == 0` at their own `_ready()` time — `_init_grid()` hasn't run yet. Allocation is therefore lazy, checked once per `step()` rather than done in `_ready()`, mirroring the existing `is_grid_valid()` gate pattern already used to defer work until the grid exists.

**Also fixed while touching this path:** `compute_helper.gd`'s `buffer_grow()` used a per-byte GDScript copy loop (review-flagged bug #7 — a multi-second hitch on any buffer >~1MB). Since the new `_ensure_flux_buf()` growth path calls it directly, fixed it in the same change: `PackedByteArray.resize()` already zero-fills the new tail, so the manual loop was pure overhead.

**Step 2 — Sparse page table** ✅ COMPLETE (2026-07-01)
Added `_flux_chunk_table: Dictionary` (chunk_pos → page index) + `_flux_freelist` (array-backed pool, same pattern as `WorldOrchestrator._free_slots`) to `fluid_simulation.gd`. `add_candidates()` allocates a page for any candidate chunk missing from `_flux_chunk_table` via `_allocate_flux_page()`. Eviction rides the existing idle-guard counter in `step()`'s active-chunk bookkeeping — a chunk falling out of `_active_chunks` calls `_free_flux_page()`, which also zeroes the vacated page (Step 6b requirement — see below). `_flux_buf` is now sized in pages (`_flux_page_capacity`), not voxel-slot capacity; grows via the existing (now-fixed) `buffer_grow`.

**GPU-side mirror implemented per review note:** `_flux_page_table_buf`, a dense `int` array sized to `grid_span` (fixed after `_init_grid()`, mirrors `WorldOrchestrator._chunk_table`'s own invariant — never needs to grow), value = page index or −1. Written with 4-byte `buffer_update`s on allocate/free. `fluid_flux.comp` gained binding 8 (`FluxPageTable`) and a `flux_flat(gpos)` helper that resolves a world position to its page-based flux index, separate from `world_to_flat(gpos)` (voxel index, unchanged). `main()` now computes both `my_flat` (voxel indexing — `voxels_a`/changelog/dirty-bitmask, untouched) and `my_flux_flat` (page indexing — the only thing that changed) side by side. A chunk with no page returns −1 from `flux_flat()` and is skipped exactly like an out-of-grid neighbor — no special-case branching needed.

**Known 1-tick lag (acceptable, same category as existing residual propagation):** a chunk that has never held liquid has no page. If cross-chunk flux would flow into it for the first time, Pass 0's `flux_flat()` lookup returns −1 and that inbound delta is silently dropped — the same tick it happens, that chunk isn't a fluid candidate yet either, so it can't have a page. It becomes a candidate (and gets a page) once flux actually materializes a voxel there via a *direct* neighbor dispatch or edit, at which point normal propagation continues next tick. Mirrors how granular's dirty-bitmask discovery already chains one commit at a time — not a new class of bug.

**Step 3 — Hydrostatic head function** ✅ COMPLETE (2026-07-01)
Added `int head(ivec3 p, uint mat_id)` to `fluid_flux.comp`, directly below `mark_dirty()`. Walks upward through consecutive full (fill==255) same-material liquid cells starting at `p`, summing fill; stops at (and includes) the first non-full liquid cell in that column — the water surface — or stops at (and excludes) the first non-liquid/different-material cell, capped at `HEAD_MAX_DEPTH = 24` reads ("very deep" is visually indistinguishable beyond that). A partial-fill `p` returns its own fill immediately — it *is* the surface, nothing above it can be submerged in a well-formed column. Implemented as `int` (matches the existing 0–255 fill domain everywhere else in the codebase) rather than `float` per the source design — no precision need until Pass A's damping math (Step 4) requires fractional values, at which point individual terms get cast, not the storage.

Pure read of `voxels_a` via the existing `read_a()` helper — no new persistent state, self-corrects every tick. Not wired into Pass A/B yet; this step only adds the building block Step 4 will call for all three flux axes (the review's correction that head-driven flux must apply to X/Z too, not just Y, is why every axis will route through this same function).

**Step 4 — Pass A: accelerate flux** ✅ COMPLETE (2026-07-01)
Implemented in `fluid_flux.comp` (`params.pass == 0`; passes 1–2 are stubs that return immediately, reserved for Steps 5–6). Reads `_voxel_buf_a` via `head()` and the persistent `_flux_buf`, writes `_flux_buf`. For each of the 3 owned faces (+X, +Y, +Z):
```
dh = head(p, mat_id) - head(neighbour, mat_id)   // ALL axes route through head(), per the review correction
f' = clamp(f * damp + dh * g_dt, -FILL_CAP, FILL_CAP)
```
`FILL_CAP = 255` (a single tick can't move more than one full voxel's worth of fill — matches the design's clamp intent). Applied the review correction directly: all three axes use `head()`, not raw fill for X/Z — confirmed this was the right call, since a flat-fill horizontal pair at depth would otherwise silently reintroduce the exact bug this whole rebuild exists to fix.

**Buffer layout — 3 lanes per voxel, not 1.** `_flux_buf` changed shape from Step 2's single scalar per voxel to 3 signed int lanes (fx, fy, fz), stride 3, interleaved: `page*vpc*3 + local*3 + axis`. Page allocation, growth, and eviction sizing in `fluid_simulation.gd` all updated to `vpc * 3 * 4` bytes per page. No atomics needed — confirmed: each thread writes only its own 3 owned lanes (`my_page`/`my_local` derived from its own dispatched voxel), never a neighbour's, so there's no race regardless of dispatch/warp order.

**Persistence — the core behavior change.** `flux_buf` is no longer zeroed between ticks (that zeroing was Phase 7d's root bug). `_zero_flux_buf()` and its call site are deleted from `fluid_simulation.gd` entirely; pages are still zeroed once on allocation/eviction (page reuse safety, Step 6b), which is a rare event compared to the old per-substep clear.

**Extra invalidation safeguard (beyond what was planned):** when a face's neighbour is solid or a different liquid, Pass A zeroes that lane immediately rather than letting stale flux decay slowly via `damp`. Cheap and prevents a corner case where a voxel briefly becomes non-liquid (edit, sand displacement) and its neighbour's stored outward flux toward it lingers for several ticks after the geometry changed.

**Current interim state — expected, not a bug:** Pass A alone does not move fill. It updates the persistent flux field correctly (visible only via GPU buffer inspection at this stage), but nothing reads `_flux_buf` to produce a changelog entry until Scale (Step 5) and Transport (Step 6) exist. Horizontal/vertical flow via this new system is inert until Step 6 lands — the legacy vertical CAS in `fluid_sim.comp` is still what drives all visible fluid motion in the interim, exactly as planned in the original Step 7 note ("delete `fluid_sim.comp`" only happens after Pass A/Scale/B are validated).

**Tuning exposed early (originally Step 9, pulled forward):** `FluidSimulation.damp` (`@export_range(0.0, 1.0, 0.001)`, default 0.97) and `FluidSimulation.g_dt` (`@export_range(0.0, 50.0, 0.1)`, default 6.0) are live-editable via Godot's remote Inspector on the running instance (Scene dock → Remote → select the FluidSimulation node while playing). Uploaded into `SimParams` every tick (`_flux_params_buf` grew from 40 to 44 bytes to carry the two floats), so changes take effect on the next simulation tick with no restart. First real visual test of these values has to wait for Step 6 (Transport) — until then they're only observable by their effect on the raw flux numbers, not on rendered water.

**Step 5 — Scale pass: conservation guard** ✅ COMPLETE (2026-07-01)
Implemented in `fluid_flux.comp` as `params.pass == 1`, dispatched as a second call after Pass A with its own `submit()`/`sync()` — confirmed the review's correction that this can't be fused into Pass A's tail (cross-workgroup read of neighbour-owned faces). For each voxel: `s = min(1, fill(p) / total_outflow(p))`, then every face representing outflow from `p` is scaled by `s`. This is what replaces every trace-cell hack from Phase 7d — a cell mathematically cannot be asked to give more than it has, so no floor threshold, no snap-to-zero, no minimum-fill guard is needed again.

**A voxel's total outflow is not just its 3 owned faces.** Faces are shared: `p`'s −X face is `(p−X)`'s owned +X face, `fx(p−X)`. Per the (not-yet-implemented) Transport formula `fill'(p) = fill(p) − fx(p) − fy(p) − fz(p) + fx(p−X) + fy(p−Y) + fz(p−Z)`, a **negative** `fx(p−X)` subtracts from `p` — i.e. it's outflow from `p`, even though `p−X` owns that storage. So `total_outflow(p) = sum(max(0, owned_face)) + sum(max(0, −neighbour_owned_face))`.

**No-double-scale proof (verified, not just asserted):** a face is scaled by whichever cell's outflow it represents, decided purely by sign — positive → the face's owner scales it (found in its own "owned faces" loop); negative → the −axis neighbour scales it (found in its "neighbour faces" loop). Exactly one of the two ever writes to a given face this pass. Confirmed this holds regardless of GPU thread/dispatch ordering: scaling multiplies by `s ∈ [0,1]`, which is sign-preserving, so a reader that races a writer within the same Scale dispatch sees either the original or the already-scaled value — both have the same sign — so its own read-the-sign decision is correct either way. No atomics needed.

**Dispatch range correction (found while implementing, not previously flagged):** `_build_flux_chunk_list` only added the 4 *horizontal* neighbours to the flux dispatch range — a leftover from when flux was horizontal-only (pre-Step-4). Since Pass A folded the Y axis into flux, Scale's neighbour-owned-face lookups can now cross a *vertical* chunk boundary too. Added `+Y`/`−Y` to the neighbour offsets so the full 6-neighbour set is covered, matching what Y-axis flux actually needs.

**Step 6 — Transport: changelog proposal** ✅ COMPLETE (2026-07-01)
Implemented as `params.pass == 2` in `fluid_flux.comp`, dispatched third (after Pass A and Scale, each with its own submit/sync — Transport also reads neighbour-owned faces, the same cross-workgroup dependency as Scale). Reads `_voxel_buf_a` (fill) + the now-conservation-safe `_flux_buf`, writes the standard changelog — never writes `_voxel_buf_a` directly, same single-writer discipline as every other simulation:
```
fill'(p) = fill(p) - fx(p) - fy(p) - fz(p) + fx(p-X) + fy(p-Y) + fz(p-Z)
```
mat_id/temperature/flags pass through unchanged. **Air-materialization**, flagged as a gap in the source doc's pseudocode, is implemented: when `p` is currently air and receives net positive inbound flux, Transport scans all 6 neighbours for a liquid donor's mat_id (identical logic to the deleted Phase 7d Pass 1 `is_air` branch) before writing a valid changelog entry.

**No trace-cell handling needed — confirmed, not just hoped.** Because Scale (Step 5) already guarantees a voxel can never be asked to give more than `fill(p)` on the outflow side, Transport doesn't need any of Phase 7d's MIN_FILL/snap-to-zero/outflow-cap hacks. The formula is applied directly; conservation is a structural guarantee from the pass before it, not a per-pass patch.

**GDScript wiring (`fluid_simulation.gd`):** Transport is the first of the three flux passes that touches the changelog, so it gets its own `reset_changelog()`/dirty-bitmask-zero immediately before dispatch and `apply_changelog()`/bitmask-download immediately after — mirroring the exact pattern the vertical CAS dispatch already uses earlier in the same sub-step. The per-substep dirty-bitmask merge (`merged_bitmask`) now ORs in the flux Transport bitmask again, restoring what Step 4 temporarily removed when Pass A had nothing to report.

**This is the pass that makes water move under the new system.** Steps 4–5 only built up to a correct, conservation-safe flux field with no observable effect; this is what finally reads it and turns it into fill changes. `fluid_sim.comp` (legacy vertical CAS) is still dispatched alongside this shader — full removal is Step 7, pending validation that Y-axis flux alone produces equivalent-or-better fall behavior.

**Step 6b — Flux invalidation (added in review, 2026-07-01)**
Persistent flux means stale momentum can outlive the water that carried it. Whenever a liquid voxel is destroyed (fill → 0), displaced by sand (`try_displace`), or overwritten by an edit, its 3 owned face fluxes must be zeroed — otherwise the next occupant of that cell (air, sand, or a different poured liquid) inherits the old flow, and Pass B materializes water out of nothing or instantly drains a fresh cell. Cheapest enforcement: Pass A zeroes (instead of integrating) any face whose owning cell is not currently the same liquid material; edits additionally zero the flux page region for chunks they touch. Without this, Step 8's sand-water displacement corrupts the very flux field it writes impulses into.

**Step 7 — Fold vertical fall into the same mechanism, delete `fluid_sim.comp`** ✅ COMPLETE (2026-07-01)
Arrived at earlier than the plan expected — not from a scheduled validation pass, but because keeping `fluid_sim.comp` running alongside the new system turned out to be actively destroying fill (see the resolved bug entry above: its `try_compress` still carried a pre-Phase-7e destructive snap-to-zero hack). Deleted `shaders/fluid_sim.comp` and stripped all associated scaffolding from `fluid_simulation.gd` (`_pipeline`, `_zero_pipeline`, `_chunk_list_buf`, `_zero_chunk_list_buf`, `_global_params_buf`, and their build/upload functions) — no hybrid fallback, per the original "full adoption" decision. Y is now purely another flux axis, driven by `head()` exactly like X/Z, with no separate discrete-move mechanism. `damp`/`g_dt` govern vertical free-fall the same way they govern horizontal spread; no separate tuning knob was needed for "terminal velocity" — worth watching whether free-fall through open air still feels snappy enough once more test scenes exist, but nothing in the data so far suggests it needs a dedicated fix.

**Step 8 — Sand-water displacement via velocity impulse**
`granular_sim.comp`'s `try_move` gains a destination check: if the target voxel is a LIQUID material (not just air==0), perform `try_displace` — sand CASes into the slot, the displaced water's changelog entry moves it to the sand's old position (mirrors the existing `try_move` two-entry changelog pattern). Additionally, write an outward impulse into the 4 horizontal neighbors' `_flux_buf` entries at the impact point, scaled by fall speed (approximable from consecutive `sim_active` ticks). Next tick's Pass A naturally propagates that impulse as a pressure wave — this is also the mechanism that makes a Rayleigh-jet-like splash plausible (convergent inbound flux raises local `fill`, which raises local `head`, which drives strong +Y flux upward the following tick). Expect a mound/bump, not a thin jet, at current voxel scale — see conversation notes; not a limitation worth chasing further right now.

**Step 9 — Tuning UI**
Debug-only slider(s) (reuse the pause-button HUD pattern from earlier) exposing `damp` and `g*dt` per liquid material at runtime, so values can be dialed in interactively instead of edit-recompile-test. First test case per the source design: drop one full cell into a flat pool and watch the ring — cleanest way to isolate `damp` before terrain complexity is involved.

**Step 10 — U-tube / overpressure validation**
Explicit test scene: two vertical columns connected by a horizontal passage at the bottom, unequal water heights. Confirms Pass B's air-materialization (Step 6) correctly lets the low column's top surface rise into previously-air cells as flux pushes it upward — this is the same mechanism needed for the splash jet in Step 8, so validating it here de-risks both.

##### Sand-water interaction (superseded by Step 8 above — reference only)

With pressure-based horizontal flow, the water displacement problem simplifies. When sand falls into water:
1. `try_displace` CAS: sand claims the water slot. Water's slot becomes air (0u). Changelog: slot → sand.
2. Sand writes a velocity impulse to `vx`/`vz` of the 4 horizontal neighbors of the impact point. The impulse magnitude scales with impact velocity (fall height can be approximated from `sim_active` data).
3. Next tick's flux pass naturally propagates the velocity outward as a pressure wave.

No complex level-1/level-2 displacement logic needed. The velocity system handles propagation.

##### Rendering with pressure

The `tint = 0.4 + 0.6 * (pressure / 255.0)` coloring in `raymarch.comp` stays unchanged — it already reads byte 1. No renderer changes needed.

**Option D — Pressure height field** — per-column height + shallow water equations. Better for large open bodies of water but doesn't handle enclosed spaces (caves). Not planned.

##### Multi-fluid interaction (water + oil + lava)

MaterialDef gains a `density: u8` field in the GPU buffer. This drives all cross-material fluid behavior.

**Horizontal flux — same-material only.** A water cell adjacent to an oil cell computes zero horizontal flux toward it. Different fluids never directly exchange pressure. This naturally keeps them separated at a shared boundary without any explicit "don't mix" logic.

**Vertical displacement — density-ordered.** In the vertical CAS pass, after checking for air and same-material compression:

```
if below_mat != my_mat and below_density < my_density:
    try_displace (denser fluid sinks, lighter fluid rises — same CAS as sand-water)
if below_mat != my_mat and below_density >= my_density:
    skip (stable — lighter fluid above denser, or equal density)
```

Water above oil: `water_density > oil_density` → water sinks, oil rises. ✓  
Oil above water: `oil_density < water_density` → stable, nothing happens. ✓  
Water above lava: `water_density < lava_density` → stable (water floats on lava). ✓  

**Diagonal displacement inherits the same density check.** Water falling diagonally into an oil cell triggers displacement, which is how sorting propagates through mixed columns.

**Known limitation — flat vertical interfaces:**

```
W O
W O    ← neither fluid is above the other → vertical displacement never triggers
W O
```

A perfectly flat wall of water adjacent to a perfectly flat wall of oil at identical heights produces no interaction. Neither horizontal flux (different materials) nor vertical displacement (no height offset) fires. They sit side-by-side indefinitely. This is physically wrong (the interface should be Rayleigh-Taylor unstable) but is an edge case that never arises naturally in gameplay — the player always pours one fluid into another from above, which does trigger correct sorting.

**Fix — stochastic perturbation (planned, ~5 lines):**

In the vertical CAS pass, each liquid voxel adjacent horizontally to a lower-density fluid has a small random chance (~5%, driven by the existing `randf(seed)`) of attempting a diagonal displacement *downward* into that neighbor's layer, even when not vertically blocked. This seeds the instability stochastically — the boundary begins to finger and the column sorts naturally within a few ticks. Looks organic; Noita uses a similar approach.

Not implemented yet. Note here so we don't re-derive the problem.

#### 6.4 Granular-fluid interaction

Dispatch order matters: **granular runs before fluid** each frame. Sand reads from `voxels_a` (pre-fluid state). If sand falls into a water cell, `try_displace` swaps sand and water — sand wins the position, water is relocated to sand's old cell. After the granular pass, `apply_changelog()` updates `voxels_a`. Fluid then reads the post-granular `voxels_a` and simulates the relocated water normally. No special handling required — the sequential dispatch order makes this correct by construction.

#### 6.5 `fluid_simulation.gd`

Drop-in `VoxelSimulation` subclass — `SIMULATION_CLASS = LIQUID`, same `add_candidates(chunk_positions)` / `step(ctx)` / `has_work()` interface as `GranularSimulation`. Registers itself with `SimulationRuntime` at `_ready()`. Zero runtime or orchestrator changes required.

Discovery: `_on_chunks_committed` already checks all simulation bits against `active_simulation_mask()`. Adding fluid just means there's now a non-zero entry in `_simulations` for the `LIQUID` bit. No runtime changes.

#### 6.6 Rendering

`fill` is already uploaded to the GPU in byte 1 of each voxel word. The raymarcher can use `fill / 255.0` as a partial-occupancy scalar for rendering partially-filled water cells (e.g., render water surface at the fill-height within the voxel). This is a renderer concern deferred to after the simulation is stable.

Fire, gas, and thermal simulations follow the same pattern in later phases.

---

### Phase 8 — Fire & Thermal

**Goal:** Fire spreads to flammable materials, consumes fuel over time, produces smoke (gas), and raises local temperature. Temperature diffuses between adjacent voxels. Materials combust when temperature exceeds their ignition threshold.

**Why this order:** Fire requires gas (smoke output) to be specced first so the fire→gas interaction is designed correctly, but fire's *input* side (heat, ignition) is independent of gas. Fire lands here to give gas a concrete consumer before Phase 9 implements it.

#### 7.1 Temperature field

The `temperature` byte is already stored in every voxel (`Chunk` byte 2, signed -128..127). Thermal simulation reads and writes this field. Range: -128 = absolute cold, 0 = ambient, 127 = max heat.

Thermal diffusion runs as a compute pass over all loaded chunks (not just active simulation chunks) — even "stable" chunks conduct heat slowly. Diffusion rate is a per-material constant stored in the `MaterialDef` GPU buffer (new field: `thermal_conductivity: float`).

#### 7.2 Fire rules (per-voxel, each sim tick)

1. **Burn:** if `mat_id == FIRE`, decrement a fuel counter encoded in `flags` byte. When fuel exhausted, voxel becomes SMOKE (gas) or AIR depending on fuel type.
2. **Spread:** for each of the 6 face-adjacent neighbours, if neighbour temperature > ignition threshold for that material AND neighbour is flammable (new `MaterialDef` field: `ignition_temp: int8`), set neighbour to FIRE with full fuel.
3. **Heat:** FIRE voxels raise their own temperature to max each tick and radiate heat to neighbours via the thermal diffusion pass.

#### 7.3 Thermal diffusion shader: `shaders/thermal_sim.comp`

Separate from fire spread. Dispatched over ALL chunk slots (not just active simulation chunks). Each voxel reads its 6 neighbours' temperatures and moves toward the weighted average, scaled by `thermal_conductivity`. AIR has low conductivity; rock has high; water has medium.

This pass runs every sim tick regardless of activity — there is no "stable" thermal state until temperature equalises across the world. Optimisation: only dispatch chunks where `max_temperature_delta > threshold` this tick (tracked via a reduction pass or approximated by a dirty flag on the chunk).

#### 8.4 MaterialDef additions for Phase 8

| Field | Type | Description |
|-------|------|-------------|
| `flammable` | bool | Can this material catch fire? |
| `ignition_temp` | int8 | Minimum temperature to ignite |
| `fuel_ticks` | uint8 | How many sim ticks this material burns for |
| `thermal_conductivity` | float (0–1) | How fast temperature equalises with neighbours |
| `produces_on_burn` | uint8 (material_id) | What material replaces this when fuel exhausted |

These are GPU-buffer-only additions (packed into reserved bytes in the 32-byte material layout). No CPU-side simulation logic reads them.

#### 7.5 Interaction priority (dispatch order)

1. Thermal diffusion (all chunks)
2. Granular (sand falls through fire correctly — reads post-thermal voxels_a)
3. Fluid (water absorbs heat, steam production deferred)
4. Fire spread (reads post-granular, post-fluid state — correctly sees if water extinguished it)

---

### Phase 9 — Gas Simulation

**Goal:** Smoke and gas rise, spread horizontally, and dissipate over time. Gas is produced by fire (smoke) and is the first simulation with upward gravity.

Gas shares the pressure-field model introduced in Phase 7d but with `GravityMode.UP`. Gas rises, spreads horizontally via flux (same symmetric delta buffer), and dissipates (pressure decrements each tick by a material-specific rate until the voxel disappears). Velocity bytes (vx/vz) are reused the same way — momentum carries gas in a direction even when pressure is balanced.

#### 8.1 Differences from fluid

| Property | Fluid (water) | Gas (smoke) |
|---|---|---|
| Gravity | DOWN | UP |
| Fill behaviour | Conserved (water doesn't disappear) | Dissipating (fill decrements each tick) |
| Pressure | Equalises to flat surface | Spreads and thins toward ceiling |
| Interaction with fire | Extinguishes (future: steam) | Produced by fire |

#### 8.2 `gas_simulation.gd`

Same `VoxelSimulation` interface, `SIMULATION_CLASS = GAS`. Can share `fluid_sim.comp` via a shader variant (a `#define GRAVITY_UP 1` preprocessor flag inverts vertical movement direction) or have a dedicated `gas_sim.comp`. Dedicated shader preferred for clarity.

#### 8.3 Discovery

The `active_simulation_mask()` check in `_on_chunks_committed` already handles GAS chunks — no runtime changes. Gas chunks enter `_active_chunks` when `chunks_committed` fires for a chunk with a GAS-class voxel. Gas dissipates to AIR, which removes the GAS mask bit, which causes the chunk to exit `_active_chunks` naturally after the 2-frame idle guard.

---

### Phase 9.5 — GPU Particle Layer (Explosions, Splashes, Debris)

**Goal:** A thin GPU particle buffer that sits alongside the voxel grid. Voxels can be *ejected* into the particle layer, simulate freely (velocity, gravity, collision), and *rebake* back into the voxel grid on impact. This gives Noita-style physics — sand flung by an explosion actually arcs through the air and lands, rather than instantly appearing in a new voxel position.

#### Why a separate layer

The voxel grid is a spatial index — it answers "what is at position X?" in O(1). It is not designed to track objects that are between cells. A grain of sand mid-flight has a sub-voxel position and a continuous velocity. Storing it in the voxel grid during flight requires either quantizing its position every tick (jumpy movement) or choosing a temporary "occupied" voxel (which blocks other things from entering that space incorrectly).

The particle layer stores a flat list of active particles. Each particle has continuous position and velocity, computed on the GPU each tick. The voxel grid is read-only to the particle pass (collision detection only). This is a clean layer separation: the particle sim doesn't mutate voxels except when rebaking.

#### Voxel layout — particles reuse the existing word

A particle in flight carries its full 4-byte voxel word (mat_id + pressure/fill + vx + vz). When it rebakes, this word is written directly into voxels_a via the changelog. No format conversion needed.

#### `_particle_buf` layout

Fixed-capacity buffer (e.g., 65536 particles). One entry per particle:

| Field | Bytes | Description |
|-------|-------|-------------|
| voxel | 4 | Full voxel word (mat_id, fill, vx, vz) |
| pos_x | 4 | float — world X |
| pos_y | 4 | float — world Y |
| pos_z | 4 | float — world Z |
| vel_x | 4 | float — velocity X (world units/tick) |
| vel_y | 4 | float — velocity Y |
| vel_z | 4 | float — velocity Z |
| alive | 4 | u32: 0 = dead slot, 1 = active |

Total: 32 bytes/particle. 65536 particles = 2 MB.

Slot allocation: use `atomicAdd` on a global counter to claim slots (same CAS pattern as the changelog). Dead slots are reclaimed by a compaction pass (low priority — deferred).

#### Ejection

When an explosion fires or a high-velocity collision occurs, the ejecting system (edit_dispatcher or a new explosion shader):
1. For each voxel in the blast radius, read the voxel word, write `0u` to its voxel_buf_a slot (via changelog), and claim a particle slot.
2. Write the particle with position = voxel center, velocity = direction from explosion center × blast strength + noise.
3. Particle goes active immediately next tick.

Sand ejection from granular is the first use case. A high-velocity sand grain (detected when `sim_active` shows it fell more than N cells in one tick) can optionally eject into the particle layer rather than landing immediately, giving it ballistic flight.

#### Particle tick (`particle_sim.comp`)

Each active particle per tick:
1. Apply gravity: `vel_y -= gravity_constant`
2. Apply drag: `vel *= 0.99`
3. Advance: `pos += vel`
4. Collision check: read voxel at `ivec3(floor(pos))` from voxels_a. If non-air → rebake.
5. Rebake: find the nearest air voxel adjacent to the collision point, write the particle's voxel word there via changelog. Mark slot dead.
6. Velocity impulse on rebake: write vx/vz to neighboring fluid cells (water splash).

#### Dispatch order with particles

```
1. GranularSimulation CAS → apply_changelog
2. FluidSimulation vertical + horizontal flux → apply_changelog
3. particle_sim.comp → apply_changelog   (rebaked particles write via changelog)
```

Particle sim runs after fluid so rebaked particles land in stable post-fluid positions.

#### Pillar compatibility

Particles are a **consumer** of the voxel grid, never a writer during their flight. WorldOrchestrator remains the sole writer via changelog. The `_particle_buf` is a new buffer owned by WorldOrchestrator alongside `_voxel_buf_a`/`_voxel_buf_b`. `SimulationRuntime` schedules `ParticleSimulation` as another `VoxelSimulation` subclass (always-active, no chunk sleeping — the particle buffer is global, not chunked).

---

### Phase 10 — World Persistence

**Goal:** Save and restore the voxel world. A player should be able to quit, reopen, and continue exactly where they left off.

#### 9.1 What to save

`Chunk.block_data` is the authoritative CPU state — 4 bytes × 32³ = 131,072 bytes per chunk. Saving is a direct byte dump. Loading is a direct restore into a fresh `Chunk`, followed by `rebuild_counts_and_mask()` to restore the simulation mask.

GPU state (`_voxel_buf_a`) is always derived from `block_data` — it's reconstructed on load by the existing `_add_chunk_to_grid` path. No GPU buffer serialization needed.

#### 9.2 Format

One file per chunk, named by chunk position: `world/chunks/{x}_{y}_{z}.vox`. Binary: 4-byte header (version + chunk_size), then raw `block_data`. Kept simple intentionally — compression, region files, and metadata can be layered on later.

#### 9.3 Save trigger

Explicit player save (key press) + auto-save every N seconds in `WorldOrchestrator._process`. Save runs on a background thread (block_data is read-only from the save path — no mutation can occur during a save because `commit()` is the single writer and runs on the main thread).

#### 9.4 Load path

On startup, if save files exist, `ChunkManager` loads each chunk from disk instead of generating it. The background worker reads the `.vox` file, populates `block_data`, builds collision boxes, then calls `rebuild_counts_and_mask`. No changes to the chunk lifecycle or grid init — loaded chunks enter the world through the same `_pending_results` path as generated chunks.

#### 9.5 What is NOT saved

- Simulation state (`_active_chunks`, `_active` sims) — reconstructed on first `chunks_committed` after load
- GPU buffers — reconstructed from `block_data`
- Temperature — currently stored in `block_data` byte 2, so it IS saved automatically. Deferred: decide whether to persist or reset temperature on load.

---

### Phase 11 — World Streaming (MOVED UP — see Near-Term Roadmap)

**Note:** Streaming is now planned before Phase 7 (Fluid), not after Phase 10. See the Near-Term Roadmap section for the full rationale and design. The technical design below still applies.

**Goal:** The world is larger than what fits in VRAM or RAM at once. Load chunks as the player approaches, unload as they recede. The chunk grid expands from a fixed allocation to a sliding window around the player.

#### 10.1 The constraint

`_voxel_buf_a` and `_chunk_table` are pre-allocated for a fixed `_grid_span`. Moving the player requires the grid window to shift, which invalidates `_chunk_table` indices. This is the structural change that must be designed carefully.

#### 10.2 Approach: rolling grid

Keep `_grid_span` fixed (e.g., 16×8×16 chunks). As the player crosses a chunk boundary, shift `_grid_min` in that direction. Chunks that slide out of the window are saved to disk and their slots freed. Chunks that slide into the window are loaded from disk (or generated) and assigned free slots. The chunk table is rebuilt cheaply: only the rows that changed need updating (one `buffer_update` per shifted chunk position).

The voxel buffer itself does not move — slots are recycled via `_free_slots` exactly as today. Only the chunk table changes on grid shift.

#### 10.3 Grid shift trigger

Fired when `_player_chunk` crosses a boundary by more than 1 chunk from `_grid_min + _grid_span / 2`. The shift is queued and applied asynchronously: chunk saves run on background threads, chunk loads use the existing `_dirty_worker` path. The renderer continues to use the current `_voxel_buf_a` uninterrupted during the shift — chunks that are still loaded are unaffected, and newly-loaded chunks appear as they finish.

#### 10.4 Simulation interaction

Simulations hold chunk positions in `_active_chunks`. After a grid shift, chunk positions that have been unloaded must be evicted from `_active_chunks`. `GranularSimulation` (and future sims) subscribe to a new `ChunkManager.chunk_unloaded` signal and remove the position from their active set. Chunks that have been saved and unloaded mid-simulation are simply dropped — their state is preserved on disk and the simulation restarts from the saved position when the chunk re-enters the window.

---

## 4. Bug Log & Lessons Learned

Moved to [docs/BUG_LOG.md](docs/BUG_LOG.md) (2026-07-02) — this file grew too large as a single document. Read that file for the full history of non-obvious bugs, failed approaches, and GDScript/GPU gotchas. Update it whenever a bug takes more than one session to fix.

---

## 5. Current To-Do

Ordered by priority. Update this section as work is completed or new items are discovered.

### Active / Next Up

- [x] **Fluid never-settles bug** ✅ FIXED & VERIFIED (2026-07-04) — See `docs/BUG_LOG.md` "FIXED (2026-07-04...)" entry for full details. Root cause was Pass A's Y-axis forcing having no hydrostatic equilibrium (`dh*g_dt*gravity_bias - gravity_force`, where a column at perfect rest still generated constant nonzero forcing every tick). Fixed by replacing with elevation/piezometric head, `(dh - FILL_CAP) * g_dt * gravity_bias`, deleting `gravity_force` entirely (export + `SimParams` field, buffer shrunk 56→52 bytes). In-editor verification confirmed a flat settled lake reaches exact zero flux, pouring water spreads and fully settles, and disturbing a settled surface nearby doesn't reawaken an already-stable voxel.
- [x] **Phase UX-1 — Sphere edit tool + player scale** ✅ — See Phase UX-1 section.
- [x] **Collision wireframe debug** ✅ — `collision_debug.gd`, yellow ImmediateMesh boxes, toggled via `RaymarchRenderer.debug_collision_shapes` export. C key binding removed.
- [x] **Immediate collision removal on erase** ✅ — `CollisionSystem.invalidate_sphere()` called from `world.gd` on every remove_sphere event. Box-sphere overlap test strips stale shapes immediately.
- [x] **Collision rebuild batch dispatch** ✅ — `extract_surface_voxels_batch` replaces per-chunk single dispatch. N chunks → 1 GPU stall. Needs profiling to confirm target drop from 170ms to ~5ms.
- [ ] **Collision rebuild — profile after batch fix** — Confirm frame time drops to acceptable level with SYNCS_PER_FRAME=4 batch. — Each `_rebuild_chunk_collision` call is a serial GPU dispatch+sync stall. Need batch surface extract (one dispatch for all dirty-interest chunks) and/or async deferred pipeline. See Collision section for full diagnosis. Player movement on slopes is also jagged — collision shapes don't cover slopes well because the greedy merge produces rectangular boxes only.
- [ ] **Chunk streaming** — Replace the fixed full-world load with a sliding-window chunk loader centred on the player. See Near-Term Roadmap → Step 1 for design. This is the next major architectural milestone.
- [ ] **Simulation distance** — Only tick chunks within a configurable radius of the player. See Near-Term Roadmap → Step 2. Depends on streaming being in place.

### Known Issues

- [ ] **Dirty bitmask overflow is REACHABLE IN NORMAL PLAY (upgraded from theoretical, review 2026-07-01)** — `DIRTY_BUF_BYTES = 64` covers 512 slots, but streaming never unloads columns (`_generated_columns` only grows; `_remove_chunk_from_grid` is never called). Initial load is 49 columns × 8 = 392 slots; generating past 64 total columns (a short walk) crosses 512 slots, `buffer_grow` doubles the voxel buffers, and every chunk assigned slot ≥ 512 has `mark_dirty()` atomicOr past the end of the 64-byte buffer. With robustBufferAccess the write is discarded — those chunks never appear in the changed set, so sand/water in them silently freezes (same symptom family as the Phase 6.5 "pop" bug); without robustness it is UB. Affects `DIRTY_BUF_BYTES` in `granular_simulation.gd` + `fluid_simulation.gd` and binding 7 of `granular_sim.comp` / `fluid_sim.comp` / `fluid_flux.comp`. Fix: size the dirty buffer from `_voxel_buf_capacity` (`ceil(capacity / 32) * 4` bytes) and grow it alongside the voxel buffers.
- [ ] **Cross-chunk diagonal CAS edge case** — Diagonal moves into non-active chunk slots may fail CAS due to stale `voxels_b` data. Rare (requires specific geometry at chunk boundaries), causes at most 1-grain loss per collision, not mass loss.
- [ ] **Simulation mask false positives** — `_simulation_counts` is only incremented (adds from changelog), never decremented. Counts drift high over time. False positives cause 4 extra idle dispatch frames at worst. Corrected on `_sync_chunk_to_cpu` / save. No correctness issue in practice.
- [ ] **`voxel_scale` duplicated as a hardcoded constant, not read from `ChunkManager`** — `Player/player_controller.gd:24` has `const VS := 0.5  # voxel scale - must match ChunkManager.voxel_scale`. Every cursor/placement/indicator calculation uses this local constant instead of reading `ChunkManager.voxel_scale` directly. Changing `ChunkManager.voxel_scale` silently desyncs add/remove placement from the actual voxel grid — edits land at the wrong world position. Needs fixing before any voxel-scale change (e.g. moving to a smaller voxel scale for finer fluid wave detail, see Phase 7e). Fix: read `ChunkManager.voxel_scale` at `_ready()` instead of hardcoding `VS`.
- [ ] **Changelog overflow applies stale entries (review 2026-07-01)** — `reset_changelog()` zeroes only the 4-byte count; the entry array keeps last tick's data. On overflow, a shader thread whose `atomicAdd` allocation spans `MAX_CHANGELOG` (e.g. `log_slot = MAX−1` for a 2-entry move, or a 3-entry displacement) writes nothing, but the CPU clamps count to MAX and `patch_apply.comp` applies entries `[0, MAX)` — the skipped in-range indices still hold stale `(flat_idx, new_value)` pairs from an earlier tick, which get re-applied to `voxels_a` (resurrecting old moves = voxel corruption). Overflow was observed in practice with large spheres (Phase 6.6). Fix: when the downloaded count exceeds `MAX_CHANGELOG_ENTRIES`, skip the patch dispatch entirely and warn — dropping the whole tick is consistent (B is scratch; A simply doesn't advance), partial application is not.
- [ ] **`try_compress` vs. vacating destination — fluid duplication/loss race (review 2026-07-01)** — `fluid_sim.comp`'s "No checkerboard — CAS already prevents races" comment is wrong for one case: CAS protects *destinations*, not *sources that other threads target*. A partial-fill cell can simultaneously (a) be compressed into from above (`try_compress` logs `below_flat → topped-up`) and (b) itself fall into air below (`try_move` logs `below_flat → 0`). Both succeed (different CAS destinations; source writes are unconditional), producing two changelog entries for the same `flat_idx` applied in nondeterministic order: water is either duplicated or the transferred fill destroyed. Common at waterfall fronts. Phase 7e deletes `fluid_sim.comp`, which resolves this — until then, expect volume drift under falling water; don't chase it as a flux bug.
- [ ] **Flux pass 0/1 dispatch-range mismatch leaks volume (review 2026-07-01)** — both flux passes dispatch over `flux_list` (active + 4 horizontal neighbours). A donor cell in a *neighbour* chunk can atomicAdd flux into a chunk two steps from the active set (or a diagonal neighbour), which pass 1 never covers — the donor's `−total_out` is applied but the receiver's `+flux` is discarded (later zeroed when that chunk joins the list). Net water loss at the coverage boundary. Fix: dispatch pass 0 over `active_list` only, pass 1 over `flux_list`; then every pass-0 write lands inside pass-1 coverage. (Moot once 7e replaces `fluid_flux.comp`, but the same coverage rule applies to the 7e Transport pass.)
- [ ] **Surface extraction treats liquids as solid — player walks on water (review 2026-07-01)** — `surface_extract.comp` only filters in-flight GRANULAR voxels via `sim_active`; LIQUID/GAS/EFFECT materials become `is_solid = 1` in `VoxelMover`'s bitmap, violating the collidability rule (NONE + GRANULAR only, §Pillar 1.1). Fix: early-return in the shader when `material_sim.data[mat_id]` is LIQUID/GAS/EFFECT.
- [ ] **Edit changelog cap vs. brush radius (review 2026-07-01)** — `MAX_EDIT_CHANGELOG = 131072` is 1 MB, not "16 MB" as the comment in `edit_dispatcher.gd` claims, and is NOT large enough for a 256³ sphere. The brush allows radius 64 ≈ 1.1 M voxels ≈ 8× overflow. Overflowed voxels still write `voxels_a` correctly, but `extract_changed_chunks` misses their chunks: masks aren't stamped, sims aren't woken, `chunks_committed` omits chunks, and `VoxelMover`'s collision cache goes stale. Fix: derive affected chunks from the edit AABB on the CPU (known exactly — iterate the chunk range the AABB covers) instead of from the changelog.
- [x] **`buffer_grow` copies byte-by-byte in GDScript (review 2026-07-01)** ✅ FIXED (Phase 7e Step 1) — `compute_helper.gd`'s `buffer_grow` used a per-byte `for` loop. Growing past 512 slots copied 64 MB × 4 buffers through GDScript → multi-second hitch. Fixed opportunistically while wiring `_flux_buf`'s own growth path: `resize()` already zero-fills the new tail, so the manual copy loop was pure overhead.
- [ ] **Falsy `Vector3i(0,0,0)` cursor regression (review 2026-07-01)** — `player_controller.gd` `_ray_hit_distance()` (`if rem_pos:`) and `_process` (`if not cursor_voxel:`) truthiness-check the Variant from `get_cursor_remove_pos()`. A cursor hit at voxel (0,0,0) is falsy → treated as no-hit. Same bug class as the documented (0,0,0) chunk bug (§4 Bug Log); use `== null` / `!= null`.
- [ ] **Runtime cross-activation between sims (review 2026-07-01)** — `SimulationRuntime._on_chunks_committed` unconditionally adds every committed chunk to *every* registered simulation (the "mask may be stale" fast-path). Both sims now stamp `mark_simulation_active` on their dirty chunks *before* emitting `chunks_committed`, so the mask is no longer stale for the committed chunk itself — the unconditional add just makes every fluid tick re-activate granular over all water chunks (and vice versa), each costing 4 idle dispatch frames and re-added every tick while the other sim runs. Perf, not correctness. Fix: apply the same mask check to the committed chunk as to neighbours.
- [ ] **Fluid zero pass erases granular's `sim_active` marks (review 2026-07-01)** — `zero_slots.comp` clears `sim_active` for its chunk list; fluid's zero pass runs after granular within the same tick and wipes the "moved this tick" marks for overlapping chunks, so surface extraction sees falling sand as settled (player collides with in-flight grains) whenever fluid is active nearby. Fix: per-sim active buffers, or zero `sim_active` once per tick at the orchestrator level instead of per-dispatch.
- [x] **`_zero_flux_buf` uploads 128 KB of zeros per chunk per sub-step from the CPU** ✅ RESOLVED (Phase 7e Step 4) — `_zero_flux_buf` and its call site are deleted entirely. Flux is now persistent by design (never zeroed between ticks — that's the whole point of Step 4), so there is nothing left to zero on the hot path. Pages still get zeroed once on allocation/eviction (`_ensure_flux_buf_capacity`, `_free_flux_page`), which is orders of magnitude rarer than per-substep.
- [ ] **Brush cannot add material into liquid (review 2026-07-01, gameplay quirk)** — `edit_apply.comp` OP_ADD skips cells occupied by a *different* material, so pouring sand directly into a pond does nothing underwater; sand only interacts with water by falling into it. Decide whether ADD-into-liquid should displace (matches `try_displace` semantics) or stay skipped.
- [ ] **Sand "Z jumping to the top" — reported 2026-07-01, not yet investigated** — user observed sand appears to jump/teleport to the top (of the pile? of the Z axis?) during granular simulation. Reported while testing fluid Phase 7e; unclear if pre-existing or a regression from `_build_flux_chunk_list`'s vertical-neighbour addition (Step 5) or the flux dispatch range change interacting with `granular_sim.comp`'s own zero/dispatch ordering. Needs repro steps and a dedicated look — not touched during Phase 7e work so far, granular_sim.comp has had no changes this phase.

### Future Phases

- [x] **Phase 7 — Fluid (water, Phase 1)** ✅ — `fluid_sim.comp` + `FluidSimulation`, Noita-style whole-voxel CAS, water+oil active
- [ ] **Phase 7b — Fluid fill semantics** — partial fills, `atomicAdd` accumulation, shallow rendering tint, granular-fluid interaction
- [ ] **Phase 8 — Fire & Thermal** — temperature diffusion, ignition, material burn properties
- [ ] **Phase 9 — Gas** — smoke rise, dissipation, fire→smoke production
- [ ] **Phase 10 — World persistence** — save/load per-chunk `.vox` files
- [ ] **Editing UX** — rectangle brush, material HUD, undo queue, fill-plane stamp

---

## 6. Design Principles

### 4.1 Separation of Concerns

- **Voxel storage** stores state. It does not simulate.
- **Material database** defines properties. It does not store state.
- **Simulation** reads state + properties, writes *mutations*. It does not commit directly.
- **Orchestrator** validates, resolves conflicts, and commits. It is the single writer.

### 4.2 Minimal Voxel State

A voxel stores exactly what is unique to its location: material identity, fill amount, temperature, engine flags. Nothing else. Properties belong to materials. Behavior belongs to simulations. Gameplay state belongs to sparse dictionaries.

### 4.3 AIR = 0u

All air voxels are stored as exactly zero in GPU buffers. This is a hard constraint for CAS-based simulation operations. Every field (material_id, fill, temperature, flags) must be zero for air.

### 4.4 Single Writer

Only the Orchestrator writes to `block_data`. Systems submit mutations. The Orchestrator commits. This eliminates race conditions by construction, not by convention.

### 4.5 Incremental Updates

Never rebuild what you can update. Chunk table slots are updated in 4 bytes. Voxel data from player mutations is updated per-chunk (128KB, one `buffer_update` per affected chunk). Voxel data from simulation changelogs is updated per-voxel (4 bytes per entry, one `buffer_update` per changelog entry after Step 10). Grid expansion is rare and cheap. No full-buffer rebuilds.

### 4.6 Read-Only Simulation

Simulation systems run on staging buffers, never on the canonical buffer. They produce changelogs. The Orchestrator commits approved changes. This prevents GPU races and makes the simulation sandboxed and testable.

### 4.7 Deterministic Order

The Orchestrator dispatches simulations in registration order. Systems cannot depend on each other's intra-frame state — they submit mutations independently and the Orchestrator resolves conflicts by priority. Granular is dispatched before fluid; sand sinks through water because granular mutations win on conflict.

### 4.8 Execution Agnosticism

The `VoxelSimulation` interface (`add_candidates` / `step` / `has_work`) makes no assumptions about where computation happens. `GranularSimulation` currently runs on the CPU; Phase 5 moves it to a compute shader without changing the interface, the runtime, or the orchestrator.
