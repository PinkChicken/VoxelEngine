# Voxel Engine — Architecture

**Single source of truth.** This document is the design, the current state, the history of what was
tried and why it failed, and the to-do. Everything else in the repo (control tables, briefs,
handoffs) is either generated from this or explicitly historical.

- **Status:** MLS-MPM is the **only** solver. The CA voxel solvers (granular + fluid) were deleted
  on 2026-08-13 (`mpm-rewrite` @ `7123606`). There is no solver-selection gate.
- **Companion:** `docs/BUG_LOG.md` — the non-obvious bug history (multi-session dead ends, GDScript/
  GPU gotchas). Read it before re-attempting anything that smells familiar. It is a *log*, not a
  design doc; this file is the design.
- **How to run:** see `CLAUDE.md` (commands, editor path, in-editor controls).

---

## 1. What this engine is

A GPU-driven voxel engine in Godot 4.6 (GDScript + GLSL compute, no C#). Voxels are stored in 32³
chunks, rendered by a compute raymarcher (no meshing), edited on the GPU, and simulated by an MLS-MPM
particle solver that writes results back into the canonical voxel buffer.

The design pressure throughout has been **"Falling Everything"** (Noita-like): every material moves
under the same rules, debris is real material, and the world is not player-dependent.

### The three-layer split

| layer | owns | does not know about |
|---|---|---|
| **Storage** (`Chunks/*`) | voxel data, chunk streaming, chunk table | physics, materials' behaviour |
| **Simulation** (`simulations/mpm_simulation.gd`) | one MPM region, particles, grid | chunk streaming policy, rendering |
| **Presentation** (`RaymarchRenderer.gd`, `Player/*`) | pixels, input, camera | how material got there |

The persistent voxel world (`_voxel_buf_a`) is the seam. Simulation writes into it (through the
changelog); rendering reads from it. Nothing else is shared.

---

## 2. Foundational pillars

### 2.1 Voxel storage (`Chunks/chunk_instance.gd`, `chunk_manager.gd`)

- 32³ chunks, **`VOXEL_STRIDE = 1`: one byte per voxel — the material id.** AIR is `0`.
- The GPU buffer holds the same bytes, **packed four to a u32**. Little-endian storage means the
  CPU's byte-per-voxel `block_data` uploads pre-packed, with no CPU pack/unpack step. Shaders index
  it as `data[flat >> 2] >> ((flat & 3) << 3)`; writes are `atomicAnd` + `atomicOr` on one byte,
  which is safe because two threads touching *different* bytes of a word commute.
- This replaced the old 4-byte word `[material_id, fill, temperature, flags]`. `fill` lost its last
  reader (the LIQUID depth tint) and `temperature`/`flags` were never read, so all three were
  retired — voxel-buffer memory and raymarcher bandwidth are both **4× smaller** (`§9.2`).
- `_voxel_buf_a` is the persistent world: the renderer binds it, collision extracts from it, MPM
  writes to it (see §10.2 for what "persistent" means versus particles).
- Per-chunk `_simulation_mask` (a bitmask of which material *classes* are present) and
  `_simulation_counts` are maintained by the edit path (`apply_mutation`,
  `rebuild_counts_and_mask`). This is material-category bookkeeping, **not** solver gating — MPM
  uses it to cheaply find chunks that already contain simulated material.

### 2.2 Materials (`material_def.gd`, `material_db.gd`)

Materials are identity only: phase, simulation class, gravity, color, plus the MPM
physics fields. No simulation state lives in a material. Mirrored to the GPU as two tables:

- **`buffer_rid`** (32 B/material) — color + the CA-era render/sim fields the raymarcher still reads.
- **`mpm_buffer_rid`** (32 B/material) — the MPM dispatch table (§4.4). The only thing that decides
  material behaviour; nothing is hardcoded by material id.

### 2.3 Edits (`simulations/edit_dispatcher.gd`)

Player edits are the only external writer besides the simulation. `EditDispatcher`:

1. validates the request,
2. dispatches `edit_apply.comp` (writes `_voxel_buf_a` directly),
3. emits `voxels_placed(positions, materials)` so MPM can promote the new material into particles,
   and updates the per-chunk simulation masks.

`chunk_loaded` / `voxels_placed` are the signals the rest of the engine listens to. (A
`chunks_committed` signal used to exist; it was removed with the collision surface cache it fed —
see §5.)

### 2.4 Simulation runtime (`simulations/simulation_runtime.gd`)

Fixed 60 Hz accumulator, `MAX_CATCHUP = 1`, `tick_advanced` signal, and the MPM perf-timing fields.
It ticks `MPMSimulation` directly — MPM is a `Node` child of `SimulationRuntime`, not a registered
plugin. There is no plug-in interface and no mode switch.

---

## 3. GPU buffer architecture (`WorldOrchestrator.gd`)

### 3.1 Buffers

| buffer | role |
|---|---|
| `_voxel_buf_a` | canonical voxels; renderer reads, MPM writes |
| `_chunk_table` | `chunk_coord → slot` for the voxel array (persistent) |
| `_changelog_buf` | `[u32 count \| (flat_idx, new_value) × count]`; simulation → voxels |

**CA-era buffer subsystem removed (2026-08-13).** `_voxel_buf_b` (granular CAS scratch),
`_sim_active_buf` (granular in-flight-grain flag), `_material_flow_buf` / `_material_density_buf`
(fluid flux density sorting), and `_material_sim_buf` (a GPU mirror only `surface_extract.comp`
read) are gone — each was allocated at full voxel-buffer size and never written. With them went
`swap_voxel_bufs()`, `sync_block_data_from_buf()`, the `VoxelMutation`/`submit_mutation*`/
`submit_batch`/`commit()` API (zero callers since `EditDispatcher`), `debug_dump_fluid_stats()`
(an 82-line full-buffer readback), and `Chunk.has_simulation()`. `surface_extract.comp` itself was
deleted later, with the whole collision layer (§5).

### 3.2 The write path

```
edit:        edit_apply.comp ────────────────────────────► _voxel_buf_a
simulation:  readout.comp ──► _changelog_buf ──► patch_apply.comp ──► _voxel_buf_a
```

Both hand the changelog to `WorldOrchestrator.apply_changelog()`, which patches `_voxel_buf_a` on the
GPU (never a CPU round-trip). `reset_changelog()` zeroes the count.

### 3.3 The render path

`RaymarchRenderer.gd` dispatches `raymarch.comp` into an output buffer and downloads it. It binds
`_voxel_buf_a`, `_chunk_table`, the material color table, and the debug buffers. The uniform set is
cached and rebuilt only when a bound RID changes.

---

## 4. The MPM solver

### 4.1 Vocabulary

| term | meaning |
|---|---|
| **voxel** | one cell of the world; the canonical, persistent representation |
| **chunk** | 32³ voxels; the world's storage and streaming unit |
| **particle** | an MPM material point; lives only while a region is awake |
| **region** | the simulated volume: ONE contiguous grid + ONE particle set. The unit of cost |
| **requested set** | `_active_chunks`, a dictionary of chunk coords — an intent list, nothing more |

The critical relationship: **chunks are storage; the region is simulation.** That mismatch is the
source of most of the difficulty (and of §4.9's history).

### 4.2 The tick at a glance

```
edit (GPU) ──► voxels_placed(pos, mat) ──► _active_chunks += chunk(pos)
                                                 │
                                                 ▼
                                         _ensure_region()      ← build / grow
                                                 │
 every tick:                                     ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ ×24 substeps:  clear_grid → p2g → grid_op → g2p               │
 │ readout:       grid → voxels, occupancy, dirty chunks, faces   │
 │ _ensure_region()  (rate-limited: once per 10 ticks)           │
 │ idle check → destroy region when quiet                        │
 └───────────────────────────────────────────────────────────────┘
```

### 4.3 Data

**Particle state (per region):** `pos` (3 f, region-local voxels), `vel` (3 f), `C` (9 f, APIC
affine), `J` (1 f, volume ratio for FLUID), `F` (9 f, deformation gradient for GRANULAR/ELASTIC),
`mat` (id), `mass` (`density × p_vol`), `alive` (0 kills without compacting). Capacity is
`max_particles` for the WHOLE region.

**Grid buffers (per region):** `g_mom` (3×int), `g_mass` (int), `g_mass_mat` (8×int), `g_vel`
(3×float), sized `dims.x*dims.y*dims.z` nodes (or `num_blocks × 32³` of dispatch after S1).
Fixed-point integers, not float atomics — portable and bit-deterministic.

**Derived scalars** (computed in ONE place, `_refresh_scales()`, because they must scale together —
see §4.10.2):

| name | value | why |
|---|---|---|
| `dx`, `inv_dx` | 1.0 | one MPM cell = one voxel |
| `p_vol` | `(dx/2)³ × stride³ / subdiv³` = 0.125 default | per-particle rest volume |
| `fp_scale` | `NODE_BUDGET / p_vol`, `NODE_BUDGET = 1e6` | fixed-point conversion |
| `sim_dt` | `sim_seconds_per_tick / substeps` = (1/60)/24 | substep size |
| `gravity` | 9.8 | voxel units / simulated second² |

`NODE_BUDGET = 1e6` is a *correctness* constraint, not a tuning knob: at 1e7 the int32 `g_mom`
accumulator wraps above ~190 voxel/s and the body spazzed at ~2400 voxel/s. The `max_speed` clamp
(200 voxel/s) also exists because of that failure.

### 4.4 The MPM material table (`mpm_buffer_rid`)

32 bytes per material id, read directly by `p2g/grid_op/g2p/readout`:

| offset | field |
|---|---|
| 0 | `mpm_model` (0 none, 1 static, 2 fluid, 3 granular, 4 elastic) |
| 4 | `mpm_density` |
| 8 | `mpm_stiffness` (E) |
| 12 | `mpm_viscosity` |
| 16 | `alpha_sand` (Drucker-Prager) |
| 20 | `mu_sand` (Lamé μ, derived from E) |
| 24 | `lambda_sand` (Lamé λ, derived from E) |
| 28 | reserved |

Derived fields (20/24) are computed on the CPU in `_encode_mpm_blob()` so the conversion lives in one
place. The water physics sliders (`water_density/stiffness/viscosity`) drive `override_mpm(3, …)`,
which updates this table in place.

### 4.5 Region lifecycle

**What creates a request** — three paths, all adding to `_active_chunks`:

1. `_on_voxels_placed(positions, materials)` — edits.
2. `scan_loaded_chunks_for_existing_material()` — at `setup()`, adds loaded chunks whose
   `_simulation_mask` reports GRANULAR|LIQUID.
3. `_on_chunk_loaded(cp)` — same test as chunks stream in.

**`_ensure_region()`** — called from the edit path AND once per tick (rate-limited to one move per 10
ticks).

*First build* (`_region == null`): bbox of requested chunks → `span`;
`start = clamp(max(region_chunks, span), 1, max_axis_chunks)` per axis;
`origin = (bb.min - (start-span)/2) * n_grid`, `dims = start * n_grid`. `_build_region()` allocates
and `promote_existing_voxels()` seeds one particle per MPM voxel.

*Subsequent calls* are driven by `w.occupied` — the per-chunk occupancy mask `readout.comp` produced
from ACTUAL particle mass last tick. Any axis whose occupied chunks touch a face grows one chunk that
way, capped by `max_axis_chunks` and `MAX_REGION_NODES`.

Both are **full rebuilds**: new buffers at the new size, particles copied across with a position
offset (`_copy_particles_shifted`). Nothing is inserted incrementally. In-flight velocities are lost
across a rebuild — acceptable and rare (only on chunk-set changes).

**Sleep:** if the readout produced no changed voxels for `idle_threshold` (4) consecutive ticks AND
`max_speed < rest_speed` (0.02), the region is destroyed and `_active_chunks` cleared. Material
survives as voxels; a later edit re-activates it. There is no per-chunk deactivation — see §4.8.

Note: with `viscosity 3` and no surface tension, a residual ~0.14 voxel/s jitter floor means the
region never fully sleeps in practice. This is known, not a bug.

**Known limitation — the sleep/wake bridge loses velocity and re-quantises mass.**
`readout` writes material identity per cell (no mass, no velocity, no `C`/`J`/`F`).
`promote_existing_voxels()` seeds **one full-mass particle per occupied cell** on wake
(`mass_each = density · p_vol`). So the bridge is *material-exact* but *mass-quantised*: a thin
film spread over several cells becomes a full particle in each on wake. "Mass is conserved by
construction" is therefore a **particle-domain** claim — it holds within one region life, not
across a destroy → re-promote cycle. The S0/S1 exact-mass results were all measured inside a single
life and do not exercise this.

### 4.6 The five passes

All five are loaded as text from `shaders/mpm/*.comp` at runtime (a `.comp` edit needs a full game
restart, not a reimport).

**`clear_grid.comp`** — zeroes `g_mom`, `g_mass`, `g_mass_mat`. Consequence: the grid carries NO
state between substeps; it is a scratch accumulator rebuilt from particles every substep. Empty cells
are the normal state, not missing data.

**`p2g.comp`** — per particle: build the stress tensor, scatter mass/momentum to its 3×3×3
neighbourhood with quadratic B-spline weights.
- FLUID: `s = -dt·4E·p_vol·(J-1)·inv_dx²` + viscosity shear `-dt·4ν·p_vol·inv_dx²·(C+Cᵀ)`.
- GRANULAR: SVD of `F`, Drucker-Prager return mapping, σ clamped to [0.05, 5.0], Hencky strain.
- ELASTIC: zero stress (not ported; no material uses it).
- STATIC/NOT_SIMULATED: skipped.
- Accumulation is fixed-point int32 atomics.

**`grid_op.comp`** — per node: `vel = mom/mass`, gravity on every node, speed clamp, **FACE PUSH**
(diagnostic occupancy of outward-velocity faces), and the `bound`-cell boundary band that zeroes
outward velocity.

**`g2p.comp`** — gather velocity + APIC affine `C` from the 3×3×3 neighbourhood; `pos += dt·vel`
clamped to `[0, dims)`; FLUID `J *= 1 + dt·trace(C)`; GRANULAR/ELASTIC `F ← (I + dt·C)F`.

**`readout.comp`** — per node: argmax over `g_mass_mat` for the dominant material, write the
material id through the shared changelog. Guards: never overwrite STATIC voxels; a cell only writes
when its *material* changes (no mass quantisation, so a settled pool's jitter registers nothing and
the idle timer can reach zero); sets occupancy and face bits.
Suspended entirely if `n_particles >= max_particles` (writing air over material the sim cannot
represent would destroy it).

### 4.7 Sparse grid (S1)

The grid is block-paged: 32³ blocks, one per chunk. `block_map` (dense→slot, -1 absent) and
`block_list` (slot→dense) let the four grid passes dispatch only allocated blocks. The active block
set is the readout's occupancy mask ∪ 1-block halo, compacted on the CPU each tick.

**Phase 1 (block-indexed addressing, all blocks allocated)** and **Phase 2 (drop empty blocks)** are
both DONE and correctness-verified (exact mass conservation on every run). Phase 2 deviated from the
"new mark pass" plan: it reuses the occupancy mask rather than adding a shader.

**S1 does NOT deliver** (honest status):

1. **Multiple regions** — placing water in a separate location still cannot activate a second
   region; the request goes into `_active_chunks` but the single region ignores it.
2. **Growth toward far edits** — `_growth_for` only reacts to occupancy INSIDE the window, so water
   placed outside the window is never seen and never simulated.
3. **Sparse memory** — buffers are still sized to the dense bbox; only the dispatch bound shrank. The
   memory win is a deferred follow-up.

**Acceptance criterion:** for the same particle state, sparse MPM must produce the same physical
result as dense MPM, while grid cost follows occupied nodes rather than the bounding box. "Sparse ≈
dense" is the whole point — it gives a continuously-comparable reference.

**Absolute constraint:** chunks/blocks are NEVER independent MPM domains. Per-block grids each had
their own index space so a particle's 3×3×3 stencil could not cross a boundary (four walls per
chunk). The sparse grid is ONE global index space with holes; the block only decides *where a node is
stored*, never *whether it exists*.

### 4.8 The hard constraint: chunks ≠ region granularity

The deleted CA solver had per-chunk activation (a voxel moving into a chunk added it; a resting chunk
was removed). MPM cannot do that: P2G scatters and G2P gathers through one shared accumulator, so a
chunk cannot be "off" while its neighbour is "on" — a particle crossing the boundary must find the
destination nodes allocated and contiguous.

| | CA (deleted) | MPM |
|---|---|---|
| activation unit | chunk | region (contiguous volume) |
| cost of N chunks | N × 32³ | bbox³ |
| deactivation | per chunk, on rest | whole region only |
| boundaries | between chunks | one hard boundary around the region |

The cost difference is the whole problem: 8 chunks in a 2×2×2 arrangement is 64³ = 262,144 nodes, not
8 × 32,768.

### 4.9 History — every allocation strategy tried

Recorded because several look reasonable in isolation and were not. All of this machinery has been
deleted from the tree; if a strategy is re-tried, start from here.

1. **Per-chunk independent windows** — failed, four walls at every chunk boundary (§4.8).
2. **Shared region + growth on "chunk changed"** — failed; a chunk is 32 deep so a body anywhere
   inside an outer chunk counted as touching the wall. Measured 2→3→4→5 chunks over one run.
3. **Growth on mass at the exact boundary plane** — never fired; the damping band holds material
   `bound` cells short of the plane.
4. **Growth on mass in the band** — worked, then ran away; a pool at rest touches every wall of its
   box by definition. Measured 128×64×128 at the node cap.
5. **Growth on outward velocity at a face** — ran away again; pressure on a wall is
   indistinguishable from pressure into free space.
6. **Fixed window anchored on flood-fill centroid** — destroyed the body; the window chases the
   sliver still inside and abandons the bulk.
7. **Fixed window, edge-follow** — worked, but material still hit the wall with only 4 active chunks.
8. **Occupancy-based chunk release** — the release rule was true of resting water, so it shrank over
   settled pools and froze material mid-air.
9. **Current: smallest region that fits the edit, bounded growth** — `region_chunks` is a floor, the
   first build sizes to the edit's span, and growth expands up to `max_axis_chunks` /
   `MAX_REGION_NODES`.

The lesson: **one dense allocation trying to decide its own extent from wall contact** is the root
disease. The sparse grid (S1) removes it.

### 4.10 Theories, with evidence

**10.1 Free-surface pressure was "under-resolved" — RESOLVED, was not a defect.** The original
symptom (bodies filming out instead of holding depth) was the momentum overflow (§4.3's `NODE_BUDGET`
note) masquerading as a free-surface defect. Once the accumulator was fixed and viscosity added, the
body forms a coherent blob and levels out as water should.

**10.2 The `particle_subdiv` freeze — was a bug, fixed.** Dividing mass by `sub³` without dividing
`p_vol` left each particle claiming a full voxel's pressure response for 1/8 the mass; effective
stiffness became `E × 8`, past the stability point. `p_vol` is now derived in one place and
everything scales together. Recorded because the first result looked like a finding about MPM and was
purely arithmetic.

**10.3 Stiffness has an upper stability bound, not a tuning range.** Sweep E = 500 → 128,000:
character changes at 500, but from baseline upward the endpoint is identical (fills the window, every
face lit). Too coarse a substep or too stiff a material and `J` diverges.

**10.4 No surface tension.** MPM water has no cohesion term, so it does not bead — small quantities
film out. Real, cheaply observable, a contributor rather than the root cause.

**10.5 S0 CLOSED (basin test).** The discriminator was "does water HOLD DEPTH in a container":

```
alive=13648 mass=1706.00 depth=27.0 footprint=30x30 max_speed=0.0000 compactness=0.90
```

`compactness 0.90` and `max_speed 0.0000` is a compact, resting, contained body — exactly what S0
asked for. Regression oracle: **water density 1.0 / stiffness 3000 / viscosity 3.0**, sphere → basin.

### 4.11 Measured cost

Same hardware (RTX 5070), 24 substeps, water, single body:

| window | nodes | particles | ms/tick |
|---|---|---|---|
| 32³ | 32,768 | 5,575 | ~0.3 |
| 64³ | 262,144 | 5,575 | **2.4** |
| 64³ | 262,144 | 44,600 (subdiv 2) | **6.4** |
| 96³ | 884,736 | 5,575 | **~5** |
| 128³ | 2,097,152 | 5,575 | **8–17** |

Two conclusions that held up:

1. **Grid cost is linear in nodes** and dominates at low particle counts — a single chunk of water
   is not what costs 2.4 ms, the box is.
2. **Particle cost becomes significant at high subdiv** — 8× particles took 2.4 → 6.4 ms.

Cost per tick vs particle count (nodes fixed at 262,144, 24 substeps): ~2.8 ms at 8.9k particles,
~8.2 ms at 63k, ~13.4 ms at 100k. Cost is ~linear in particles; 100k ≈ 13 ms is ~80% of a 60 Hz frame
before rendering.

Window moves reallocate every buffer: measured hitch of **34–91 ms** at 34–120 MB, which is why moves
are rate-limited to one per 10 ticks.

### 4.12 Knobs

All are `@export` and editable on a running game via the remote Inspector
(`Debugger → Remote Scene Tree → WorldOrchestrator/SimulationRuntime/MPMSimulation`). Defaults in
parentheses; `world.tscn` overrides several.

| export | default | meaning |
|---|---|---|
| `n_grid` | 32 | chunk size / grid block size |
| `gravity` | 9.8 | voxel units / simulated second² |
| `sim_seconds_per_tick` | 1/60 | substep dt is derived |
| `max_particles` | 65,536 (scene 100,000) | region-wide particle capacity |
| `promote_stride` | 1 (scene 2) | one particle per N³ voxels (8× water per particle at 2) |
| `particle_subdiv` | 1 | particles per voxel per axis (2 = 8× particles) |
| `region_chunks` | (1,1,1) (scene 2,2,2) | MINIMUM window, in chunks — not the size |
| `max_axis_chunks` | 3 | per-axis ceiling (3 = up to 96³) |
| `bound` | 1 | boundary damping band width, cells |
| `idle_threshold` | 4 | quiet ticks before sleeping |
| `rest_speed` | 0.02 | sleep speed threshold |
| `water_density` | 1.0 | → MPM table, rebuild |
| `water_stiffness` | 8000 (scene 3000) | stability ceiling applies |
| `water_viscosity` | 0.0 (scene 3.0) | |

Constants not exposed: `MAX_REGION_NODES` (1,048,576), `NODE_BUDGET` (1e6), `NMAT` (8),
`DIRTY_BUF_BYTES`/`OCCUPANCY_BYTES` (64), `PUSH_SPEED` (0.5, grid_op), `MASS_THRESHOLD` (1e-6),
the speed clamp (200 voxel/s).

---

## 5. Player movement — INTERIM: free-fly, no collision

**There is no collision system.** It was removed deliberately (2026-08-13) so the MPM solver could be
made solid without a half-working physics layer in the way. What went:

- `Player/voxel_mover.gd` — the CPU AABB mover + its per-chunk surface bitmap cache.
- `shaders/surface_extract.comp` — the GPU surface extractor that fed that cache.
- `WorldOrchestrator`: `_surface_*` buffers, `_init_surface_extract()`,
  `extract_surface_voxels_batch()`, the `SURFACE_*` constants.
- The `chunks_committed` signal (its only consumer was the surface cache; it had a known gap
  anyway — MPM's own voxel writes never fired it, so rising water was invisible to collision).
- `RaymarchRenderer`'s `_collision_chunks_buf` + `raymarch.comp`'s collision-tint debug block
  (which was already dead — no flag bit ever enabled it).
- `Chunk` is now a plain `Node3D` (was `StaticBody3D`); `chunk.tscn`'s empty `CollisionShape3D` is
  gone. The `jump` / `toggle_fly` input actions are gone with the ground-movement path.

**What remains:** `player_controller.gd` free-flies — WASD in the pivot plane, Space/Ctrl for
vertical, Shift to sprint, mouse to look. It is a debug camera, not a game character: it passes
through everything.

**What this does NOT mean:** the world is still fully simulated and the player still edits it (the
cursor raycast, sphere placement, remove are all live). Only *the player's own motion* stopped
interacting with the world.

**The open design question** — how the player interacts with the simulation — is deliberately
deferred until MPM is solid. The expected direction is collision/resolution driven by MPM state
rather than a separately-extracted surface bitmap, which is why this was cut now rather than patched:
a direct-MPM query needs the region addressing that dynamic region building will build.

---

## 6. Rendering (`RaymarchRenderer.gd`, `shaders/raymarch.comp`)

- Compute raymarcher: one dispatch per frame into an output buffer, downloaded and blitted.
- Binds `_voxel_buf_a`, `_chunk_table`, the material color table, cursor/debug/active-chunk buffers.
- Uniform set cached; rebuilt only when a bound RID changes.
- Debug overlays: chunk borders, active-chunk tint, material hide (bitmask).
- The CA punch "density_touch" tint (binding 11) was removed with the CA solvers.

---

## 7. Cutover record — why MPM, and what was deleted

**Decision (2026-08-13):** keep MPM, migrate to it, delete the CA solvers once MPM owned the
materials. Finishing a cutover already committed to, not a third design.

**The line count:** CA system 4,835 lines (`fluid_flux.comp` 1,521, `fluid_simulation.gd` 1,588,
`granular_simulation.gd` 580, `momentum_resolve.comp` 533, `granular_sim.comp` 337,
`demand_resolve.comp` 276) vs the MPM replacement 1,190 (5 shaders + `mpm_simulation.gd`).

**The real reason — the bug class:** the recurring failures were not physics errors but bookkeeping
errors moving integer fill amounts through a sparse buffer under CAS + changelog + partial failure:
changelog overflow applying stale entries → corruption; `try_compress` vs vacating-destination race →
duplication/loss; flux pass 0/1 range mismatch → volume loss; punch demand-queue overflow → ~20% mass
erased; `momentum_resolve` `touched_cells` capacity → silent unreachable accumulation; granular/fluid
`sim_active` erasure, cross-activation, cross-chunk diagonal CAS.

MPM removes this class structurally: mass is the sum of particle masses, so it is conserved by
construction, and there is no changelog, no CAS conflict resolution, and no demand queue.

**The specific complexity deleted:** `fluid_flux.comp` grew 609 → 1,521 lines (+224 phi-flood
pressure, +471 liquid-interaction punch/buoyancy/fill-history), plus `momentum_resolve` (533) and
`demand_resolve` (276) — ~1,400 lines whose entire job was density sorting and momentum displacement.
Under MPM, "which materials sort against each other" disappears because sorting is emergent.

**Sand migrating was the payoff, not a cost:** `granular_simulation.gd` + `granular_sim.comp` was 917
lines with its own bug list, and the granular↔fluid seam was itself a documented bug source
(`sim_active` erasure, cross-activation, a hardcoded dispatch-order hack: "sand sinks through water
because granular mutations win on conflict"). MPM sand (Drucker-Prager) removes the seam.

**What does not migrate:** FIRE (no conserved mass) and SMOKE (viable but a cost question) stay on
separate simple paths. Both are unbuilt today, so nothing was lost.

**Deleted at S5 (`7123606`):** `simulations/voxel_simulation.gd`, `granular_simulation.gd`,
`fluid_simulation.gd`; `shaders/granular_sim.comp`, `fluid_flux.comp`, `momentum_resolve.comp`,
`demand_resolve.comp`, `zero_slots.comp`; the `SolverMode` gate (`-- mpm` / `-- ca`); the
`GranularSimulation`/`FluidSimulation` nodes in `world.tscn`; the CA-only debug keys.

**Kept deliberately:** `docs/BUG_LOG.md` (institutional memory — the asset the rewrite is for),
`edit_apply.comp`, `patch_apply.comp`, the raymarcher, `ChunkManager`, `WorldOrchestrator`, and all
solver-agnostic infrastructure. (`surface_extract.comp` and `VoxelMover` were kept at S5 but have
since been deleted with the collision layer — §5.)

---

## 8. Current to-do / next architecture decision

**Done:** CA deletion; S0 (solver validated, basin test); S1 phase 1 + phase 2 (sparse dispatch,
correctness verified).

**Next major project: dynamic region building** (once this documentation is in order). The
single-region limitation is the real blocker for "place water anywhere in a dynamic world". Options,
in the order the pillars favour them:

1. **Region expansion toward far edits** — grow the window to include a far edit (bounded by
   `MAX_REGION_NODES`), so water placed outside the window is at least discovered and simulated.
   Cheapest, and keeps ONE index space (§10.1).
2. **Multiple regions** — one sparse region per connected component. This is §4.8's seam at a
   coarser scale (two touching regions need one index space), so it needs a merge-on-contact rule.
3. **World-spanning region with raised bounds** — simplest, most memory-hungry.

Also open:
- **Sparse memory resize** — shrink grid buffers to `num_blocks × 32³` (captures the memory win S1
  phase 2 deferred).
- **Mass-aware promotion** — §4.5's sleep/wake bridge re-quantises mass (one full particle per
  occupied cell on wake). Measure the drift, or track per-cell mass to make wake exact.
- **Sand repose validation** alongside water.
- **Optimization pass** (§9) — re-derive; the CA-derived costs are gone.

**Orthogonal, deliberately deferred:** resolution LOD (coarsen grid + particle size together, 2×
coarser = 8× fewer nodes for the same volume). `promote_stride = 2` already proved the coarse-
resolution lever in embryo.

---

## 9. Optimization pass — re-scoped

The original inventory (`§7` of the pre-consolidation doc) was written against the CA solver and is
now obsolete: the phi-flood and the ~60 submit/sync round-trips per fluid substep disappeared with the
CA code. What remains, ranked:

**9.0 Measure before cutting.** There is a per-tick MPM wall-clock (`last_mpm_tick_usec`, `G` key
averaging) and nothing else. Add per-pass GPU timing before optimizing against estimates.

**9.1 Region-move hitches.** A rebuild reallocates every grid buffer: measured 34–91 ms at 34–120 MB.
Moves are rate-limited to one per 10 ticks. Growing the buffers in place instead of free+realloc, or
doing the rebuild on a worker, is the fix if it bites.

**9.2 Ramarch: no empty-space skipping.** The raymarcher marches every pixel through the chunk table
without skipping empty space. This is the biggest per-frame cost and independent of the solver.

**9.3 Particle cost at high density.** ~linear in particles; 100k ≈ 13 ms. Levers in order:
`promote_stride` (8× water per particle at stride 2), fewer substeps (trades the stability ceiling),
then LOD.

**9.4 Sparse memory.** S1's deferred buffer resize.

**9.5 CPU hot spots.** Region/occupancy bookkeeping per tick; lower priority.

---

## 10. Design principles

Eight invariants. These are not aspirations — each one either already constrained a decision or
explains a documented dead end. Read §4.9 before violating one.

**10.1 Chunks are storage; the region is simulation.** The two have different granularity and must
never be conflated. A chunk is a streaming/allocation unit; the region is ONE contiguous grid + ONE
particle set with ONE index space. Per-chunk scheduling *inside* a region is the §4.9 disease.

**10.2 Voxels are the persistent world; particles are working state; the bridge is lossy and is
accounted for.** `_voxel_buf_a` is the canonical *persistent* representation, not the truth while a
region is awake — the particle set is. `readout` projects mass into voxels (quantised, deadbanded,
carrying no velocity/`C`/`J`/`F`), and promotion seeds particles from voxels without reading the
projection's own fields. Every sleep/wake or rebuild crosses this lossy bridge; conservation claims
are particle-domain, not world-domain.

**10.3 The world is not player-dependent.** Activation is chunk/material-driven (`setup()` scans all
loaded chunks; `chunk_loaded` keeps scanning). Nothing may centre a region on the player or pause
simulation for being off-screen.

**10.4 Bounded by construction; refusal is a first-class outcome.** `max_axis_chunks` and
`MAX_REGION_NODES` refuse rather than over-allocate. A refused edit means "that chunk silently does
not simulate" — a deliberate correctness trade that must surface (warning + the `Y` dump), never a
quiet stall.

**10.5 GPU simulates, CPU schedules.** Simulation and edits are GPU passes, and the changelog is
never downloaded. But the per-tick active-block set *is* built on the CPU from the downloaded
occupancy mask (64 B) and re-uploaded — a real per-tick GPU→CPU→GPU round trip. Bounded and
intentional; it is "GPU simulates, CPU schedules", not "GPU-primary".

**10.6 Materials are identity, not state; a voxel is one byte: `material_id`.** A material holds
properties; a voxel holds identity. The old `fill`/`temperature`/`flags` bytes are gone (§2.1) —
none of them were ever engine state. The MPM dispatch is table-driven (`mpm_buffer_rid`), nothing
hardcoded by material id.

**10.7 AIR = 0u as an empty-test invariant.** All air voxels are exactly zero. The original rationale
(CAS) died with the CA solver; it survives because it makes "is this voxel empty" a 32-bit zero test
and lets zeroed memory mean air.

**10.8 In-editor-first discipline.** The agent cannot load a Godot project (headless hangs), so every
shader/script change is static-review-only until a human runs it. §4.9's dead ends and
`docs/BUG_LOG.md` record what skipping this costs.

**Scoping notes on the retired pillars.** "Incremental updates" still governs voxel *content*
(`patch_apply.comp`, per-voxel) but **not** region *geometry*, which is a full rebuild by design and
rate-limited (§4.5). "Single writer" is now "single *commit* path": edits and simulation both land
through the changelog/patch step. "Conservation by construction" is particle-domain only (§10.2,
§4.5).

---

## 11. Debug tooling

### In-editor keys (`world.gd` unless noted)

| key | action |
|---|---|
| `P` | pause / unpause |
| `→` | single-step one tick while paused |
| `T` | cycle `sim_seconds_per_tick` (1/60, 1/120, 1/240, 1/480, 1/30) |
| `O` | cycle water physics presets (drives the same exports) |
| `X` | hide the selected material in the renderer only (bitmask; multiple at once) |
| `C` | active-chunk tint (shows grid coverage) |
| `Y` | region dump: origin/dims/nodes, requested vs occupied vs covered, particles, idle_ticks, face mask, live water physics, timescale, last tick ms |
| `B` | spawn a radius-11 WATER body above the player (prints voxel + chunk) |
| `G` | 60-tick perf averaging |
| `Z` | drop region + active set, keep voxels (clean test state) |
| `R` | full world regen (`Player/player_controller.gd`) |

Log prefixes: `[MPM]`, `[MPM perf]`, `[MPMStress]`, `[MPM params]`, `[SimRuntime]`, `[MPM-S0]`.

### Snapshot/replay discipline

`Z` (drop the region, keep voxels) plus pause/step is the standard way to isolate one voxel's
behaviour. Debug tools must **never mutate voxels to visualize something** — an earlier
hide-material tool that cleared voxels to air was rejected for exactly that reason.

---

## 12. File map

| file | role |
|---|---|
| `simulations/mpm_simulation.gd` | the whole MPM driver: region lifecycle, buffers, uniforms, tick |
| `shaders/mpm/clear_grid.comp` | zero grid accumulators |
| `shaders/mpm/p2g.comp` | particle → grid scatter, all material stress models |
| `shaders/mpm/grid_op.comp` | grid velocity, gravity, face push, boundary band |
| `shaders/mpm/g2p.comp` | grid → particle gather, state advance |
| `shaders/mpm/readout.comp` | grid → voxels, occupancy, dirty chunks |
| `simulations/simulation_runtime.gd` | tick scheduling, MPM perf timing |
| `simulations/edit_dispatcher.gd` | player edits → `edit_apply.comp` + signals |
| `material_db.gd` | the 32 B/material MPM + color tables |
| `material_def.gd` | per-material physics fields |
| `WorldOrchestrator.gd` | canonical buffers, changelog, `apply_changelog`, chunk table |
| `RaymarchRenderer.gd` | compute raymarcher + debug overlays |
| `shaders/raymarch.comp` | the render shader |
| `shaders/edit_apply.comp` | player-edit writes |
| `shaders/patch_apply.comp` | changelog → canonical buffer |
| `Player/player_controller.gd` | free-fly camera + edit input |
| `world.gd` | test hotkeys, region dump |
| `world.tscn` | the running scene |

---

## 13. Where the history lives

- **`docs/BUG_LOG.md`** — every non-obvious bug, rejected approach, and GDScript/GPU gotcha. Add an
  entry whenever a bug takes more than one session.
- **This file** — design, current state, the strategy history (§4.9), and the to-do.
- Local-only (gitignored): `tools/fluid_prototypes/MPM_EVALUATION_BRIEF.md` (the evaluation that chose
  MPM), `tools/mpm_prototype/` (Taichi benches), `MPM reference.pdf`.
