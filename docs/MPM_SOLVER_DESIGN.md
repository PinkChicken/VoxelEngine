# MPM Solver — Design Reference

Everything the solver does today, every allocation strategy already tried, and what the
measurements actually say. Written 2026-08-13 because the region/allocation history had
been layered rather than tracked, and intuition built on a half-remembered pipeline is
worse than no intuition.

Companion documents: `ARCHITECTURE.md` (whole engine), `docs/MIGRATION_MPM.md` (the CA→MPM
cutover plan), `docs/BUG_LOG.md` (failure history), and the local-only
`tools/fluid_prototypes/MPM_EVALUATION_BRIEF.md` (the evaluation that chose MPM).

---

## 1. Vocabulary

| term | meaning |
|---|---|
| **voxel** | one cell of the world. The canonical, persistent representation. |
| **chunk** | 32³ voxels. The world's storage and streaming unit. |
| **particle** | an MPM material point. Lives only while a region is awake. |
| **region** | the simulated volume: ONE contiguous grid + ONE particle set. The unit of cost. |
| **requested set** | `_active_chunks`, a dictionary of chunk coords. An intent list, nothing more. |

The critical relationship: **chunks are storage; the region is simulation.** They are not
the same granularity, and that mismatch is the source of most of the difficulty in §7.

---

## 2. The tick at a glance

```
edit (GPU)  ──► voxels_placed(pos, mat)  ──► _active_chunks += chunk(pos)
                                                    │
                                                    ▼
                                            _ensure_region()   ← build / grow / shift
                                                    │
  every tick:                                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ clear face mask                                              │
  │ ×24 substeps:  clear_grid → p2g → grid_op → g2p              │
  │ readout:       grid → voxels, occupancy, dirty chunks, face   │
  │ _ensure_region()  (rate-limited: once per 10 ticks)          │
  │ idle check → destroy region when quiet                       │
  └──────────────────────────────────────────────────────────────┘
```

---

## 3. Data

### 3.1 Voxel storage (persistent, solver-agnostic)

32³ chunks, 4 bytes per voxel: `[material_id, fill, temperature, flags]`. AIR is exactly
`0u`. `_voxel_buf_a` is canonical and always what the renderer and collision read. The same
layout serves every phase; LIQUID reinterprets byte 1 as pressure in the CA path (unused by
MPM).

### 3.2 Material table (`MaterialDB.mpm_buffer_rid`)

32 bytes per material id, read directly by `p2g.comp` / `g2p.comp` / `grid_op.comp`:

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

Nothing about behaviour is hardcoded by material id — this table is the only dispatch.
Fields 20/24 are derived on the CPU in `_encode_mpm_blob()` so the conversion lives in one
place.

### 3.3 Particle state (per region)

| buffer | contents |
|---|---|
| `pos` | 3 floats, region-local voxel units |
| `vel` | 3 floats |
| `C` | 9 floats, APIC affine velocity gradient |
| `J` | 1 float, volume ratio (FLUID) |
| `F` | 9 floats, deformation gradient (GRANULAR/ELASTIC) |
| `mat` | material id |
| `mass` | `density × p_vol` |
| `alive` | 0 kills a particle without compacting the array |

Capacity is `max_particles` (default 65,536) for the WHOLE region, not per chunk.

### 3.4 Grid buffers (per region)

`g_mom` (3×int), `g_mass` (int), `g_mass_mat` (8×int), `g_vel` (3×float), sized
`dims.x*dims.y*dims.z` nodes. Fixed-point integers, not floats — portable atomics, see §5.2.

### 3.5 The derived scalars

| name | value | why |
|---|---|---|
| `dx`, `inv_dx` | 1.0 | grid locked to voxel resolution: one MPM cell = one voxel |
| `p_vol` | `(dx/2)³ × stride³ / subdiv³` = 0.125 by default | per-PARTICLE rest volume |
| `fp_scale` | `NODE_BUDGET / p_vol`, `NODE_BUDGET = 1e7` | fixed-point conversion |
| `sim_dt` | `sim_seconds_per_tick / substeps` = (1/60)/24 = 6.94e-4 | substep size |
| `gravity` | 9.8 | voxel units / simulated second² |

`p_vol` and `fp_scale` are computed in ONE place (`_refresh_scales()`) because every one of
them must scale together — see §9.1 for the bug that proved it.

---

## 4. Region lifecycle — the exact pipeline

### 4.1 What creates a request

Three paths, all adding to `_active_chunks`:

1. **`_on_voxels_placed(positions, materials)`** — the edit path. `EditDispatcher` dispatches
   `edit_apply.comp` (which writes `_voxel_buf_a` directly), then emits this signal with the
   individual placed voxel coords. Each voxel's chunk is added.
2. **`scan_loaded_chunks_for_existing_material()`** — at `setup()`, adds any loaded chunk
   whose `active_simulation_mask()` reports GRANULAR|LIQUID.
3. **`_on_chunk_loaded(cp)`** — same test as chunks stream in.

`_active_chunks` is capped by `max_active_chunks` (16).

### 4.2 `_ensure_region()` — where the volume is decided

Called from the edit path AND once per tick (rate-limited to one move per 10 ticks).

**First build** (`_region == null`):
- bbox of the requested chunks → `span`
- `start = clamp(max(region_chunks, span), 1, max_axis_chunks)` per axis
- `origin = (bb.min - (start - span)/2) * n_grid`, `dims = start * n_grid`
- `_build_region()` allocates every buffer, then `promote_existing_voxels()` scans those
  chunks and seeds one particle per MPM voxel.

**Subsequent calls** (`_region != null`), driven by `w.occupied` — the per-chunk occupancy
mask that `readout.comp` produced from ACTUAL particle mass last tick:
- `grow_window == true` → `_growth_for()`: any axis whose occupied chunks touch a face grows
  one chunk that way, capped by `max_axis_chunks` and `MAX_REGION_NODES`.
- `grow_window == false` → `_edge_shift()`: translate one chunk toward the touching face.

Both are **full rebuilds**: new buffers at the new size, particles copied across with a
position offset (`_copy_particles_shifted`). Nothing is inserted incrementally.

### 4.3 The substeps (×24 per tick)

`_run_region_substeps()` runs, per substep: `clear_grid → p2g → grid_op → g2p` as one
compute list, with a single submit + sync for the whole batch.

### 4.4 Readout (once per tick, after all substeps)

`_readout_region()`:
- zero the dirty-chunk bitmask, occupancy mask, and face mask
- dispatch `readout.comp`
- `apply_changelog()` — patches the writes into `_voxel_buf_a` on the GPU
- download 64 B occupancy + 4 B face mask

### 4.5 Sleep

If the readout produced **no changed voxels** for `idle_threshold` (4) consecutive ticks
**and** max particle speed < `rest_speed` (0.02), the region is destroyed and
`_active_chunks` cleared. Material survives as voxels; a later edit re-activates it.

**There is no per-chunk deactivation.** Individual chunks are never retired from a live
region — see §6.

---

## 5. The five passes

### 5.1 `clear_grid.comp`

Zeroes `g_mom`, `g_mass`, `g_mass_mat` for every node. Dispatch = `dims product / 64`.

**Consequence that matters:** the grid carries NO state between substeps. It is a scratch
accumulator rebuilt from particles every substep. Empty cells are therefore not "missing
data" — they are the normal state of most of the grid. Appending a chunk of empty space
mid-run cannot corrupt anything, because there is nothing there to corrupt.

### 5.2 `p2g.comp` — particle → grid

Per particle: build the stress tensor from the material model, then scatter mass and
momentum to its 3×3×3 node neighbourhood with quadratic B-spline weights.

- **FLUID**: `s = -dt * 4E * p_vol * (J-1) * inv_dx²`, plus a viscosity shear term
  `-dt * 4ν * p_vol * inv_dx² * (C + Cᵀ)`.
- **GRANULAR**: SVD of `F`, Drucker-Prager return mapping with σ clamped to [0.05, 5.0],
  Kirchhoff stress from the Hencky strain.
- **ELASTIC**: zero stress (not ported; no material uses it).
- **STATIC / NOT_SIMULATED**: skipped entirely.

Accumulation is **fixed-point int32 atomics** (`round(value * fp_scale)`), not float
atomics — portable, and bit-deterministic.

### 5.3 `grid_op.comp` — grid update

Per node: `vel = mom / mass`, then:
- gravity: `vel.y -= dt * g` (applied to every node, including massless ones)
- speed clamp at `8 * dx/dt` — clips genuine instability so it reads as inert rather than
  as material vanishing
- **FACE PUSH**: if the node has mass and is inside the `bound` band at a face, and its
  outward velocity exceeds `PUSH_SPEED` (0.5), set that face's bit in the face mask.
  Tested BEFORE the band clamp, which zeroes exactly that outward component.
- **boundary band**: `bound` (1) cells inside each face have their outward velocity zeroed.

### 5.4 `g2p.comp` — grid → particle

Gather velocity and the APIC affine matrix `C` from the 3×3×3 neighbourhood, then:
- `pos += dt * vel`, clamped to `[0, dims)` — the region is a hard boundary
- FLUID: `J *= 1 + dt * trace(C)`
- GRANULAR/ELASTIC: `F ← (I + dt·C) F`

### 5.5 `readout.comp` — grid → voxels

Per node: argmax over `g_mass_mat` for the dominant material, mass → fill (1..255), write
through the shared changelog (the same safe multi-writer path the CA used).

Guards and side outputs:
- never writes over a STATIC-material voxel (rock, glass)
- `FILL_DEADBAND = 2` suppresses sub-threshold fill churn so a settled pool can actually
  reach the idle threshold
- sets the **occupancy** bit for the chunk containing the node, when node mass ≥ 1e-6
- sets the **face-push** bits (see 5.3, which owns them)

Also suspended entirely if `n_particles >= max_particles` — writing air over voxels the
simulation cannot represent would silently destroy material.

---

## 6. The hard constraint: chunks ≠ region granularity

The CA solver this replaces had **per-chunk** activation: a voxel moving into a new chunk
added that chunk; a chunk coming to rest was removed. N active chunks cost N × 32³.

**MPM cannot do that.** P2G scatters and G2P gathers through one shared accumulator grid, so
a chunk cannot be "off" while its neighbour is "on" — a particle crossing the boundary must
find the destination nodes already allocated and contiguous. There is no way to cut a hole
in the grid that particles can still reach.

Therefore:

| | CA | MPM |
|---|---|---|
| activation unit | chunk | region (contiguous volume) |
| cost of N chunks | N × 32³ | **bbox³** |
| deactivation | per chunk, on rest | whole region only |
| boundaries | between chunks, cheap | one hard boundary around the region |

The cost difference is the whole problem: 8 chunks in a 2×2×2 arrangement is 64³ = 262,144
nodes, not 8 × 32,768. A scattered or L-shaped body pays for the box, not the material.

---

## 7. History — every allocation strategy tried

Recorded because several of these look reasonable in isolation and were not.

### 7.1 Per-chunk independent windows (the original evaluation)

Each active chunk got its own 32³ grid. **Failed** — four walls at every chunk boundary:
1. `p2g` dropped out-of-grid nodes → kernel mass lost, phantom free surface
2. `g2p` dropped them too → velocity/affine under-weighted
3. `g2p` **clamped position to the chunk** → particles could not cross at all
4. `grid_op` damped a 3-cell band at every edge → a soft wall

### 7.2 Shared region + growth on "chunk changed"

One grid spanning the bbox of active chunks; grow a face when a face chunk produced any
readout change. **Failed** — a chunk is 32 voxels deep, so a body sitting anywhere inside an
outer chunk counted as "touching the wall". Measured 2×2 → 3×3 → 4×4 → 5×5 over one run.

### 7.3 Growth on mass at the exact boundary plane

**Never fired.** `grid_op`'s damping band holds material `bound` cells short of the plane, and
a particle at local y=3.0 scatters to nodes 2,3,4 — node 0 receives no mass at all. The test
was measuring a plane material physically cannot occupy. Region stayed 1 chunk tall in Y.

### 7.4 Growth on mass **in the band** (presence)

Y growth finally worked — and ran away. **A pool at rest in a box touches every wall of that
box by definition.** Every face lit, the box grew, the fluid spread to fill the larger box,
every face lit again. Measured: 128×64×128, all 32 chunk-volumes occupied, at the node cap.

### 7.5 Growth on outward velocity at a face

**Ran away again, for the same underlying reason.** Pressure on a wall is indistinguishable
from pressure into free space — that is what pressure *is*. A pressurised fluid genuinely
reaches every wall at genuine speed, so the test fires legitimately forever.

### 7.6 Fixed-size window, anchored on the flood-fill centroid

**Destroyed the body.** Occupancy is derived from INSIDE the window, so once the window is
slightly off-centre the centroid describes only the sliver still inside; the window chases
that sliver and abandons the bulk. Measured `14146 particle(s) left the window` on a
14,147-particle body — total loss, repeatedly.

### 7.7 Fixed-size window, edge-follow (translate one chunk)

**Worked** — the window follows the material without stranding it. But material still hit the
wall with only 4 chunks active, because the window was placed in its own min corner by
bbox-padding.

### 7.8 Fixed window, occupancy-based chunk release

The release rule was "no readout change for 30 ticks", which is true of a chunk holding
**resting** water — so the window shrank over settled pools and killed the particles it left
outside, freezing water in mid-air. Replaced with occupancy-based release, then removed
entirely.

### 7.9 Current: smallest region that fits the edit, bounded growth

`region_chunks` is a **floor** (default 1×1×1), the first build sizes to the edit's own chunk
span, and growth from `w.occupied` expands up to `max_axis_chunks` / `MAX_REGION_NODES`.

**All of §7.2–7.8's machinery has been deleted from the tree.** If a strategy is re-tried,
start from §9 rather than from memory.

---

## 8. Measured cost

Same hardware (RTX 5070), 24 substeps, water, single body:

| window | nodes | particles | ms/tick |
|---|---|---|---|
| 32³ | 32,768 | 5,575 | ~0.3 (implied) |
| 64³ | 262,144 | 5,575 | **2.4** |
| 64³ | 262,144 | 44,600 (subdiv 2) | **6.4** |
| 96³ | 884,736 | 5,575 | **~5** |
| 128³ | 2,097,152 | 5,575 | **8–17** |

Two conclusions that have held up:

1. **Grid cost is linear in nodes** and dominates at low particle counts — a single chunk of
   water is not what costs 2.4 ms, the box is.
2. **Particle cost becomes significant at high subdiv** — 8× particles took 2.4 → 6.4 ms, so
   the particle half is roughly 3.5 ms at 44,600. That caps how far particle density can be
   pushed at 60 Hz.

Window moves also reallocate every buffer: measured hitch of **34–91 ms** at 34–120 MB,
which is why moves are rate-limited to one per 10 ticks and why `region_chunks` ≥ 3 is
expensive in stutter as well as average.

---

## 9. Theories, with evidence

### 9.1 Free-surface pressure is under-resolved at 1 particle per cell — LEADING

**Mechanism.** P2G scatters pressure weighted by mass. With one particle per cell, a particle
at a free surface has only part of its 3×3×3 gathering stencil populated; the empty half
contributes gravity (applied to all nodes) but no pressure. G2P averages over the whole
stencil, so the surface particle's velocity is diluted toward "no pressure". The body has
nothing holding it together laterally and films out instead of holding depth.

**For:** the reference implementation this port follows uses 2×2×2 = 8 particles per cell
precisely to avoid this. Behaviour is insensitive to stiffness across 256× (see 9.3), which
a missing-pressure-source explanation fits better than a wrong-constant one. It explains
films, creeping, and the pancake.

**Against:** not yet cleanly tested. The first `particle_subdiv = 2` run was invalid (9.2);
the second ran at 6.4 ms with the body still occupying 8/8 chunks — but chunk occupancy is a
weak signal (one particle's 27-cell kernel marks a whole chunk), so this is not yet
conclusive either way.

### 9.2 The `particle_subdiv` freeze — a bug, now fixed

Dividing mass by `sub³` without dividing `p_vol` left each particle claiming a full voxel's
pressure response for 1/8 the mass. Effective stiffness became `E × 8` = 64,000, past the
stability point, and the body jammed against the speed clamp. `p_vol` is now derived in one
place and everything scales together. **Recorded because the first result looked like a
finding about MPM and was purely an arithmetic error.**

### 9.3 Stiffness has an upper stability bound, not a tuning range

Sweep of E = 500 → 128,000: character changes (500 is "soft"), but from baseline upward the
endpoint is identical — body fills the window, every face lit. One narrow band looked
plausible. This is consistent with a stability ceiling rather than a quality knob: too coarse
a substep or too stiff a material and J diverges, and the body expands to fill whatever
volume it is given.

### 9.4 No surface tension

MPM water has no cohesion term, so it does not bead — small quantities film out. Real, and
cheaply observable, but its absence would not explain insensitivity to stiffness, so it is
probably a contributor rather than the root cause.

### 9.5 Fixed-point headroom at high particle density — UNCHECKED

`fp_scale` is `NODE_BUDGET / p_vol`, so `g_mass` works out to `density × weight × NODE_BUDGET`
per particle regardless of `p_vol`. With 8 particles per cell a node accumulates ~8× more
contributions than the constant was chosen for. It remains far below int32 max (~2e9) in the
arithmetic, but this has not been verified empirically against a dense body — **check it
before trusting any `subdiv ≥ 2` result.**

---

## 10. Knobs

All are `@export`, so the **remote Inspector** (Debugger → Remote Scene Tree →
`WorldOrchestrator/SimulationRuntime/MPMSimulation`) edits them on a running game.

| export | default | meaning |
|---|---|---|
| `region_chunks` | (1,1,1) | MINIMUM window, in chunks — not the size |
| `grow_window` | true | grow on occupancy contact vs. translate |
| `max_axis_chunks` | 2 | per-axis ceiling (2 = up to 64³, ~2.4 ms) |
| `max_particles` | 65,536 | region-wide particle capacity |
| `particle_subdiv` | 1 | particles per voxel per axis (2 = 8× particles) |
| `promote_stride` | 1 | sample one particle per N³ voxels |
| `water_density` | 1.0 | → region rebuild |
| `water_stiffness` | 8000 | instant; stability ceiling applies |
| `water_viscosity` | 0.0 | instant |
| `sim_seconds_per_tick` | 1/60 | substep dt is derived |
| `bound` | 1 | boundary damping band width, cells |
| `rest_speed` | 0.02 | sleep threshold |
| `idle_threshold` | 4 | quiet ticks before sleeping |
| `max_particles`, `mpm_substeps_per_tick` | 65,536 / 24 | capacity / substeps |

Constants not exposed: `MAX_REGION_NODES` (1,048,576), `NODE_BUDGET` (1e7), `NMAT` (8),
`PUSH_SPEED` (0.5, in `grid_op.comp`), `MASS_THRESHOLD` (1e-6), `FILL_DEADBAND` (2),
`REFERENCE_MASS` (4.0).

---

## 11. Debug tooling

| key | action |
|---|---|
| `B` | spawn a radius-11 WATER body above the player (prints voxel + chunk) |
| `Y` | region dump: origin/dims/nodes, requested vs occupied vs covered, particle count, `idle_ticks`, face mask, **live water physics**, timescale, last tick ms |
| `G` | 60-tick perf averaging |
| `Z` | drop region + active set, keep voxels (clean test state) |
| `T` | cycle `sim_seconds_per_tick` (1/60, 1/120, 1/240, 1/480, 1/30) |
| `O` | cycle water physics presets (drives the same exports) |
| `R` | full world regen (also resets the solver via `slab_invalidated`) |
| `C` | active-chunk tint (shows grid coverage) |
| `P` / `→` | pause / single-step |

Log prefixes: `[MPM]`, `[MPM perf]`, `[MPMStress]`, `[MPM params]`, `[SimRuntime]`.

---

## 12. File map

| file | role |
|---|---|
| `simulations/mpm_simulation.gd` | the whole driver: region lifecycle, buffers, uniforms, tick |
| `shaders/mpm/clear_grid.comp` | zero grid accumulators |
| `shaders/mpm/p2g.comp` | particle → grid scatter, all material stress models |
| `shaders/mpm/grid_op.comp` | grid velocity, gravity, face push, boundary band |
| `shaders/mpm/g2p.comp` | grid → particle gather, position/state advance |
| `shaders/mpm/readout.comp` | grid → voxels, occupancy, fill deadband |
| `simulations/simulation_runtime.gd` | tick scheduling, solver-mode gate, perf timing |
| `material_db.gd` | the 32 B/material MPM table (`_encode_mpm_blob`) |
| `material_def.gd` | per-material physics fields |
| `world.gd` | test hotkeys, region dump |

---

## 13. Open decisions

1. **Is the fluid's behaviour a solver defect or a fidelity ceiling?** §9.1 is the test.
   Nothing about allocation or windowing should be redesigned until it is answered, because
   every allocation change so far was made to work around symptoms that may share one cause.
2. **If bodies must exceed ~64³**, the only scalable answer is resolution LOD: coarsen grid
   AND particle size together (2× coarser = 8× fewer nodes for the same volume). Cross-
   resolution seams are the hard part, and they are the same class of seam that produced
   §7.1.
3. **Per-connected-component regions** would stop two separate puddles sharing one box. It
   does not help a single large body, and it does not change the cost curve.

---

## 14. Research direction — sparse scheduled MPM

Research surfaced four families that attack the spatial-activation problem directly
(citations below are pointers to follow up, not yet verified against the papers):

1. **Sparse background grids / active-node MPM** — the grid exists only where particles need
   it, not as a dense bounding box. (arXiv:2605.28525, "Unified sparse framework for
   large-scale material point method simulations".)
2. **Dynamic Background Domain** — detect active particles, restrict the update to the
   affected domain. (Wiley, DOI 10.1155/2021/8854318.)
3. **Regional time stepping** — schedule small blocks independently, respecting the
   particle↔grid coupling. (Wiley, DOI 10.1111/cgf.13524, uses SPGrid.)
4. **Adaptive grid extension/reduction** — open-source "Matter" solver (github
   larsblatny/matter) explicitly lists this, with APIC/Drucker-Prager/μ(I), i.e. the same
   mechanics family this port uses.

### What this resolves and what it does not

The correct framing, which supersedes "how do I add/remove chunks from one MPM sim":

> **Chunks are storage and scheduling units, sitting on top of one sparse shared grid — not
> the simulation units.** This keeps a single P2G/G2P continuum while restoring CA-like
> "this block is asleep" behaviour.

Concretely for our code: `clear_grid`/`p2g`/`grid_op`/`g2p`/`readout` all iterate the flat
`dims.x*dims.y*dims.z` node space. The sparse version replaces that flat index with a global
**active-node map** (one index space with holes). The dispatch count becomes "active nodes",
not "bbox product". This is NOT the per-block grids of §7.1, which failed because each block
had its own index space and produced walls; a sparse grid is one shared index space.

### Two hard constraints the research also implies

- **Spatial sparsity and temporal sleeping are separate.** A sparse grid fixes "I am paying
  for empty bbox"; it does not fix "I am updating resting material". Both are needed, and they
  must not be conflated.
- **You cannot sleep half a pressure volume.** Hydrostatic equilibrium is global — the bottom
  of a column feels the height above it. If the top of a water column is baked to voxels while
  the bottom stays active, the active region must still be told "there are N voxels of water
  above me". This is the same connected-vessel / pressure-projection problem the CA solver
  spent weeks on (`ARCHITECTURE.md`, `docs/BUG_LOG.md`). The voxel field is already the
  boundary representation (readout bakes particles to voxels on settle); the missing piece is
  re-reading column height as a pressure boundary condition at wake.

### Ordering — the one correction to the research

Sparse grid and dispersion (§9.1) are **independent**: sparse grid fixes cost, dispersion
fixes correctness. They are not alternatives. But they should not be built in parallel:

1. **S0 — settle dispersion first.** A sparse grid's payoff is proportional to how compact
   the material is; a body that films out into a sheet has less sparsity to exploit, and more
   importantly, no visual judgement of the solver is valid while water doesn't pool. The
   `particle_subdiv = 2` test is one clean run from conclusive (§9.2's arithmetic error is
   fixed; §9.5's fixed-point headroom still unchecked).
2. **S1 — sparse shared grid.** Replace the dense bbox with an active-node map. This is the
   answer to the §8 cost curve.
3. **S2 — block sleep/wake + wake propagation**, with a 1–2 block halo for the 3³ B-spline
   support, and the hydrostatic boundary read from voxels.
4. **S3 — resolution LOD** for bodies that must exceed one grid resolution.

### Rejected (again, now with the research agreeing)

- Independent per-chunk MPM grids — §7.1, falsified; the research's "don't make chunks the
  simulation units" is the same conclusion.
- Bounding-box growth and centroid tracking — §7.6/§7.8, falsified; both are "one dense
  allocation trying to decide its own extent from wall contact", which is the root disease
  the sparse grid removes.

---

## 15. S0 result (2026-08-13)

Working parameters found by hand-tuning: **density 1.0, stiffness 3000, viscosity 3.0,
subdiv 1**. Measured behaviour from the S0 instrumentation:

- A single sphere forms a blob first (depth 14, footprint 32×32, compactness 0.44) and then
  spreads into a puddle on the flat rock floor (depth ~3, footprint ~58×63, compactness 0.05).
  Mass is exactly conserved throughout (928.50 = 7428 × 0.125; 1768.38 = 14147 × 0.125).
- Two spheres slosh: max_speed oscillates 2–238 voxel/s, the body fills the 64³ region, and it
  never reaches rest (idle_ticks stays 0). A "random ping" is one particle reaching 238 voxel/s.

Two conclusions:

1. **Spreading thin on a flat floor is CORRECT water**, not the pancake. The earlier "dispersion"
   (§9.1) was most likely the momentum overflow (§9.5) masquerading as a free-surface defect —
   once the accumulator was fixed and viscosity added, the body forms a coherent blob and levels
   out as water should. The compactness collapse (0.44 → 0.05) is a puddle forming, not a bug.
2. **The "random ping" is a residual velocity spike at 238 voxel/s**, which is ABOVE the 200
   clamp added in e408c8d. A weighted gather of clamped node velocities cannot exceed 200, so
   this is evidence the clamp is not in the running build — grid_op.comp is loaded once at setup,
   and material-parameter sliders do not reload shaders. One full restart loads it.

Remaining known gaps, none blocking: under-damped sloshing (viscosity 3 is low, no surface
tension), and the momentum-clamp fix must actually be loaded before the "ping" is judged.

