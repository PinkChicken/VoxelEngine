# S1 — Sparse Shared Grid: Design

The plan for replacing the dense bounding-box grid with a globally-indexed sparse subset of
the MPM background grid. Written after S0 validated the dense solver (see
`docs/MPM_SOLVER_DESIGN.md` §15), which is the reference implementation this work must match.

Status: **Phase 1 DONE (commit 63b61a0), verified by hand-test.** Phase 2 (drop empty blocks)
is next.

---

## 1. The one absolute constraint

> **Chunks/blocks are NEVER independent MPM domains.**

§7.1 of `MPM_SOLVER_DESIGN.md` is the proof: per-block grids each had their own index space,
so a particle's 3×3×3 B-spline stencil could not cross a block boundary — P2G dropped
out-of-block nodes, G2P clamped to the block, and grid_op damped a wall at every edge. Four
artificial walls per chunk.

The correct model:

```
              ONE GLOBAL INDEX SPACE
                      │
       ┌──────────────┼──────────────┐
       │              │              │
    block A        block B        block C
    active          empty         active
```

not

```
   grid A        grid B
 ┌────────┐    ┌────────┐
 │ fluid  │  │ │ fluid  │   ← four walls at the seam
 └────────┘    └────────┘
```

A particle's kernel crosses block boundaries **naturally**, because the global node
coordinate is the same whether the node lives in block A or block B. The block only decides
*where the node is stored*, never *whether it exists*.

---

## 2. Conceptual model

> **The sparse grid is a globally indexed sparse subset of the MPM background grid; blocks are
> allocation/storage units.**

Do NOT phrase it as "a collection of active chunks." That phrasing invites the §7.1 mistake.
The physical coordinate system is the global grid; sparsity is purely an allocation concern.

This matters when S2 arrives. S2 will want:

```
block A: sleeping     block B: awake      block C: sleeping
```

which must mean "A and C's nodes are not allocated / not advanced" — NOT "A and C are
separate simulations." Sleep is a scheduling/allocation state, not a physical boundary.

---

## 3. Why block-paged, not hash-indexed

Two ways to make the grid sparse:

1. **Hash-indexed active nodes** (per-particle 27 hash lookups, open addressing, probing).
   Truly node-granular sparsity, but scattered memory access, collision handling, and the
   readout must iterate a hash table.
2. **Block-paged** (tile the grid into fixed blocks; allocate only blocks that contain ≥1
   active node; dense indexing within a block).

Choose **block-paged**, because:

- Our world is already 32³ chunks, and `chunk_table` already does `chunk_coord → slot` for
  voxels. The grid's block map is the same shape, a separate table.
- Dense indexing within a block keeps P2G/G2P's inner 27-node loop contiguous and fast.
- A particle's 1.5-cell kernel touches at most 2×2×2 = 8 blocks; the halo is a handful of
  blocks, not 27 scattered nodes.
- The "block" maps 1:1 onto what S2 wants to sleep/wake, and onto streaming chunks.

Block size = 32³ (one chunk). Coarser than node-granular sparsity, but a thin film still
allocates only the blocks it crosses — and the win is the *empty bbox between scattered
bodies*, which is the whole problem.

---

## 4. Addressing

Today a node is a flat dense index over the region bbox:

```
idx = (z * dim_y + y) * dim_x + x        // [0, dim_x*dim_y*dim_z)
```

Under S1, a node is addressed by a **global grid coordinate** decomposed into block + local:

```
block_coord = floor(gpos / 32)           // ivec3, world-voxel space (dx == 1)
local       = gpos - block_coord * 32
slot        = block_map[block_coord]     // ivec3 -> int, -1 if unallocated
idx         = slot * 32^3 + (lz*32 + ly)*32 + lx
```

`block_map` is the sparse table (a compacted block list with a hash from `block_coord` →
`slot`, or — during phase 1 — a dense map where every block in the region bbox is present).
It is **distinct from** the voxel `chunk_table`; it indexes the *grid*, not the voxels.

### The active-node set and halo

The active block set = every block touched by at least one particle's kernel
(`base-1 .. base+1`), computed once per tick by a **grid-construction pass**:

1. For each particle, mark the 8 candidate block coords its stencil touches.
2. Compact the marked blocks into a list, assign slots.
3. Upload `block_map`.

A 1-block halo is implicit: marking the 8 touched blocks per particle includes the neighbor
blocks a near-face particle scatters into. Safety: max particle displacement per tick is
`max_speed × dt × substeps = 200 × 6.94e-4 × 24 ≈ 3.3` voxels, far under 32, so the once-per-
tick construction cannot be outrun.

---

## 5. Per-pass changes

`clear_grid → p2g → grid_op → g2p → readout` all survive; what changes is *iteration and
addressing*. Phase 1 changes addressing only (all blocks allocated, dense-equivalent order);
phase 2 drops empty blocks.

| pass | today | S1 (phase 2) |
|---|---|---|
| **grid construction** | (none) | NEW — mark blocks touched by particles, compact, upload `block_map` |
| `clear_grid` | dispatch `dim_x*dim_y*dim_z` | dispatch over allocated blocks × 32³; zero each block's slot range |
| `p2g` | per particle, scatter to 27 dense-indexed nodes | per particle, resolve 27 nodes' `block_coord → slot` (≤8 blocks), scatter |
| `grid_op` | per node, dense index; reads voxel via `world_to_flat` | per node in allocated blocks; same voxel read (global coords unchanged) |
| `g2p` | per particle, gather 27 dense nodes | mirror of p2g addressing |
| `readout` | per node, dense index; writes voxel via changelog | per node in allocated blocks only; argmax → voxel unchanged |

The grid-construction pass is the new cost center and the paper's "active-node indexing
problem" made concrete. It is the only genuinely new machinery; the other four passes are
mechanical addressing rewrites.

---

## 6. Implementation ladder (two phases, boring first)

This mirrors the debugging ladder agreed for S1. **Each phase has a hard gate before the next.**

### Phase 1 — block-indexed addressing, ALL blocks allocated ✅ DONE

Refactor the five passes to address nodes via `(block_coord → slot) → local`, but keep every
block in the region bbox allocated, in lexicographic order.

**Gate 1: PASSED.** No new artifact on the regression config; exact mass conservation held;
the same blob→puddle trajectory and compactness decay reproduced. Notably it ran in a
NON-UNIFORM region (64×96×64 = 2×3×2 blocks), which exercises the non-power-of-2 block-count
divmod path (`nb_y = 3`). If the encode/decode had not been exact inverses, the water would
have been written to wrong voxels or mis-sliped against terrain — none appeared. This is a
pure refactor that isolates the addressing risk from the sparsity risk.

### Phase 2 — drop empty blocks ✅ DONE (correctness; memory deferred)

IMPLEMENTATION DEVIATED from the "grid-construction mark pass" plan: instead of a new per-tick
mark shader, it reuses the occupancy mask the readout ALREADY produces (`occupied ∪ 1-block
halo`, compacted on the CPU). Fewer moving parts, no new shader. The active set is built from
LAST tick's occupancy (dense on the first tick after a build), which is safe because a particle
moves ≤ ~3.3 voxels/tick while a block is 32.

**Gate 2 (correctness): PASSED.** Exact mass conservation on every run (1487→1487.00,
15634→15634.00, 28974→28974.00, 54466→54466.00) — a halo/indexing leak would have shown up as
mass loss, and none appeared. Water pools/spreads/settles identically.

**NOT done — the memory resize was deferred.** The grid buffers are still sized to the DENSE
bbox; only the dispatch bound (num_blocks) shrank. That captures the COMPUTE win but not the
MEMORY win. Shrinking the buffers to `num_blocks × 32³` is a separate follow-up.

### What S1 does and does NOT deliver (honest status)

S1 delivers: one region whose EMPTY blocks inside the bbox cost no compute. It does NOT deliver:

1. **Multiple regions** — placing water in a separate location still can't activate a second
   region; the request goes into `_active_chunks` but the single region ignores it.
2. **Growth toward far edits** — `_growth_for` only reacts to occupancy INSIDE the window, so
   water placed outside the window is never seen and never simulated.
3. **Sparse memory** — buffers remain dense-sized.

So "place water anywhere in a dynamic world" needs, beyond S1: the memory resize, region
expansion toward far edits (or a world-spanning region with raised bounds), and/or multiple
regions. That is the next architectural decision, not more of S1.

### The clean debugging ladder (user-agreed)

1. One 32³ active block — reproduce dense exactly.
2. Two adjacent blocks; a body crosses the boundary — no seam.
3. Three blocks, middle mostly empty — no boundary artifact.
4. Remove inactive nodes; measure node count and runtime.

---

## 7. Acceptance criterion (non-negotiable)

> **For the same particle state, sparse MPM produces the same physical result as dense MPM,
> while grid cost follows occupied/active nodes rather than the enclosing bounding box.**

"Sparse ≈ dense" is the whole point — it gives a continuously-comparable reference, so any
divergence is a bug in the addressing/halo, not a "sparse approximation" to hand-wave.

---

## 8. Risks and gotchas

- **The halo.** A particle whose stencil reaches an unallocated block is a silent mass/momentum
  loss (exactly §7.1's phantom free surface, in miniature). The 8-touched-blocks rule + the
  once-per-tick safety margin must be airtight before phase 2.
- **The readout's iteration domain.** readout writes voxels only where mass exceeds threshold;
  iterating only allocated nodes is safe, but the *occupancy/face masks* must still cover the
  region's chunk bbox, not just the material blocks, or growth/sleep sees a wrong extent.
- **Buffer sizing.** g_mom/g_mass/g_mass_mat/g_vel are sized `num_blocks × 32³` in phase 2.
  A block-allocation spike (one splash) reallocates; keep the growth path rate-limited as today.
- **Block map access pattern.** p2g/g2p's 27 lookups per particle are the hot path. Keep the
  map small and, during phase 1, branch-free (all blocks present) so the compiler can optimize
  before sparsity is introduced.
- **Do not reuse the voxel `chunk_table` for the grid map.** Different lifetimes (grid map
  rebuilds every tick; chunk_table is the persistent voxel world). Separate tables.

---

## 9. S2 foreshadowing (do not implement in S1)

Once the grid is block-paged and sparse, sleep/wake becomes a property of the block map:

- A block whose material is at rest → remove from the active set → its particles bake to
  voxels → the block's grid nodes are freed.
- Wake on disturbance: re-add the block, re-promote its voxels to particles.

The hard constraint from `MPM_SOLVER_DESIGN.md` §14 still applies: **you cannot sleep half a
pressure volume.** The hydrostatic boundary condition (column height above a sleeping block)
must be read back from the voxel field at wake. That is the connected-vessel problem the CA
solver already faced, and it is *why* the voxel field is the canonical store, not just a
fallback.

---

## 10. Out of scope for S1

- Resolution LOD / `promote_stride` coarsening (S3 — already proven in embryo).
- Sleep/wake and the hydrostatic boundary (S2).
- Multi-region interaction (two separate sparse regions touching is the §7.1 seam again; S1 is
  ONE region, made sparse).
- Particle-buffer sparsity (particles remain a flat `max_particles` array; LOD handles volume).

---

## 11. Definitions

| term | meaning |
|---|---|
| global node coordinate | integer (ix, iy, iz) in world-voxel space; dx == 1 |
| block | 32³ tile of the grid = one chunk; allocation/storage unit |
| block map | `block_coord → slot` table for the GRID (separate from voxel chunk_table) |
| active block | block whose nodes participate in a pass (touched by ≥1 particle kernel) |
| grid construction | the per-tick pass that builds the active block set |
| halo | the neighbor blocks a near-boundary particle's kernel reaches |
