# MPM Migration

Tracked plan for replacing the CA solvers (`FluidSimulation`, `GranularSimulation`) with the
MLS-MPM solver (`MPMSimulation`). Written 2026-08-13 so the reasoning survives a context switch.

Related: `tools/fluid_prototypes/MPM_EVALUATION_BRIEF.md` §35-37 (the evaluation that decided
this, including the material-by-material audit), `ARCHITECTURE.md` §5 (to-do), `docs/BUG_LOG.md`
(why the CA systems accumulated what they did).

---

## 1. Decision

**Keep MPM. Migrate to it. Delete the CA solvers once MPM owns the materials.**

This is not a rewrite of water into a third design — it is finishing the cutover that brief §37
already committed to.

### The evidence, in numbers

| | lines |
|---|---|
| CA system to be deleted: `fluid_flux.comp` 1521, `fluid_simulation.gd` 1588, `granular_simulation.gd` 580, `momentum_resolve.comp` 533, `granular_sim.comp` 337, `demand_resolve.comp` 276 | **4,835** |
| MPM replacement: 5 shaders (635) + `mpm_simulation.gd` (555) | **1,190** |

### Why — the bug class, not the line count

Reviewed the known-issues list and `BUG_LOG.md`: the recurring failures are **not** physics
errors. They are bookkeeping errors around moving integer fill amounts through a sparse buffer
under CAS + changelog + partial-failure conditions:

- changelog overflow applying stale entries → voxel corruption
- `try_compress` vs vacating-destination race → fluid duplication/loss
- flux pass 0/1 dispatch-range mismatch → volume loss at coverage boundaries
- punch demand-queue overflow → ~20% mass erased (the "real leak")
- `momentum_resolve` `touched_cells` capacity → accumulation silently unreachable (same class, one pass later)
- granular/fluid `sim_active` erasure, cross-activation, cross-chunk diagonal CAS

MPM removes this class structurally: mass is the sum of particle masses, so it is conserved by
construction, and there is no changelog, no CAS conflict resolution, and no demand queue.

### The specific complexity being deleted

`fluid_flux.comp` grew from **609 → 1,521 lines**. Of the additions:

- **+224** phi-flood pressure field
- **+471** the liquid-interaction pipeline (punch, buoyancy, fill-history)

Plus `momentum_resolve.comp` (533) and `demand_resolve.comp` (276) — roughly **1,400 lines whose
entire job is density sorting and momentum displacement.**

Brief §35.5: under MPM *"the 'which materials sort against each other' question disappears with
it, because sorting is emergent"*. The single most expensive, most-failed feature in this project
becomes free.

### Sand migrating is the payoff, not a cost

`granular_simulation.gd` + `granular_sim.comp` is 917 lines with its own bug list, and the
granular↔fluid **seam** is itself documented as a bug source (`sim_active` erasure, cross-
activation, and a hardcoded dispatch-order hack: *"sand sinks through water because granular
mutations win on conflict"*). MPM sand already exists (Drucker-Prager). Migrating it removes the
seam and makes sand-sinks-through-water a consequence of mass rather than dispatch order.

### What does not migrate

- **FIRE** — no conserved mass; stays out of MPM (brief §35.4).
- **SMOKE** — viable per §36 but a cost-scaling question; stays on a separate simple path.

Both are unbuilt today (Phase 8/9 are empty design slots), so nothing is lost.

---

## 2. Current state

`mpm-rewrite` @ `067bca1` — MPM exists as a production-shaped fourth writer:

- Child of `SimulationRuntime`, chunk-aligned (never player-centred), 24 substeps/tick.
- Physics is **fully table-driven**: `MaterialDef.mpm_model/mpm_density/mpm_stiffness/
  mpm_viscosity/mpm_friction_deg` → 32-byte rows via `MaterialDB.build_mpm_buffer()`.
  This clears brief §35.2's blocker (hardcoded `is_sand = (mat == 2)`, one global
  `E`/`mu`/`lambda`/`alpha`) and closes §35.1a (viscosity) and §35.1b (static solids —
  `grid_op.comp` takes no-slip from real STATIC voxels, so terrain needs no particles).
- Passes: `clear_grid`, `p2g`, `grid_op`, `g2p`, `readout`; `readout.comp` writes simulated
  mass back into the canonical `_voxel_buf_a`, guarded against overwriting STATIC voxels.

**Not done:** water parity vs CA; the CA deletion.

### 2.1 Cross-chunk continuity — shared region grid (2026-08-13)

The per-chunk-window model is gone. It had **four** walls, not the one the old header admitted,
and all four lived in the shaders:

| # | wall | where |
|---|---|---|
| 1 | out-of-grid P2G nodes dropped → kernel mass lost, phantom free surface | `p2g.comp` bounds check |
| 2 | out-of-grid G2P nodes dropped → velocity/affine under-weighted | `g2p.comp` bounds check |
| 3 | **position clamped to `[0, n_grid)`** → particles physically could not cross | `g2p.comp` clamp |
| 4 | 3-cell velocity damping band at every edge → a soft wall | `grid_op.comp` (`P.bound`) |

**Replacement:** one `ChunkRegion` covers the **bounding box of all active chunks**, and the grid
is rectangular (`dims: Vector3i`) instead of a cube. Chunk boundaries inside the region are now
ordinary interior nodes — no special handling anywhere. The four walls above survive only at the
true edge of the simulated domain, where they are correct.

Consequences worth knowing:

- **Particle capacity is now region-wide**, not per-chunk (`max_particles` = 4000 total, not
  4000 × chunks). This is the next thing to raise.
- **Idle is a region property.** A chunk going quiet next to a flowing neighbour is not a reason
  to drop anything — that would tear a shared grid mid-simulation.
- **Rebuilds re-seed from voxels.** Region geometry only changes when the active chunk set
  changes (a rare, structural event), and a rebuild re-seeds particles from the voxel world, so
  in-flight velocities are lost across one. Acceptable, documented, not per-tick.
- **Far-apart active chunks waste grid** between them, because the region is one bounding box.
  `MAX_REGION_NODES` (1M ≈ 60 MB) bounds this by refusing to grow and warning, rather than
  allocating gigabytes. The refinement — one region per *connected component* — is deliberately
  deferred; it adds merge/split bookkeeping without changing anything a contiguous lake sees.

---

## 3. One solver at a time (enforced)

`SimulationRuntime.SolverMode` — `CA` or `MPM`, exactly one owns the voxel buffer per run.

- **CA** (default): `GranularSimulation` + `FluidSimulation` step; MPM is never bound
  (no buffers, no tick).
- **MPM**: only `MPMSimulation` steps; `_on_chunks_committed` returns early so the CA sims are
  never even woken.

Override without editing scenes: `-- mpm` or `-- ca` on the command line.

**Why MPM is now the default (2026-08-13):** the CA solvers are already `enabled = false` in
`world.tscn` — the scene had been switched to MPM by hand before the gate existed, so a CA
default made the world sit completely still (nothing to step, and MPM gated off). MPM is what
actually runs. CA remains selectable with `-- ca`, **but both CA nodes must also be re-enabled in
`world.tscn`** before CA produces motion, because their own `enabled` export is authoritative.

---

## 4. Verification status

The gate change and the pilot were **not** verified in-engine: Godot's project-load path does not
run inside the agent's sandbox (even a minimal `project.godot` hangs headless with no output).
Static review only. Confirm with:

```
a:/Godot_v4.6.3-stable_mono_win64/Godot_v4.6.3-stable_mono_win64.exe --path . res://world.tscn
```

Expected on boot: `[SimRuntime] solver mode = CA ...` then `[MPM] not started ...`. For MPM:
append `-- mpm` after the scene path.

---

## 5. Cutover order

**S1 — Done (2026-08-13).** Gate the solvers; stop paying for both engines.
Acceptance: boot log shows exactly one solver mode; only that solver's passes appear in the
frame.

**S2 — MPM water parity.** Drive `MPMSimulation` to match CA water on a fixed scene (a U-tube and
a waterfall, the two shapes that broke the CA mechanisms).
Acceptance: visually indistinguishable at the same voxel scale; no settling deadlock; no volume
drift.

**S3 — Implemented 2026-08-13, NOT yet verified in-engine.** The shared region grid replaces the
four per-chunk walls (§2.1).
Acceptance (still to be observed, not assumed): a lake spanning ≥3 chunks behaves as one body; a
waterfall crossing chunks does not shed or pile particles at a boundary; a body of water poured
across a chunk seam settles flat with no ridge at the seam. **This is the real gate — without it
MPM cannot replace CA.**

**S4 — Partially done (2026-08-13):** the default is now MPM (the CA nodes were already disabled
in the scene). Still open: raising `max_particles` above its region-wide 4000, and validating
sand's repose angle alongside water.

**S5 — Delete the CA solvers** (§6 manifest). Re-validate collision, rendering, and edits after.

**S6 — Then** resume the optimization pass (`ARCHITECTURE.md` §7), re-scoped: the phi-flood and
the ~60 submit/sync round-trips per fluid substep disappear with the CA code, so §7.1 must be
re-derived rather than ported.

---

## 6. Keep / delete manifest

### Delete at S5

| file | lines |
|---|---|
| `shaders/fluid_flux.comp` | 1521 |
| `shaders/momentum_resolve.comp` | 533 |
| `shaders/demand_resolve.comp` | 276 |
| `shaders/granular_sim.comp` | 337 |
| `simulations/fluid_simulation.gd` | 1588 |
| `simulations/granular_simulation.gd` | 580 |

Also at S5: `world.tscn`'s `FluidSimulation`/`GranularSimulation` nodes, the `SolverMode` gate
itself (it becomes a constant), and the CA-only entries in `MaterialDef` (`flow`, the `density`
sentinel) once nothing reads them.

### Keep deliberately

- `docs/BUG_LOG.md`, `ARCHITECTURE.md` — the institutional memory. This is the asset the rewrite
  is *for*; do not prune it during a "scratch".
- The debug tooling: `KEY_X` hide, `KEY_C` active-chunk tint, `KEY_T`/`KEY_N`/`KEY_J` tint toggles,
  `FluidWatch`, the debug tint passes, `RaymarchRenderer.debug_*`. Per `ARCHITECTURE.md` §5, this
  tooling is what resolved the 2026-07-10 "bench-validated fix shows no in-engine change" mystery
  and was the missing ingredient in four failed attempts. Rebind it to MPM — do not delete it.
- `tools/fluid_prototypes/python/` benches — reference only, local-only by `.gitignore`.
- `edit_apply.comp`, `surface_extract.comp`, `patch_apply.comp`, the raymarcher, `ChunkManager`,
  `WorldOrchestrator`: all solver-agnostic, unchanged.

---

## 7. Open risks

1. **Region footprint (replaces the old cross-chunk seam risk).** One bounding box over all active
   chunks means far-apart material buys a large mostly-empty grid. `MAX_REGION_NODES` refuses
   rather than over-allocates, so the failure mode is "a distant chunk silently doesn't simulate",
   which is a correctness trade, not a crash. Per-connected-component regions are the fix if it
   ever bites.
2. **Particle budget is now shared.** 4000 total across the whole region. A lake spanning 3
   chunks has a third of what it had under per-chunk budgets. Raise this before judging water
   quality — and re-check the fixed-point `fp_scale` derivation (§36.3) when you do.
3. **Rebuilds discard in-flight velocity** by re-seeding from voxels. Only on chunk-set changes,
   but a waterfall spawning new active chunks mid-fall could visibly hiccup.
4. **Static terrain is read per substep.** `grid_op.comp` reads `_voxel_buf_a` for no-slip, adding
   a read dependency between MPM and the canonical buffer that CA didn't have. Measure it.
5. **None of the 2026-08-13 work is verified in-engine.** The agent's sandbox cannot load a Godot
   project at all (§4), so the shader/driver signature change and the region logic are static
   review only. Treat the first run as the real test.
6. **Parity is a human judgement**, not a test. Use the U-tube and waterfall scenes, and the same
   in-editor-first discipline §5 of `ARCHITECTURE.md` prescribes: confirm a change is live with an
   unmistakable visual marker before trusting a bench result.
