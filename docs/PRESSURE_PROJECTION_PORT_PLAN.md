# Implementation Plan — Connected-Vessel Hydrostatics via Pressure Projection

**Status:** design validated in a 2D Python bench; ready to port to GLSL/GDScript.
**Read first:** `docs/BUG_LOG.md` → *"DESIGN (2026-07-04): connected-vessel hydrostatics — SOLVED IN A 2D BENCH via pressure projection"*. That entry is the source of truth for **why** this design (and why four cheaper ones were falsified). This file is the **how-to-build-it**.

This is a handoff plan for the primary agent/session that owns the live codebase. It is intentionally staged with hard validation gates so each milestone is provably correct before the next.

---

## 0. One-paragraph summary

The head()-driven flux model cannot solve U-tubes: local straight-up `head()` is blind to pressure queued behind a horizontal connection, and **any max/envelope-propagated head field (Build B) deletes the lateral gradient that must drive equalization** (proven: uniform bottom-row pressure at 2:1 arm imbalance). The fix is **Chorin pressure projection**: each tick, build a *provisional* flux from gravity, compute its divergence (plus a masked density-correction source), solve `∇²P = divergence` iteratively, correct the flux with `−∇P`, and feed the **existing conservative `s_out`/`s_in` transport** unchanged. Pressure is recomputed every tick, never stored as momentum; transport remains the only writer of fill.

---

## 1. Non-negotiables (these are load-bearing — the bench fails if any is dropped)

1. **RHS must carry the mass imbalance.** Provisional gravity flux is scaled by the water's actual weight (`fill`), and a **density-correction source** `STIFF·(FILL_CAP − fill)` is added on *submerged* cells. A topology-only RHS produces a uniform-at-elevation field (the max-relaxation failure) and freezes arms lopsided.
2. **The density-correction term is masked off at the free surface** (any liquid cell with air directly above). Unmasked, it force-fills legitimately-partial surface cells and destroys the rest state (breaks the pressure-null invariant).
3. **Sealed-boundary rule:** solid, off-grid, **and any withheld/unavailable chunk page** are Neumann boundaries — the face is *dropped* from the Poisson stencil. **Never** treat an unavailable neighbor as air or `P=0`. This is exactly the dispatch-range fix already in `fluid_flux.comp` (`page < 0 → return 0.0`); the same discipline applies to both the pressure solve and the transport.
4. **Free surface is Dirichlet `P = 0`** (liquid→air face). Air is excluded from the pressure domain.
5. **Transport stays the only place fill changes**, through the existing `s_out`/`s_in` budget — conservation is structural and must remain bit-exact.
6. **Do not weaken the pressure model to save cost.** All optimization targets the solver's *domain, resolution, iteration schedule, and initialization* — never the RHS, the mask, or the projection itself.

---

## 2. The algorithm (per tick)

Coordinates/conventions match `fluid_flux.comp`: owned faces are +X, +Y, +Z; `+Y` is up; `FILL_CAP = 255`.

```
1. BUILD RHS  (one pass over pressure-active liquid cells → _div_buf)
     provisional vertical flux on each owned +Y face (p → p+Y):
         u*_y(p) = -G * (fill(p+Y)/255)   if p+Y is liquid and neither side solid, else 0
         (u*_x = u*_z = 0 — gravity is vertical only)
     divergence for liquid cell p:
         div(p) = (u*_y(p) - u*_y(p-Y))            // + x,z terms, both 0
         if (p+Y is liquid):                        // submerged (NOT a free surface)
             div(p) += STIFF * (FILL_CAP - fill(p)) // density correction, masked at surface

2. SOLVE  ∇²P = div   (RED-BLACK Gauss-Seidel, N_ITER sweeps; each sweep = 2 dispatches)
     for each liquid cell p (its color this pass):
         s = 0 ; k = 0
         for each of the 6 axial neighbors nb:
             if nb is liquid:  s += P(nb) ; k += 1
             elif nb is air:   k += 1           // free surface, Dirichlet P=0 (adds 0 to s)
             else:             (skip)           // solid / off-grid / withheld page → sealed
         if k > 0:
             target = (s - div(p)) / k
             P(p) += OMEGA * (target - P(p))    // OMEGA=1 is plain GS; 1.5–1.8 = SOR

3. CORRECT  F = u* - ∇P   (one pass → the flux the transport will apply)
     F_y(p) = u*_y(p) - (P(p+Y) - P(p))         // P=0 for air/sealed neighbors
     F_x(p) =         - (P(p+X) - P(p))
     F_z(p) =         - (P(p+Z) - P(p))
     (only on faces where at least one side is liquid and neither side is solid)

4. TRANSPORT  (existing conservative pass, UNCHANGED)
     read F through eff(); compute s_out/s_in; apply symmetric budgeted deltas; write changelog.
```

**Persistence decision (flag for the owner):** the bench validated the *quasi-static* form — `F` is recomputed from scratch each tick, **not** persisted. This removes wave/slosh inertia (water relaxes to equilibrium, more "molasses" than "splash"). The current engine's water identity comes from `flux_buf` persistence. **Milestone 1 uses quasi-static** (simplest, matches the bench). Re-introducing inertia (persist a corrected flux and add it back into next tick's provisional field) is a later, separately-validated step — and per the earlier design discussion, the pressure correction must happen *before* the flux is persisted, never as a late transfer fighting an accumulated `flux_buf`.

---

## 3. Mapping onto the existing architecture

Relevant existing pieces (see `CLAUDE.md` Key Paths):
- `shaders/fluid_flux.comp` — Accelerate (pass 0) / Transport (pass 1), `compute_s_out`/`compute_s_in`, sparse `flux_page_table`, changelog writes, the `page<0 → return 0.0` sealed rule. Chunk size 32³ (dispatch `groups = (active*4, 4, 4)`, 8³ local, 4³ workgroups/chunk).
- `simulations/fluid_simulation.gd` — owns `_flux_buf` + `_flux_page_table` (sparse paging), `_build_flux_chunk_list` (active + face-neighbors), `_upload_flux_params` (`SimParams`, 52 bytes), the per-substep dispatch loop (Accelerate → Transport), page alloc/evict/zero, idle-guard sleeping.
- `WorldOrchestrator.gd` — buffer ownership, `_voxel_buf_a` (canonical), changelog buffer + `apply_changelog`.
- `material_sim` buffer — `SIM_LIQUID` classification.

### New/changed components

**Buffers (owned by `FluidSimulation`, riding the existing sparse page allocation):**
- `_pressure_buf` — one `float32` per voxel slot, paged exactly like `_flux_buf` (reuse `_flux_page_table`; the pressure domain ⊆ the flux-active region). Red-black GS updates in place, so **no ping-pong needed** (one buffer). If you fall back to Jacobi, you need two.
- `_div_buf` — one `float32` per voxel slot (the RHS), same paging. Built once per tick, read by every solve sweep.
- Keep `_flux_buf` as the corrected-flux carrier feeding Transport (or write corrected `F` straight into it before the Transport pass).

**Shader passes** (extend `fluid_flux.comp` with new `pass` values, or add a `fluid_pressure.comp`; extending keeps the paging/helpers in one place):
- `pass = build_rhs` — compute `u*` + `div(p)` → `_div_buf`. Reads `voxels_a`, `material_sim`, `flux_page_table`.
- `pass = solve_red` / `pass = solve_black` — one red-black GS sweep each. Reads `_div_buf`, `_pressure_buf`, `voxels_a` (for topology/BCs); writes `_pressure_buf`. Dispatched `N_ITER` times (2 dispatches per sweep).
- `pass = correct` — compute `F` → `_flux_buf`. Reads `_pressure_buf`, `_div_buf`(for `u*`) or recompute `u*`.
- `pass = transport` — the existing Transport pass, unchanged, reading `_flux_buf`.

**Red-black coloring:** color = `(gx + gy + gz) & 1` in *world* voxel coords (must be consistent across chunk boundaries — use global position, not local). Each sweep dispatches one color reading the other.

**`SimParams` additions:** `G`, `STIFF`, `OMEGA` (floats), plus a pass enum wide enough for the new passes. Grow the struct and update `_upload_flux_params` encoding + the buffer size (currently 52 bytes) in lockstep — a struct/buffer size mismatch silently corrupts every field after the change point (see the Build-B-era `SimParams` note in `BUG_LOG.md`).

**GDScript dispatch loop (`FluidSimulation.step()`):** replace `Accelerate → Transport` with `BuildRHS → (SolveRed → SolveBlack)×N_ITER → Correct → Transport`. Each is a submit/sync like today. **This raises the per-tick dispatch/sync count from 2 to `~4 + 2N`** — Milestone 2 (temporal amortization) is what brings that back down.

---

## 4. Milestones & validation gates

### M0 — Preserve the bench (do first, ~30 min)
Copy the validated 2D emulators into the repo as a permanent regression rig: `tools/fluid_prototypes/utube_projection.py` (the winner) plus `utube_emulator.py`, `utube_pressure.py` (the falsification ladder). They currently live only in ephemeral session scratchpad.
**Gate:** `python tools/fluid_prototypes/utube_projection.py` prints `ALL TESTS PASSED`.

### M1 — Full-resolution, active-masked projection in GLSL (the core port)
Implement §2/§3 with **quasi-static** flux (no persistence), **red-black GS**, and the active set = the existing flux-active chunk list (active + face-neighbors) for now. Fixed `N_ITER` (start ~40; re-measure — see Risks).
**Gates (must reproduce the bench invariants in-engine):**
1. **Pressure-null:** a settled full column/lake produces zero net fill movement per tick (read back `changelog.count` ≈ 0 / conservation delta 0).
2. **Bottom-pipe gradient:** in a U-tube test scene, before mass moves, `P` at the tall-arm base > `P` at the short-arm base (add a debug readback, reuse the `debug_dump_fluid_stats` pattern).
3. **U-tube equalizes** to level-matched **full** columns (not thin films), mass-exact (conservation check ≈ 0), and the final level is **independent of pipe length**. ⚠️ **Judge by per-arm water mass + a fill-profile readback, NEVER by surface height** — a lopsided thin smear reaches equal column *heights* at unequal *densities*, so a surface-height check gives a false pass (this fooled both the author and a reviewer; see `docs/BUG_LOG.md` "Metric pitfall").
4. **Thin-film:** an unsupported 1-voxel shelf drains and pools; it does not stay pinned.
5. **No regression** on flat-lake settling, pouring, sphere-drop (the existing baseline scenes).
Build the concrete acceptance scene described in `ARCHITECTURE.md` Step 10 (two columns + bottom passage, unequal heights), viewable through the GLASS material.

### M2 — Temporal amortization (make it affordable)
Run a small fixed iteration budget per tick and let long-range pressure crawl across ticks:
- **Warm-start** `_pressure_buf` from last tick (do **not** zero it), with local invalidation (reset `P` to 0 on cells touched by edits, topology/material changes, fill crossing the surface-mask threshold, and newly mapped pages — the events the changelog/edit path already surfaces).
- **Residual-driven continuation:** after a small batch of sweeps, compute a cheap per-page residual `max|∇²P − div|`; carry a "needs-more-solve" bit so only unconverged pages re-queue next tick. Priority proxy: `max(|div|, |fill−FILL_CAP|, surface-proximity)`.
- Split budgets if useful: a few local sweeps every tick (kills compression artifacts / thin-film pinning instantly) + a deeper pass every 4–8 ticks (carries U-tube info farther).
**Gate:** all M1 gates still pass; a long U-tube may take more ticks to equalize but must remain stable and reach the correct level; per-tick cost is bounded and dominated by the *active* region only.

### M3 — Optional optimizations (only if profiling demands, gated on evidence)
- **Coarse `2³` pressure grid for bulk bodies only.** ⚠️ A `4³` block erases a 1-voxel pipe — dangerous for the exact plumbing case this targets. Only viable if gameplay channels are reliably ≥2–3 voxels wide (**open design question — decide before building**). Keep transport and conservation full-resolution.
- **Lightweight multigrid V-cycle** for long pipes (principled long-pipe fix; real implementation risk on a sparse/irregular/free-surface domain — defer until temporal amortization is shown insufficient).
- **Graph "pipe mode"** (collapse degree-2 liquid chains to graph edges, solve a small graph Laplacian). Strong for long 1-wide plumbing; many edge cases; likely unnecessary if M2 suffices.

---

## 5. Parameter starting points (bench values — re-tune in 3D / red-black)

| Param | Bench value | Notes |
|---|---|---|
| `G` (provisional gravity) | 64.0 | per open vertical face, ×(fill/255) |
| `STIFF` (density correction) | 0.5 | too high oscillates, too low is slow to fill |
| `OMEGA` (SOR) | 1.7 (serial) | **red-black GS converges differently — re-measure** |
| `N_ITER` | ~40–60 (serial SOR) | **serial GS ≠ red-black/Jacobi; re-measure the iteration count for the parallel solver before sizing the budget** |
| `MIN_FLOW` (eff deadzone) | 0.5 | reuse existing `eff_flux` discipline |

---

## 6. Risks & open questions (surface these to the owner)

1. **Solver iteration count on GPU is unmeasured.** The bench used serial SOR (Gauss-Seidel with a sequential dependency) — **that does not exist on a GPU**. Use **red-black GS** (parallel, keeps most of SOR's rate) or **Jacobi** (ping-pong, ~2× iterations). The bench's `N_ITER`/`OMEGA` **will not transfer** — first task in M1 is to re-measure convergence for the parallel solver (a small red-black bench, or instrument in-engine).
2. **Per-tick dispatch/sync count** goes from 2 to `~4 + 2N`. Each dispatch is a submit/sync stall. M2 (temporal amortization) is the primary mitigation — it's not optional polish.
3. **Free-surface detection in 3D** must be robust (a liquid cell with air directly above → surface; masks density correction and sets the Dirichlet boundary). Get this wrong and the rest state breaks.
4. **Quasi-static removes waves.** Confirm whether the game wants inertial water (splashes/slosh) — if so, the persistent-flux variant is a follow-on that must be re-validated against the same invariants.
5. **Interaction with granular + edits.** Pressure-active masking must include cells near edits and sand-water impacts (the existing `mark_dirty`/`chunks_committed` events); the sealed-page rule must be consistent with granular's own paging.
6. **Dirty-bitmask overflow** (existing Known Issue) — sizing `_pressure_buf`/`_div_buf` and their masks from `_voxel_buf_capacity` avoids inheriting that bug; grow them alongside the voxel buffers.

---

## 7. First PR scope (concrete)

M0 + the M1 skeleton: add `_pressure_buf`/`_div_buf` with paging, the four new shader passes (build_rhs, solve_red, solve_black, correct), wire the new dispatch loop, add `G`/`STIFF`/`OMEGA` to `SimParams`, and stand up the U-tube acceptance scene with a `P`-field debug readback. Land it behind a flag (e.g. `FluidSimulation.use_projection`, default off) so the current baseline stays shippable while the new path is validated against the M1 gates. Update `ARCHITECTURE.md` (mark the to-do item in progress) and add any new bugs to `docs/BUG_LOG.md` as you go.
