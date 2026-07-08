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
7. **Sealed-column vacuum lock (KNOWN LIMITATION, scoped out of v1 — kept intentionally as a future aquifer/pump feature seed, not something to fix).** A liquid column sealed to a solid ceiling with less than a 2-row air gap does not drain (physically correct incompressible vacuum lock; no free-surface anchor). Correlates with the seal, not imbalance magnitude. Naive "P=0 at solid ceiling" fix is wrong (it also fires on confined pipe cells and breaks the normal case). Principled fix = `P ≥ 0` non-negativity/LCP constraint; defer unless gameplay shows fully-sealed full bodies are common. Pinned by `test_sealed_column_vacuum_lock_KNOWN_LIMITATION()` in the bench; full write-up in `docs/BUG_LOG.md`.

**Checkpoint 1 — CLOSED (2026-07-04).** Red-black GS is validated as a safe substitute for the bench's serial SOR: `tools/fluid_prototypes/utube_projection_redblack.py` reproduces the serial bench's results (near bit-identical on the default case) and converges with far fewer iterations — `N_ITER` as low as 5 passes every invariant, including a harder free-surface imbalance (`fill_left_to=15`, proper air gap) specifically re-tested to close the "does convergence need grow with arm-height difference" question: it doesn't — a bigger imbalance takes more *ticks* to fully settle, not more solve iterations per tick. Recommended starting point for the GLSL port: `N_ITER≈10-15`, `OMEGA≈1.6` (vs. the bench's serial `OMEGA=1.7`/`N_ITER≈60`) — re-tune against the actual in-engine acceptance scene in Checkpoint 6, this is a starting point, not a guarantee.

**Checkpoint 2 — implemented, needs in-editor verification.** `simulations/fluid_simulation.gd`: new `use_projection` (default false), `proj_G`/`proj_STIFF`/`proj_OMEGA`/`proj_N_ITER` exports (defaults from Checkpoint 1: `64.0`/`0.5`/`1.6`/`12`); new `_pressure_buf`/`_div_buf` (`RID`), riding the exact same page allocation as `_flux_buf` — grown/zeroed in lockstep inside `_ensure_flux_buf_capacity()` and `_free_flux_page()`. `shaders/fluid_flux.comp`: `SimParams` grew 52→64 bytes (`proj_G`/`proj_STIFF`/`proj_OMEGA` appended; `_upload_flux_params()` updated in lockstep — a struct/buffer size mismatch would otherwise silently corrupt every field after the change point); new bindings 11 (`PressureBuf`) / 12 (`DivBuf`), `coherent float data[]` matching `FluxBuf`'s pattern. Nothing reads or writes the two new buffers yet (`params.pass` is still only handled for 0/1) — this checkpoint is buffer/struct plumbing only, gated to be inert regardless of `use_projection`'s value. **Verify:** with `use_projection` at its default (false), confirm zero regression — settled lake stays at zero flux, sphere drop still falls/settles/conserves mass, sand/water interaction unaffected — exactly the existing manual regression pass from the Build-B-revert and mass-loss-fix sessions.

**Checkpoint 3 — implemented, needs in-editor verification.** `shaders/fluid_flux.comp`: new `params.pass == 2` (BuildRHS) — computes the provisional-gravity-flux divergence into `div_buf` for every liquid cell, per `tools/fluid_prototypes/utube_projection.py`'s `prov_fy()`/`solve_pressure()` RHS logic, ported to 3D (Y up; X/Z never contribute, gravity is vertical only). New helpers `proj_is_sealed()` (solid OR off-grid/chunk-not-loaded — **not** air; this is the port plan's non-negotiable #3, and it's why a loaded air cell and an unloaded chunk must be told apart, which `world_to_flat()`'s existing -1 convention happens to already conflate correctly for this purpose since both cases should be sealed here), `proj_is_liquid_same()` (mirrors Accelerate's "different liquids never exchange" rule), `proj_prov_fy()` (the provisional vertical flux itself). Density-correction masking (non-negotiable #2) implemented via `proj_is_liquid_same(gpos + up, mat_id)`.

Since this pass isn't wired into `step()`'s normal dispatch loop yet (that's Checkpoint 5), two new debug tools exist purely for manual verification: `FluidSimulation.debug_dispatch_build_rhs()` (one-shot dispatch over the current active+neighbour flux list) and `debug_dump_div(chunk_pos, local)` (prints `div_buf` at a voxel), both bound to a new debug key, **`G`**, in `world.gd` (dispatches BuildRHS then dumps `div` at the existing `F7`-key watch target). Keybind note (2026-07-04): `F7` sets the fluid watch target (crosshair raycast), deliberately kept off `V` since `V` is `player_controller.gd`'s sphere-visibility toggle; `G` is the pressure-debug trigger.

**Verify (no live gameplay change yet — BuildRHS is dispatched on demand but Transport still ignores its output) — CONFIRMED (2026-07-04):**
1. Point the crosshair at a fully-submerged voxel (liquid directly above and below) in a settled body, press `F7` then `G` — expect `div ≈ 0`. **Confirmed.**
2. Point at a free-surface voxel (air directly above) in the same settled body, press `F7` then `G` — expect the density-correction term absent (masked), matching the no-correction case rather than the submerged one.
3. Build a small U-tube (unequal arms, connecting passage, viewable through the GLASS material), dump `div` at both arm bases via the same `F7`/`G` sequence — expect a distinctly different value between the taller arm's base and the shorter arm's, mirroring the bench's non-uniformity (the engine-side precursor to Checkpoint 4's bottom-gradient check). **Confirmed, non-uniform** — top of the U-tube read `div=57.47`, bottom read `div=45.68`: real, non-trivial, non-constant values across the shape (not the flat-zero-everywhere failure mode). Re-check the *specific* arm-base-vs-arm-base comparison once Checkpoint 4 adds the actual pressure gradient readout, since that's the invariant that matters most.

**Checkpoint 4 — CLOSED (2026-07-04), verified.** `shaders/fluid_flux.comp`: new `params.pass == 3` (solve_red) / `== 4` (solve_black) — one red-black GS sweep each, coloring `(gx+gy+gz)&1` in **world** voxel coordinates (not `local_pos` — must be consistent across chunk boundaries). New helper `proj_classify_neighbor()` returns liquid/air/sealed for the 6-neighbour stencil, using `resolve_page_local()` (not just `world_to_flat()`) so a neighbour whose page is **withheld this tick** (outside the 1-hop dispatch radius) is correctly treated as sealed too — the same discipline as `compute_s_out`/`compute_s_in`'s existing `page<0` fix, and the port plan's non-negotiable #3. `proj_pressure_at()` reads a neighbour's `pressure_buf` value through the same paging.

`simulations/fluid_simulation.gd`: `debug_dispatch_solve_pressure()` — one-shot BuildRHS + `proj_N_ITER` red-black sweeps (zeroes `pressure_buf` fully first; no warm-start yet, that's a later concern), and `debug_dump_pressure(chunk_pos, local)`. Both `debug_dispatch_build_rhs`/`debug_dispatch_solve_pressure` now share a `_debug_prep_projection_dispatch()` helper. `world.gd`'s `G` key now runs the full solve and prints both `div` and `P` at the `F7`-key watch target.

**Verify — this is the single most important check in the whole plan** (the exact invariant that falsified Build B's max-relaxation field): build the U-tube test scene, press `F7` on each arm's base voxel then `G`, and confirm the **taller arm's base pressure is measurably higher than the shorter arm's**. Also check a settled full column at rest converges to a stable, sane profile (not oscillating/diverging across repeated `G` presses). If the bottom-gradient check doesn't hold, stop and re-examine before Checkpoint 5.

**In-editor readings so far are promising but need controlled geometry to confirm.** Manual (hand-dug) readings on a real U-tube showed real, non-trivial values at matched elevation (`P=143.98` vs `P=59.20` at world `Y=8` in two different arms, `div=0` in both — a substantial, non-uniform gradient, not the flat-zero/uniform failure mode). However hand-carving with the sphere/single-voxel tools produces inconsistent connector width and uncertain arm volumes, which could itself explain part of an observed imbalance rather than the pressure model — not a clean enough signal to close this checkpoint on.

**Scripted U-tube builder added (`world.gd`, key `U`)** to remove that uncertainty: `build_utube_test_scene()` places a solid shell (ADD, guaranteed sealed regardless of surrounding terrain) sized to exactly contain two equal-height 3×3 shafts (`UTUBE_ARM_HEIGHT=20`, `UTUBE_GAP=8` apart) joined by a bottom passage (all carved via `WorldOrchestrator.edit_box()`, a new general box-edit wrapper generalizing `edit_add_voxel`/`edit_remove_voxel` to arbitrary half-extents), then places a known-volume water block (`UTUBE_WATER_BLOCK=5` voxels tall) near the top of one shaft — not pre-filling the whole arm by hand — so it falls and fills the tube naturally once unpaused, matching an actual pour rather than a hand-placed splat of uncertain size. Aim at solid ground and press `U`; it pauses the sim immediately afterward so the rig can be inspected (`F7`/`G`) before manually unpausing (`P`) to watch it settle.

**Closing measurement (2026-07-04):** three separate scripted builds, each watched (`F7`/`G`) at a different, meaningful position along the source→target axis — pour arm's connector-floor cell (`P≈76.2` settled), the exact midpoint of the connecting passage (`P≈48.7-48.9`), and the far (target) arm's connector-floor cell (`P≈2.6`). A smooth, **monotonically decreasing** gradient from source → middle → target, exactly the physically-sensible profile a real connected-vessel pressure field should produce — not just "two points differ," a clean three-point ordering in the right direction. Since these were three independent builds rather than one shared structure (the debug tooling can only watch one voxel at a time), there's some residual chance of per-build confounding, but a perfectly-ordered monotonic 3-point result across identical script parameters is a strong, unlikely-to-be-coincidental signal. **Checkpoint 4 is closed on this basis** — the exact invariant that falsified Build B (a real, non-uniform, correctly-directed lateral gradient) is confirmed in-engine.

---

**Checkpoint 5 — implemented, needs in-editor verification.** `shaders/fluid_flux.comp`: new `params.pass == 5` (Correct) — `F = u* - grad(P)`, written into `flux_buf`'s same 3 owned lanes Accelerate would have written, mirroring Accelerate's exact face-participation rules (solid/air/different-liquid gating) so both passes gate identically; only the drive term differs. No `axis_cap` clamp (matches the bench's `correct_flux()` exactly — an oversized request is bounded downstream by Transport's existing `s_out`/`s_in` budgeting). Transport (`pass == 1`) is **completely unchanged** — it only ever reads `flux_buf`, never cares which pass populated it.

`simulations/fluid_simulation.gd`'s `step()`: when `use_projection` is true, the per-substep dispatch is now `BuildRHS → zero pressure_buf → (solve_red → solve_black)×proj_N_ITER → Correct → Transport`, replacing `Accelerate → Transport`. `pressure_buf` is zeroed fresh every tick (quasi-static M1 design — no warm-start/inertia yet, that's a separate M2 concern per the risks section above). When `use_projection` is false, the dispatch is byte-for-bit identical to the pre-port baseline.

**Verify, in order:**
1. **Regression, flag off (default):** confirm zero change from the current baseline — settled lake stays at zero flux, sphere drop still falls/settles/conserves mass.
2. **Pressure-null at rest, flag on:** a settled full column produces zero corrected flux — `debug_conservation_check` reads exactly 0, no changelog entries for that region across many ticks.
3. **Flat lake no-op, flag on:** pour a flat lake, confirm it doesn't churn or drift.
4. **The acceptance test:** build the scripted U-tube (`U` key), unpause with `use_projection` on, and confirm the two arms actually equalize — watched via `debug_dump_fluid_stats` (total fill stays stable, no repeat of the Build B mass-loss catastrophe) and visually through the GLASS shell.
5. **Cost sanity check:** confirm the per-tick GPU cost increase (2 → `4 + 2×proj_N_ITER` dispatches/sub-step) is tolerable at `proj_N_ITER=12` before considering M2 (temporal amortization) — this is expected to be slow, per the plan's cost callout; the question is only whether it's *unusably* slow yet.

**Live acceptance test result (2026-07-04): the U-tube actually equalizes with `use_projection` on.** This is the real milestone — not a diagnostic snapshot, live water movement driven by the pressure solve. Two issues found during this testing, tracked separately (neither blocks Checkpoint 5, both are real):

1. **Low-level settle-time jitter / checkerboard fill pattern.** A settled body (e.g. the sphere test) develops a persistent, non-uniform pattern of near-full vs. partial-fill voxels that doesn't clean up regardless of `G`/`STIFF`/`OMEGA`/`N_ITER` tuning. Log evidence at a single watched voxel: fill oscillates in a narrow band (e.g. 251-255) with `debug_conservation_check` printing a small (1-4 unit) gain/loss almost every tick for hundreds of ticks — never once landing on a genuinely silent "nothing changed" tick. This rules out "just needs more iterations": it's a persistent oscillation, not slow convergence. Leading hypothesis: the density-correction term (`STIFF*(255-fill)`) has no deadzone (nonzero for any cell not at *exactly* fill=255, so it perpetually nudges toward full), combined with `pressure_buf` being zeroed fully fresh every tick (quasi-static M1 design) meaning the solve re-converges slightly differently each tick, and `int(round(f))` in Correct rounding that tiny residual to `±1` — which is *above* the existing `MIN_FLOW=1` deadzone, so it's applied as real signal instead of filtered as noise. Same bug *class* as the original "fluid never settles" issue (a rounding operation with a nonzero fixed point at small magnitudes), different specific mechanism.

   **Diagnostic added: `proj_warm_start` export (default `false`).** When true, skips the per-tick `pressure_buf` zero in both the real dispatch (`step()`) and the debug tool (`debug_dispatch_solve_pressure()`), letting `solve_red`/`solve_black` refine last tick's field instead of re-solving from scratch — this is exactly what M2 (temporal amortization) does anyway for performance; testing it now is purely diagnostic.

   **RESOLVED (2026-07-04).** Tested `proj_warm_start=true` first — made things dramatically *worse* (whole voxels flashing between near-empty and 100+ fill every single frame), which ruled out the "convergence-consistency artifact" theory: if fresh-from-zero solves were merely landing in slightly different places near the same rest point, a better starting point should have calmed the jitter, not amplified it. This pointed at `OMEGA=1.6`'s over-relaxation itself — starting from zero every tick and running only `N_ITER` iterations was accidentally *damping* a real oscillation by capping how far it could run each tick; warm-starting removed that implicit damping and let it compound across ticks instead.

   Tested `proj_OMEGA=1.0` (plain Gauss-Seidel, no overshoot) with `proj_warm_start` back off and `proj_N_ITER=24` — **confirmed genuine, silent settling**: the exact same watched-voxel output (fill, flux, everything) repeated identically for 60+ consecutive ticks, something that never happened at `OMEGA=1.6` (which always printed a small nonzero `[Conservation]` delta every tick, even at apparent rest). Root cause confirmed: `OMEGA=1.6`'s intentional per-iteration overshoot (meant to speed up convergence) was instead overshooting into a stable, non-decaying limit cycle. Plain GS has no overshoot to go unstable with.

   **Defaults updated**: `proj_OMEGA` 1.6→1.0, `proj_N_ITER` 12→24 (compensating for losing SOR's convergence speed boost). `proj_warm_start` stays a diagnostic-only toggle, default off — real warm-starting (M2) will need its own design work (partial invalidation, residual-driven re-solve) rather than the naive "just don't zero it" version tested here, which is unstable independent of the `OMEGA` finding.

2. **A settled arm doesn't react to new water poured into a separate, already-idle arm** — confirmed as the expected sparsity limitation described earlier in this doc (the pressure solve only runs over the active+1-hop-neighbour chunk set; a fully idle far arm gets zero information until the disturbance propagates to it hop-by-hop through chunk activation). Not a bug, a known gap — see the Tier 1(a) discussion above. Not pursued further tonight.

---

## 7. First PR scope (concrete)

M0 + the M1 skeleton: add `_pressure_buf`/`_div_buf` with paging, the four new shader passes (build_rhs, solve_red, solve_black, correct), wire the new dispatch loop, add `G`/`STIFF`/`OMEGA` to `SimParams`, and stand up the U-tube acceptance scene with a `P`-field debug readback. Land it behind a flag (e.g. `FluidSimulation.use_projection`, default off) so the current baseline stays shippable while the new path is validated against the M1 gates. Update `ARCHITECTURE.md` (mark the to-do item in progress) and add any new bugs to `docs/BUG_LOG.md` as you go.

---

## 8. Handoff (2026-07-04) — a real design gap, blocking, needs a bench-first fix

M1 Checkpoints 1-5 all landed and mostly validated (red-black GS confirmed as a safe serial-SOR substitute, the U-tube bottom-gradient invariant confirmed working in-engine, a genuine `OMEGA=1.6` settle-time oscillation found and fixed via `OMEGA=1.0`/`N_ITER=24`). But testing past the U-tube into more general shapes (a dropped single voxel, a settled sphere) found a **structural gap**, not a tuning issue: **the design has no mechanism to equalize an uneven flat (single-layer) puddle.**

**Full mechanism** is in `docs/BUG_LOG.md`'s "DESIGN GAP (found 2026-07-04...)" entry — summary: `BuildRHS`'s divergence only has two terms, both requiring something stacked *on top of* a cell (provisional vertical gravity flux, and a submerged-cell density correction). The instant a body of water finishes spreading into a single layer — nothing anywhere has same-liquid directly above it — divergence is identically 0 everywhere, the free-surface boundary is `P=0`, and a Poisson equation with zero RHS and zero boundary has exactly one solution: `P≡0`. Corrected flux is then exactly 0 everywhere, no matter how lopsided the actual fill distribution is. Confirmed directly: a single dropped voxel spread outward and settled into a permanently uneven pattern (e.g. one column at `11` fill next to four neighbours at `62` each) that the system treats as fully at rest — silent, zero flux, zero conservation delta, indefinitely.

This wasn't caught by the original bench or by Checkpoints 1-5's testing because every test up to now (U-tube, flat *even* lake, columns, the settled sphere used for the OMEGA investigation) always had real vertical stacking providing a nonzero RHS somewhere, or was already perfectly even. Nobody had specifically tested "let something finish spreading into a flat, *unevenly*-filled single layer with zero vertical stacking anywhere" until this session.

**Why the old (pre-projection) system never had this problem, for context:** the existing `head()`-driven flux (still the default when `use_projection=false`) spreads water via a much cruder mechanism — any full cell pushes into any less-full neighbour unconditionally, tamed only by `gravity_bias`/`damp` tuning. That's physically wrong (it's exactly why the U-tube problem existed in the first place) but it does mean *something* always keeps pushing horizontally, so an uneven flat layer never gets stuck. Pressure projection deliberately replaced that with a real hydrostatic solve, correctly removing the wrong force — but nothing was put in its place for the no-vertical-stacking case. Notably, no cellular-automaton-style voxel engine we're aware of seems to solve this properly either, which suggests it's a genuinely hard case, not a simple oversight.

### Questions for whoever picks this up (Fable/Opus)

1. Is the right fix to broaden what counts as a divergence *source* — e.g. treat a liquid cell as needing correction if it's under-full relative to *any* same-liquid neighbour (not just "submerged", i.e. not just the one directly above), not only if it's under-full relative to 255? That would let the RHS see within-layer unevenness directly, but risks conflicting with the existing submerged-cell correction (double-counting?) and needs its own falsification-ladder treatment (rest state, no smearing, no oscillation) before it's trusted.
2. Is there a cleaner formulation entirely — e.g. some form of a horizontal "leveling" pressure term derived from a virtual elevation potential (`fill/255` as a proxy height even in a single layer), rather than trying to patch the existing submerged-only correction?
3. Does fixing this reintroduce any of the four originally-falsified approaches' failure modes (thin-film smearing, stale plateaus, oscillation) in disguise? The falsification ladder in `docs/BUG_LOG.md`'s original DESIGN entry is the regression suite to check against.
4. Should this be validated as its own bench scenario first (a single voxel dropped on a flat plane, watched until it either equalizes correctly or gets stuck) before touching `fluid_flux.comp` again? (Strong recommendation: yes, same discipline as the original design — `tools/fluid_prototypes/` is the right place.)

**Do not attempt a GLSL fix without validating the RHS change in the Python bench first.**

---

## 9. Checkpoint 6 (2026-07-05) — surface height-leveling pass, ported to GLSL

Bench work (answering the section 8 questions) landed in `tools/fluid_prototypes/FLAT_PUDDLE_HANDOFF.md`: an in-projection RHS patch (broadening the divergence source, section 8 Q1) works but is fundamentally limited — the free-surface `P=0` Dirichlet condition pins pressure to ~0 at *every* cell on a 1-deep layer, so no RHS source can drive a strong gradient; it only levels via the tiny residual flux that leaks past the pinning, which forces a near-zero `MIN_FLOW` deadzone that in turn reintroduces permanent settled-U-tube churn. **Recommendation (section 8 Q2, "a cleaner formulation"): bypass the pressure solve entirely** — a small, conservative, surface-elevation-driven diffusion applied directly to the surface layer, run alongside (after) the projection, which keeps ownership of vertical/connected hydrostatics unchanged. Bench-validated in `tools/fluid_prototypes/height_level_prototype.py` and independently re-verified (adversarial suite had a real bug — a silently-wrong `MIN_FLOW` default that made the in-projection alternative look far better than it ships; corrected numbers now match the handoff table exactly, see `FLAT_PUDDLE_HANDOFF.md` section 7).

**Ported to `shaders/fluid_flux.comp` as `params.pass == 6` ("LevelSurface"):**
- New `SimParams` fields `proj_LEVEL_RATE`/`proj_LEVEL_MIN` (struct grown 64→72 bytes), defaults 0.20/1.0 matching the bench exactly.
- Surface-cell test: liquid with no same-material liquid directly above (`proj_is_liquid_same(p+Y)` false). Fillable-dry test: air sitting directly on a sealed floor (`proj_is_sealed(p-Y)`) — mirrors the bench's `_is_surface`/`_fillable_dry` exactly.
- **Race-free by construction, no face ownership needed**: each invocation computes its own delta by summing contributions from its 4 lateral (X/Z only, never Y) neighbours, using the SAME trick Transport's `compute_s_out`/`compute_s_in` already rely on — both endpoints of a pair independently compute the identical exchange amount (opposite sign) from the same read-only pre-pass state, so conservation holds exactly with zero shared writes.
- **Chunk-boundary rule** (per section 6's porting note): a lateral neighbour whose chunk has no flux page allocated this tick (`resolve_page_local` returns page < 0 — i.e. withheld/not part of this tick's dispatch, same discipline as `proj_classify_neighbor`) is treated as `LEVEL_NB_NONE` (no exchange), never as air — because if that neighbour's own invocation doesn't run this tick, its half of the exchange would never be applied, an asymmetric leak exactly like the flux dispatch-range bug on Accelerate.
- Writes via the changelog only (never touches `voxels_a` directly), same single-writer discipline as every other pass, including the automatic conservation-check atomicAdd and dirty-bitmask marking.
- Wired into `simulations/fluid_simulation.gd`'s `step()`: dispatched immediately after Transport (appends to the same changelog, applied together in one `apply_changelog()` call), gated behind a new `use_surface_leveling` export (default `false`, independent of `use_projection` — has no effect unless both are true).
- One-shot debug tool `FluidSimulation.debug_dispatch_level_surface()` (resets + dispatches + applies the changelog itself, since unlike `debug_dispatch_build_rhs`/`debug_dispatch_solve_pressure` this pass has a real, immediately-visible effect on `voxels_a`), bound to debug key **`H`** in `world.gd`.

**BUG (found + fixed 2026-07-06, first in-engine verification pass): unnormalized 4-direction sum lost up to 80% of an exposed droplet's fill per tick.** The 1D-only bench (`height_level_prototype.py`'s row test) has at most one active lateral neighbour per cell, so it structurally could not catch this — a fully-exposed 3D cell was giving away `4 × LEVEL_RATE` instead of `LEVEL_RATE` total. Confirmed via an isolated 2D sanity check reproducing the exact reported collapse (`255→51` in one tick). Also corrected a wrong claim below (arms are a 3×3 cross-section, not 1-cell-wide — that's *why* the connector was exposed to this bug, not immune to it). Fixed by dividing each direction's contribution by a fixed `LEVEL_DEGREE=4.0` constant. Full write-up: `docs/BUG_LOG.md`'s "FIXED (2026-07-06): Checkpoint 6 surface-leveling pass lost up to 80%..." entry.

**Status: fix applied, NOT yet re-verified in-engine — the acceptance gates below have not been run against the corrected normalization.** `use_surface_leveling` stays default `false` until they pass.

**Acceptance gates (per `FLAT_PUDDLE_HANDOFF.md` section 6, run these in-engine, judge by per-cell spread + profile, NOT height):**
1. Flat uneven puddle (e.g. drop a single voxel on a flat layer) → spread ≤ 2, mass exact, and the droplet must NOT collapse/vanish (the specific failure mode of the bug above — watch fill over several ticks, it should decay gently toward the surrounding level, not lose the majority of its mass in 1-2 ticks).
2. Wide/ramp puddle → residual within the bench's documented bound (~1-11 depending on width, see the bench table), not a loose tolerance.
3. Settled U-tube churn gate: steady-state `max_move/tick ≈ 0` — verified in the bench at the engine's real settings; must reconfirm in-engine since chunk paging/dispatch adds a dimension the bench doesn't model.
4. Even flat lake → dead quiet (no churn), pressure-null at rest still holds.
5. U-tube still equalizes (per-arm mass balanced) — the leveling pass must not interfere. **Correction:** arms are a 3×3 cross-section (`UTUBE_ARM_HALF_XZ=1`), not 1-cell-wide as previously (wrongly) claimed here — re-verify this gate specifically now that the normalization bug is fixed, since the earlier "zero interference" claim was never actually true.
6. Conservation exact throughout (the existing `debug_conservation_check` / automatic atomicAdd already covers this for free).

**Fixes 2-6 landed in the same in-engine debugging session (2026-07-06), each bench-validated before touching GLSL — see `docs/BUG_LOG.md` for full write-ups of each:** (2) classify-by-what's-actually-there, not by the asker's own material; (3) `SUBMERGED_MIN` threshold (fill-above must *exceed* it, not merely be nonzero); (4) continuous per-cell damping instead of a boolean submerged gate (fixed a flicker); (5) per-pair integer rounding instead of aggregate-then-round (fixed a real, structural conservation leak; exposed an accepted ~10-unit quantization floor, tolerance widened accordingly); (6) **most recent** — replaced the vertical-fill-based "submerged" damping entirely with a **per-pair pressure-difference damping**, `clamp(1 - |P_self - P_neighbour| / proj_LEVEL_PRESSURE_MIN, 0, 1)`, querying the already-solved `pressure_buf` field directly instead of guessing from raw fill counts. Fixes the design gap the user originally flagged: a fully-submerged cell with genuinely open lateral neighbours can now correctly act toward them while still deferring toward any neighbour with a real pressure gradient (the old single per-cell damping value could only do one or the other for ALL neighbours at once). `proj_LEVEL_SUBMERGED_MIN` renamed to `proj_LEVEL_PRESSURE_MIN` (same byte offset, default unchanged at `2.0`). Bench: `tools/fluid_prototypes/level_surface_v3.py` / `level_surface_v3_check.py`, ALL CHECKS PASSED. Not yet re-verified in-engine.

**Post-fix-6 "one-sided mass drain" scare — RESOLVED as a bench artifact (2026-07-07), no fix 7 needed.** The drop-a-tower-on-a-puddle repro (`drop_settle_transient.py`) that appeared to show LevelSurface funneling a puddle's mass into a one-sided ramp was traced to the *bench's own geometry*: its builder carved AIR at `y=1` then overwrote those same cells with LIQUID, leaving **zero air cells** — a fully sealed domain, i.e. the documented vacuum-lock configuration. With no free-surface `P=0` anchor the pressure solve is pure-Neumann and the warm-started `P` drifts unboundedly, feeding fix 6's `|ΔP|` damping garbage; the drain's *direction* was seeded by the builder's half-cell-off-center tower (`W//2=15` vs. puddle center 14.5). With a real air headspace the same scenario spreads exactly symmetrically (`mirror_err=0`, `max|P|=0`, conservation exact) under both serial and red-black solvers. Bench fixed + free-surface assert added; new `open_drop_symmetry` regression in `level_surface_v3_check.py`. Two carry-forward notes: (a) fix-6 damping is by design meaningless inside genuinely sealed pockets — if leveling inside vacuum-locked bodies ever matters, damping must use only relative liquid-liquid P, never liquid-vs-dry (`P=0` convention); (b) the in-engine "ring at fill 1 / stream never ends" report that prompted the scare is now unconfirmed — the LevelSurface-free open baseline reproduces a very similar full-core + fill-1..3-ring profile, so re-test in-engine before attributing it to LevelSurface. Full write-up: `docs/BUG_LOG.md` (2026-07-07) and `LEVELSURFACE_TRANSIENT_HANDOFF.md`'s resolution banner.
