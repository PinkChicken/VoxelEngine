# Phi-Flood GLSL Port Plan — Connected-Vessel Hydrostatics

**Status (2026-07-08): PORTED and confirmed working in-editor on the real `KEY_U`
rig** — first-fill equalization and post-settling reactivity both verified, per
`docs/BUG_LOG.md` → "PORTED & VERIFIED IN-EDITOR (2026-07-08)". `K` (flood
iterations) needed raising from the Python bench's default of 16 to 40 for the
real rig's taller arms (see that entry). **One open bug found and reverted the
same session:** the "flat lake ripple" — see the BUG_LOG entry and
`tools/fluid_prototypes/python/test_pile_ripple.py` for the repro and analysis;
not fixed yet, left as a real design problem for whoever picks it up next. The
rest of this document is the original port plan (Sections 1-6 below), kept for
reference — it describes the design as implemented.

---

## 1. What is being added (one paragraph)

A per-tick, per-material piezometric pressure field `phi` replaces `head()` as
the input to Pass A's `dh` — and nothing else. Seeds: every free-surface liquid
cell carries `phi = fill`. A bounded Jacobi **max**-flood propagates pressure
through same-liquid neighbours with integer costs (`deficit(p) = 255 - fill(p)`):

| step direction (q → p) | candidate |
|---|---|
| down (q above p)  | `phi(q) + fill(p) - deficit(p)` |
| up (q below p)    | `phi(q) - 255 - deficit(p)` |
| horizontal        | `phi(q) - EPS_H - deficit(p)`, `EPS_H = 4` |

`phi(p) = max(seed(p), best candidate)`. All deficit terms vanish on full
cells, so `phi == head()` exactly for any settled isolated body — base feel is
untouched by construction. The `EPS_H` hydraulic slope keeps the field
non-uniform-at-elevation (the exact defect that killed Build B's max field),
and the deficit "suction" lets mass convect through saturated pipes instead of
smearing (the flood-form analogue of the Chorin bench's density-correction
term). Do NOT simplify either term away — both are load-bearing, each was
added against a measured failure.

## 2. Shader changes (`fluid_flux.comp`)

New passes (numbering after existing 0=Accelerate, 1=Transport):

- **`pass == 2` PhiSeed** — for each dispatched liquid voxel: if free surface
  (`fill < 255` or above is not same-material liquid) write `fill` to
  `phi_a[lane]`, else write `0` (a valid lower bound — true phi ≥ fill ≥ 0).
  Air/solid lanes: write `-1` sentinel (or just 0; Accelerate gates on
  material anyway).
- **`pass == 3` PhiFloodStep** — one Jacobi hop: read `phi_src`, write
  `phi_dst` with the max-of-candidates rule above (6 same-material liquid
  neighbours through the normal `read_a` + page-table lookups). Ping-pong via
  a `phi_parity` param field (or two bindings swapped CPU-side). Run K times.
- **Accelerate (`pass == 0`) change** — one line pair: replace
  `head(gpos, active_mat)` / `head(nb, active_mat)` (fluid_flux.comp:523) with
  `phi_read(gpos, active_mat)` / `phi_read(nb, active_mat)`, where `phi_read`
  returns the flooded value for same-material liquid and 0 otherwise (same
  gating `head()` applies). `head()` itself can stay in the file for
  reference/debug.

New buffers/bindings:

- `phi_buf_a`, `phi_buf_b` (bindings 11, 12): one s32 per voxel, **paged
  exactly like `flux_buf`** but stride 1 instead of 3 — reuse
  `flux_page_table` (binding 8) unchanged, index `page * vpc + local`. No new
  page-management code: phi pages ride the same allocation/zero/free lifecycle
  as flux pages (`_allocate_flux_page`/`_free_flux_page` allocate both).

Integer math notes: all costs are exact integers; GLSL `int` everywhere; no
rounding is involved until Accelerate's existing `int(round(...))`, which is
unchanged.

## 3. Dispatcher changes (`fluid_simulation.gd`)

Per sub-step order: **PhiSeed → PhiFloodStep × K → Accelerate → Transport.**

Critical perf point: PhiSeed, the K flood steps, and Accelerate have **no CPU
readback between them** — record all K+2 dispatches with pipeline barriers in
a single submit, then sync once before Transport (which needs the changelog
reset around it as today). Per-tick submit/sync count stays at **2**, same as
master. (Verify `compute_helper.gd` can enqueue multiple `run()` calls with
barriers before one `submit()`; if not, that helper extension is the first
sub-task.)

**K (flood iterations) — UPDATED 2026-07-08 after in-editor `KEY_U` testing.**
Each step propagates one hop, and — critically — phi is re-seeded and
re-flooded from scratch every tick with NOTHING persisted across ticks (this
matches the validated Python port exactly). That makes K a hard **per-tick
reach budget**, not a "converges given enough ticks" iteration count: two
seeds farther apart than K can never connect, ever, no matter how many ticks
run, because nothing carries information forward between them. The Python
bench's avg-7/max-~30 convergence numbers were measured against its OWN small
test rigs and do **not** transfer to the real `KEY_U` scene, whose arms are
much taller (`world.gd`'s `UTUBE_ARM_HEIGHT = 32`). Confirmed by direct
in-editor test: `K=16` left the U-tube stably ~2x mass-imbalanced (not still
converging — a genuine per-tick reach shortfall); `K=32` fully equalized it.
Shipped default is `K=40` for headroom above the demonstrated-working 32 (see
`FluidSimulation.phi_flood_iterations`). Rule of thumb for sizing K on a new
rig: it must cover roughly (down one arm) + (across the connector) + (up the
other arm) in voxels. Under-convergence is still SAFE at any K: phi degrades
toward `head()` (the invading ramp just doesn't reach as far), never
overshoots — a too-small K produces a stable but uneven rest, never an
explosion or leak. If a much taller/longer rig needs K large enough to hurt
per-tick dispatch cost, the real fix is cross-tick warm-starting (persist
`phi_buf` and continue relaxing it tick-to-tick instead of reseeding from
zero) rather than an ever-larger K — not implemented; a good next step if a
future scene's geometry outgrows K=96 (the current export's ceiling).

**Flood dispatch range — MUST be wider than the flux dispatch range.** This is
the second structural discovery of the bench work: the sleep scheduler is
itself connector-blind. On a long connector the discharge endpoint (the far
arm's surface) sleeps several chunks outside `_build_flux_chunk_list`'s
active+1-hop range while everything inside it is saturated → zero transfers →
idle-guard sleeps the rig with the imbalance intact (measured in the port:
arm A y=8 vs arm B y=2, asleep forever). Fix, ported from the bench:

1. Build the **flood chunk list** as the connected component of liquid-bearing
   chunks reachable (through liquid-bearing face-adjacency) from any active
   chunk — a chunk-granularity CPU BFS over `ChunkManager` masks
   (`mark_simulation_active(LIQUID)` already tracks which chunks hold liquid).
   This costs O(liquid chunks), only while something is awake. The flood
   passes are read-only on `voxels_a`, so covering sleeping chunks is safe.
   Phi pages are allocated for the flood list (and freed by the same idle
   logic).
2. **Pressure-wave wake rule** (end of the last PhiFloodStep, or a tiny extra
   pass): a cell wakes its chunk (atomicOr into the existing dirty bitmask —
   reuse `mark_dirty`) iff
   `phi - fill > WAKE_MARGIN (=64)` **and** the cell is genuinely
   dischargeable: air above, OR `fill < 255 - REST_TOL` where
   `REST_TOL = max(1, settle_threshold - 1)`. Sealed-full cells are pure
   static pressure and must never wake (vacuum-lock/aquifer behaviour is
   preserved — sealed columns seed `phi = fill` exactly like `head()`).
   At equilibrium the EPS_H slope makes invading ramps undershoot local
   seeds, so the signal is exactly zero at rest and sleeping still works —
   verified `active=0` end states on every bench rig.
   The CPU then feeds those dirty bits into `_active_chunks` exactly like a
   normal transport wake.

**Do not skip the two wake-loop cost fixes** (down-step deficit, up-step
`-FILL_CAP` not `-fill(q)`): both were found as *permanent* wake-loops (chunk
awake forever, zero transfers) in the port — see `phi_flood.py` inline
comments and the BUG_LOG entry.

## 4. What does NOT change

- `head()`'s code, `eff_flux`/`MIN_FLOW`, `compute_s_out`/`compute_s_in`, the
  Transport pass, the changelog, conservation checking, dirty bitmask
  decode, page-table restore/unmap discipline — all untouched.
- Conservation is inherited: the fix only changes what Accelerate writes into
  flux lanes; the conservative transport machinery is identical. Bench:
  max |cons delta| 9–12 per tick across all rigs (same clamp-loss class as
  master).

## 5. Acceptance in-engine

Use `KEY_U` (`world.gd::build_utube_test_scene`) with the existing enlarged
rig. Gate on the same 5 criteria, with the bench-calibrated expectations:

1. Per-arm **mass** equalizes (small rig: gap 66/11475 ≈ 0.6%; expected error
   bound `EPS_H · path_len / 255` cells — sub-voxel for game-scale pipes).
2. Second pour after full settle+sleep: far arm rises to the new shared level
   (this exercises the wake rule; watch `Voxel/fluid_active_chunks` wake and
   then return to 0).
3. Injected gross staircase (`51/255/19/255/7/255`) resolves to monotone
   within ~150 ticks, then sleeps; residual inversions ≤ `settle_threshold-1`
   fill units (the substrate's own rest tolerance).
4. Perf: flood adds K small dispatches inside an existing submit; measure
   against master's frame time on the same scene before/after.
5. Feel: lake settle / falling block visually unchanged (phi == head there).

## 6. Known risks / open questions for the port

- **GPU cost of the component-wide flood** on large lakes: the flood list can
  be much bigger than the flux list. Mitigations, in order: skip flood
  entirely when no liquid chunk is active (already implicit); cap the BFS
  component radius and accept slower cross-body response; amortize (flood
  every Nth tick — safe, phi only lags). Only escalate to coarse grids if
  measurements demand it.
- **Chunk-boundary hop stalls**: transport still stalls one tick at dispatch
  frontiers (`page<0 → 0.0` guards) — the equalization wave crosses ~1 chunk
  per tick worst-case. Acceptable in the bench; watch it at KEY_U scale.
- **`glass` vs `rock` shells**: irrelevant to phi (any non-same-liquid is a
  wall), but the seed rule treats sealed-under-solid cells as seeds
  (`phi = fill`) exactly like `head()` — intentional, keeps vacuum-lock.
- **Oil/water**: flood is strictly per-material (same gating as `head()`);
  different liquids never exchange phi. Not exercised by the bench rigs —
  add a two-liquid rig before shipping if oil matters soon.
