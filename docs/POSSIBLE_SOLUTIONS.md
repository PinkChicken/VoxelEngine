# Possible Solutions — Connected-Vessel Hydrostatics (U-tube)

> **⚠️ SUPERSEDED (2026-07-08) — a solution was chosen and shipped.** Phi-flood (family
> ranked #1 below) is what actually got built, ported to GLSL, and verified in-editor —
> see `docs/PHI_FLOOD_PORT_PLAN.md` and `docs/BUG_LOG.md`. Kept as historical record of
> the option space that was considered, not as an open decision.

**Companion to `CONNECTED_VESSEL_HANDOFF.md`.** That document defines the problem, the
failed attempts, and the acceptance criteria (first-fill equalization **and** reactivity after
settling, no false-rest, fast enough to develop against, ideally preserves flux feel). This
document gathers the *many* proposed avenues into a smaller set of distinct solution families,
presents each readably and without editorial slant, and ends with a ranking.

The ten "takes" supplied as source material overlap heavily. They collapse into **ten distinct
solution families** below. Each family lists the takes that support it, the mechanism in plain
terms, why it might work, and the risks noted by the authors. No family is dismissed here; the
ranking at the end is one opinion, explicitly flagged as such.

---

## A. Revive full Pressure Projection (Chorin) — with targeted fixes

**Source takes:** #5, #7, #8 (contrast section), #10-Option B (as fallback).

**Mechanism.** The provisional flux produces a divergence field (RHS). Solve the Poisson
equation `∇²P = RHS` (red-black Gauss-Seidel on GPU) and subtract the pressure gradient from the
velocity field to enforce incompressibility. Free-surface cells are the pressure-0 anchors.

**Why it might work.** It is the textbook-correct physics method — genuinely non-local and
provably correct in principle. Bench-validated for first-fill equalization. Three specific
in-engine failures were each isolated as *fixable, bounded* problems rather than design flaws:
- **Reactivity failure** (post-settling pour didn't move the far arm): the solver domain was
  built from the current liquid surface and excluded newly-poured water. Fix: extend the solve
  domain by ≥1 air layer above the highest surface so new weight enters the RHS.
- **Staircase false-rest** (e.g. `51/255/19/255/7/255/255`): per-face residual dropped below
  `MIN_FLOW` but the column is physically wrong. Fix: add a column-smoothness residual alongside
  the per-face gradient (flag a full cell sitting directly under a partial cell).
- **Performance**: the planned Tier 1 optimizations (active-region masking, warm-starting across
  ticks, coarse grids) were never actually reached.

**Risks.** Large integration/re-import cost from the `pressure-projection-wip` branch. A full
per-tick global solve is expensive; if the missing optimizations don't close the gap, feel/perf
regression returns. Unknown whether the three fixes fully resolve it without new artifacts.

---

## B. Amortized / Conditional Pressure Projection (hybrid)

**Source takes:** #1, #6, #10-Option B.

**Mechanism.** Keep the existing flux system as the fast path 100% of the time for local
wave-like motion. Run the pressure-projection solve *not every tick* but only when needed:
every N ticks (temporal amortization), or only on active/interesting pages (divergence above a
threshold, near free surfaces, recently edited). Warm-start from the previous pressure field.
Inject the resulting gradient as an extra accel term (or as a post-Transport correction).

**Why it might work.** Directly attacks the performance complaint (#A risk) by making the
expensive solve rare, while flux handles feel and the solve handles the connector-blindness.
Short pipes equalize in a tick or two when invoked; long pipes equalize gradually across ticks,
which the handoff itself says is acceptable for gameplay. Low-perf "flat lake" case: pressure
correction is ~zero, flux unchanged.

**Risks.** The "between projections" window is still locally blind, so reactivity after settling
could be sluggish (up to N ticks of latency). Doesn't inherently fix the staircase deadzone —
that needs a separate fix. Still requires porting a real Poisson solver.

---

## C. Piezometric-Head Flood (phi-flood)

**Source takes:** #8 (primary), appears as the recommended primary in several.

**Mechanism.** Define piezometric head `phi(p) = elevation(p) + pressure(p)/ρg`, which is
*constant* across a connected body at rest — this is exactly the quantity the U-tube equalizes
on. Instead of a Poisson solve, measure it directly:
1. **Free-surface sources:** a voxel is pressure-0 if it is liquid and either partially filled or
   has non-liquid above it. Computed fresh each tick from `voxels_a` (no persisted field →
   reactivity is automatic when new water is poured).
2. **Flood:** seed every source with pressure 0, then bounded Jacobi relax
   `pressure(p) = min over liquid neighbours (pressure(q) + step_cost)`, where `step_cost` is
   `FILL_CAP` per step within the body (i.e. depth below the surface). 8–24 iterations propagate
   pressure dozens of voxels per tick. Per-material gating so water and oil don't bridge.
3. **Rewire accel:** replace `head(p)-head(nb)` with `phi(p)-phi(nb)`. Everything downstream
   (Transport, damp, conservation) is unchanged.

**Why it might work.** Cheaper and easier to get right on a GPU than a Poisson solve (no serial
SOR, no queue/atomic worklist — plain Jacobi). Minimal blast radius: only the accel *input*
changes. Reactivity is free because phi is re-derived from current fill each active tick. A
staircase has non-constant phi across it, so flux keeps correcting (provided `MIN_FLOW` stays low
enough not to lock a real imbalance). Cross-chunk works via the same liquid graph flux already
uses, with the same `page<0 → 0.0` guard that bit Build B mirrored.

**Risks.** Bounded iterations mean a connector longer than ~iterations×32 voxels may not fully
equalize in one tick (mitigated: steady-state error stays under `MIN_FLOW` and flux keeps nudging
between floods). Different-liquid pressure bleed must be strictly per-material gated. Free-surface
flicker at the fill==255 boundary — damped by recomputing from canonical `voxels_a` and by existing
settle scheduling. Per-tick cost / long-connector convergence is the open question to validate
with the enlarged `KEY_U` rig.

---

## D. Surface-Height Field Propagation (S field)

**Source takes:** #3.

**Mechanism.** Add a `_surface_height_buf` (one float per voxel slot, init -1). After Transport,
run a propagation shader: free-surface liquid cells get `S = float(world_y)`; then run a fixed
number of relaxation iterations (max over 6 neighbours). Each connected component with ≥1 free
surface ends up with every liquid cell carrying that component's surface elevation `S`. Modify
`head()` to use `(S - y) * FILL_CAP` when `S ≥ 0`, falling back to the column scan for fully
enclosed liquid.

**Why it might work.** Crucially different from Build B: it propagates only the *surface
elevation* `S` (constant across a resting component), not head itself. The gradient comes from
`S - y`, so depth/pressure is preserved — this avoids the Build B failure where propagating head
flattened the needed gradient. No Poisson solve — just a few iterations of a max-filter, far
cheaper than Gauss-Seidel. Only the `head()` call site changes. Reactivity: when water is poured,
the free surface rises, the next propagation updates S, and head difference immediately drives
flow.

**Risks.** Iteration count vs. convergence speed (temporary artifacts during fast pouring if
propagation lags). Must be per-material masked. The "max" propagation vs. the "min-depth" flood in
#C are conceptually related but #C argues the depth-below-surface form is cleaner; the distinction
is worth a bench.

---

## E. Connected-Component / Graph Mode

**Source takes:** #2 (graph complement), #4 (plumbing model), #10-Option C, #1 (graph mention).

**Mechanism.** Periodically flood-fill / union-find connected same-material liquid voxels (GPU
friendly over a few passes). Collapse degree-2 pipes into graph edges with capacity. Solve a
*small* linear system (or simpler: proportional flow from mass imbalance and path resistance) on
the junctions. Inject corrective flux back into the fine sim.

For fully submerged, rock-enclosed water bodies, one variant (#4) treats the whole pipe as a
single graph node and *teleports* a calculated integer mass directly from the high-pressure
surface to the low-pressure surface via the changelog — guaranteeing exact volume conservation
and instant equalization across any pipe length.

**Why it might work.** Powerful and exact for plumbing-heavy, player-built machines and 1–2 voxel
pipes (the U-tube's natural shape). The non-local unit becomes a discrete, countable *component*,
not a continuous pressure field — sidesteps the "every voxel" cost. Plugs naturally into future
pumps/valves/pressure sources.

**Risks.** High complexity: topology changes, dynamic graphs on GPU, extraction cost. Best as a
*complement* to flux rather than a replacement. The "teleport mass" variant bypasses the existing
conservative transport machinery and risks new seam/materialization/chunk-boundary bugs — the
handoff explicitly warns against direct cell-to-cell transfer and prefers expressing correction as
face flux through existing Transport.

---

## F. Settled-Component Equalizer (narrow, topology-aware)

**Source takes:** #2 (the "top recommendation" framing), #9 (chunk-local imbalance variant).

**Mechanism.** Treat connected-vessel behavior as a *slow constraint correction*, not the main
force model. Keep flux exactly as the fast path. Add a second, much slower solver that activates
**only** for quiet, enclosed, saturated liquid networks. Detection gates (all must hold):
- liquid cells with ≥2 separated free-surface regions;
- a fully flooded path between them;
- local flux below a quiet threshold for a short grace window;
- surface heights differ by > one fill-cell equivalent;
- not currently receiving a large external disturbance.

Then: pick representative surface heights, compute `Δh`, find a connector path (bounded BFS
predecessor map), and inject a small signed flux *along that path* proportional to `Δh`, repeated
at low frequency (every 4–8 ticks), waking on any new water added. A cheaper variant (#9): skip
path-finding — for each active fluid chunk compute its internal y-imbalance and inject local
circulation (up at tall side → across → down at short side); neighbouring chunks compose into bulk
circulation (a chunk-granularity multigrid relaxation).

**Why it might work.** Spends non-local work *only when topology demands it* — most water (falling,
pooling, open spreading) never pays for a global solve. Directly addresses the projection failure
mode: a new pour marks the component dirty and schedules another pass, so reactivity is
first-class, not reliant on residual velocity surviving. Gives an explicit place to tune
equalization speed, pipe resistance, viscosity, and to model "long narrow pipe = slow plumbing but
same final level."

**Risks.** Requires a connected-component label pass (awkward on GPU; a CPU pass over active
liquid chunks is an option but a bottleneck for large bodies). Bounded BFS path extraction can get
expensive/awkward; if so, fall back to a compressed graph for only quiet saturated components. The
chunk-local variant may miss imbalance that spans a long pipe no single chunk sees — the path-based
version fixes that.

---

## G. Hybrid Flux + Sparse Body-Level Correction

**Source takes:** #10-Option A, #1 (Tier 1 framing).

**Mechanism.** Keep flux entirely. Layer a cheap non-local "pressure anchor": identify connected
bodies of liquid (flood fill on `voxels_a`, same-material, fully/partially filled). For each
body compute a global hydrostatic baseline — e.g. total fill ÷ cross-sectional area per elevation,
or a per-body sum — telling what the "should-be" pressure is at each voxel. Apply a *sparse*
correction flux only where local `head()` disagrees with the baseline by more than `MIN_FLOW`.
The correction feeds the existing persistent-flux buffer, preserving the damped-oscillator feel.

**Why it might work.** The flux system already does ~90% correctly; this only patches the specific
connector-blindness. Flood fill over a *sparse* active-liquid set is `O(active liquid voxels)`, not
`O(world volume)`. No Poisson solver, no per-voxel pressure field to maintain. Reactivity is free
because the flood + baseline are recomputed every few ticks.

**Risks.** GPU flood fill is irregular/branchy (CPU alternative is a bottleneck for large bodies).
Connected-body ID across chunk boundaries needs care. Tuning the correction magnitude to avoid
overshoot / new oscillation modes. Cross-section profiling `A(z)` for arbitrary 3D shapes is
non-trivial on a sparse grid.

---

## H. Improved Local Propagation (cheaper, limited)

**Source takes:** #1.3.

**Mechanism.** Stay in the local model but extend reach: for horizontal faces, also sample a short
distance along the horizontal plane (or average with neighbour heads); propagate "excess
pressure"/"deficit" signals locally with multiple sub-passes; store a short-lived decaying
"pressure hint" byte per voxel refreshed from neighbours. Or simply add an occasional (every 4–8
ticks) global-ish relaxation pass on *active liquid regions only*.

**Why it might work.** Easier to integrate than any global solve; cheap experiment to test whether
mild propagation helps U-tubes without breaking everything.

**Risks.** Likely won't fully solve arbitrary connected vessels — same fundamental limit as current
`head()` (information still can't cross an arbitrarily long saturated pipe in bounded hops). At
best a partial mitigation, explicitly flagged as such by its own authors.

---

## I. Flow-Driven Particle Ejection

**Source takes:** #4.3.

**Mechanism.** When a bottleneck's computed pressure exceeds a threshold, eject the water into the
planned GPU particle layer (Phase 9.5). The water becomes ballistic, shoots through available empty
space in the lower arm, and rebakes into the voxel grid.

**Why it might work.** Feels dynamic and violent (fits a chaotic sandbox), sidesteps the need for a
perfect mathematical rest state, and naturally avoids staircase by converting static pressure into
kinetic particles.

**Risks.** Doesn't produce a clean hydrostatic rest state; a gimmick for extreme cases, not a
general solution. Depends on the particle layer existing. Not a serious candidate for the
acceptance criteria as written.

---

## J. Coarse / Multigrid Pressure Grid

**Source takes:** #1 (coarse grid mention), #10 (multigrid mention).

**Mechanism.** Downsample liquid to 2³ or 4³ super-voxels for a bulk pressure solve, then upsample
corrections back to the fine grid. Handles big connected bodies cheaply while local flux keeps
surface detail. Multigrid/V-cycle as a further acceleration of the projection solve.

**Why it might work.** Makes a global solve tractable for large/long bodies. Complements #A/#B/#C
rather than standing alone.

**Risks.** Thin 1-voxel pipes fall through the coarsening (need a minimum-channel-width special
case or full-res fallback on small bodies). Adds a resampling layer with its own artifacts.

---

## How the families relate

- **#C (phi-flood)** and **#D (S-field)** are close cousins: both replace `head()`'s input with a
  propagated non-local quantity derived from free surfaces, via cheap iterative GPU passes, changing
  only the accel input. #D propagates surface *elevation*; #C propagates *depth-below-surface*. Both
  avoid the Build B mistake of propagating head itself.
- **#A / #B** are the same physics (Chorin) at different invocation frequencies.
- **#E / #F / #G** are the "connected component is the unit" family — identify bodies, then solve
  per-body rather than per-voxel. #F is the narrowest/most targeted; #E the most general; #G the
  lightest correction form.
- **#H / #I / #J** are supporting or partial techniques, most useful *combined with* one of the
  above rather than alone.

---

## Ranking (one opinion — flagged as such)

Ranked by *expected fit against the five acceptance criteria* (non-local awareness, reactivity,
no false-rest, dev speed, preserves flux feel), weighted toward "ship a gameplay-acceptable
result without a full rewrite." This is a starting point for discussion, not a verdict.

### 1. C — Piezometric-Head Flood (phi-flood)
Best balance of the five criteria. Smallest blast radius (one new pass, only the accel input
changes), no Poisson solve, reactivity is automatic because phi is re-derived from current fill
each active tick, and a non-constant phi keeps correcting staircases. The main unknowns (per-tick
cost, long-connector convergence) are bounded and testable on the existing `KEY_U` rig. Listed by
multiple takes as the recommended primary.

### 2. F — Settled-Component Equalizer
Strongest *design* match for "non-local work only when topology demands it," and makes reactivity
first-class by waking on disturbance. Lower steady-state cost than any per-tick solve because it
sleeps when quiet. Ranked just below #C mainly because it needs a connected-component label pass
(the harder integration step) and is a larger new subsystem; #C can be prototyped faster.

### 3. D — Surface-Height Field Propagation
Nearly as good as #C and conceptually simpler to explain. Ranked just below #C because propagating
*elevation* (max-filter) is subtly different from propagating *depth* (min-flood); the handoff's
own Build B failure was specifically about `max`-style propagation flattening gradients, so #C's
min-depth form is the safer bet until benched. Worth building alongside #C as an A/B.

### 4. B — Amortized / Conditional Pressure Projection
Keeps the known-correct physics of #A while attacking its #1 weakness (perf). Ranked below the
flood family because it still requires porting a real Poisson solver and the between-projection
window stays locally blind (reactivity latency). Good fallback if #C/#D disappoint on cost.

### 5. G — Hybrid Flux + Sparse Body-Level Correction
Clean "augment, don't replace" philosophy and cheap in principle. Ranked mid because the
connected-component + `A(z)` profiling on a sparse 3D grid is genuinely fiddly, and cross-chunk
body ID adds risk. Closely related to #F; could merge.

### 6. E — Connected-Component / Graph Mode
Most exact for plumbing/machines and 1–2 voxel pipes, and the "teleport mass" variant equalizes
instantly. Ranked lower for *this* problem because it's the heaviest to build and the teleport
variant fights the handoff's explicit "don't bypass conservative Transport" rule. Best as a later
complement, not the first move.

### 7. A — Full Pressure Projection (per tick)
Correct physics but the worst on dev-speed/perf/false-rest — the exact combination that was set
aside. Listed as the "textbook" anchor and the fallback if simpler methods fail, not the lead.

### 8. J — Coarse / Multigrid Pressure Grid
Useful *accelerator* for #A/#B/#C on large bodies, not a standalone solution. Thin-pipe risk.

### 9. H — Improved Local Propagation
Cheap experiment, honest partial mitigation, but structurally cannot cross an arbitrarily long
saturated pipe. Try early as a quick probe; don't expect it to pass the full test.

### 10. I — Flow-Driven Particle Ejection
Fun for extreme pressure vents, irrelevant to clean hydrostatic rest. Not a candidate for the
acceptance criteria as written.

---

## Suggested sequencing (cross-cutting, from the takes)

1. **Freeze and instrument the baseline flux solver** as the "local water" reference, with the
   `KEY_U` rig extended to log per-arm *mass* (not just height), fill profiles, first-fill
   equalization, **and a second pour after a true settled state** — the reactivity test that was
   historically skipped.
2. **Build the acceptance harness before the next solver**, and make it engine-faithful (real 3D,
   real red-black/sparse discipline) — the bench/engine fidelity gap was a major source of wasted
   investigation.
3. **Prototype #C (and A/B it against #D)** with diagnostics only first (log/visualize phi or S,
   component IDs, surface heights, connector paths) before wiring correction.
4. If #C/#D cost or convergence disappoints, escalate to #B (amortized projection), then #F/#G
   (component equalizer).
5. Keep #H as a cheap early probe and #J as an accelerator if large bodies appear.
