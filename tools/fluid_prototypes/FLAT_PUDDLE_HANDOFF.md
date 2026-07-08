# Flat-Puddle Leveling — Findings & Handoff to the Implementation Agent

**Audience:** the agent porting the pressure-projection design into `fluid_flux.comp`.
**Scope:** a design gap discovered *after* the projection design was validated — flat, unevenly-filled puddles never level. This documents the gap, the fix that was first proposed (and why it was subtly wrong), everything else we tried (and why it failed), and the two approaches that actually work (and *why*). All claims here are backed by runnable benches in this folder.

Related files in `tools/fluid_prototypes/`:
- `utube_projection.py` — the validated base projection sim (serial SOR; the design being ported).
- `flat_puddle_test.py` — the flat-puddle bench. **Now carries the corrected in-projection fix** (`max` source + `MIN_FLOW=0.1`) plus the full regression suite.
- `height_level_prototype.py` — the **recommended** alternative fix (a dedicated surface height-leveling pass).
- `flat_puddle_adversarial.py` — adversarial stress cases (linear ramp, wide puddle, dry-gap, checkerboard, `MIN_FLOW` sweep).

> **Metric warning (this bit us repeatedly).** Judge leveling by **per-cell fill spread and a fill-profile dump**, never by "surface height" / `column_top`. A lopsided *thin smear* reaches the same surface height at different densities, so height-based checks give false passes. Every "spread=" number below is `max(fill) − min(fill)` across the puddle's cells; lower is flatter.

---

## 1. The problem

The projection's `BuildRHS` produces a divergence with exactly two terms, **both of which require vertical stacking**:
1. provisional gravity flux — only nonzero where there is same-liquid *directly above* a cell;
2. the density correction `STIFF·(FILL_CAP − fill)` — masked to *submerged* cells (same-liquid directly above).

The moment a body of water finishes spreading into a **single flat layer**, nothing anywhere has same-liquid above it, so **`div ≡ 0` everywhere**. Combined with the free-surface boundary condition (`P = 0`), the Poisson equation `∇²P = 0` with zero boundary has exactly one solution: `P ≡ 0`, *exactly*. Corrected flux `F = u* − ∇P` is then zero, so an uneven flat puddle is a **permanent, silent fixed point** — e.g. one column stuck at fill 11 next to neighbours at 62, forever, with zero conservation drift.

This was never caught because every prior test (U-tube, columns, sphere drop, even lakes) always had vertical stacking *somewhere* providing a nonzero RHS, or was already even.

---

## 2. Root cause — the insight that governs everything below

**On a 1-deep flat layer, every liquid cell has air directly above it, so the free-surface `P = 0` Dirichlet condition pins pressure to ≈0 at *every single cell*. The pressure solve therefore cannot build a meaningful gradient no matter what source you inject into the RHS.**

This is *the* reason the problem is hard, and it predicts every result that follows:
- Any **RHS-source** patch can only work by the *tiny residual flux* that leaks past the pinning — which is why it needs a near-zero `MIN_FLOW` deadzone to avoid being thresholded away.
- Any **provisional-flux** you add gets **cancelled** by the projection (the projection's whole job is to remove divergence, and a horizontal leveling flow into a dip is exactly the divergence it removes).
- The projection is additionally **blind to divergence-free flows**: a *uniform* leveling flow across a linear ramp has zero divergence, so no source term can see it — this is why averaging-type fixes plateau on smooth slopes.

**Conclusion:** a fix that avoids a near-zero `MIN_FLOW` must do the flat-surface leveling **outside the pressure solve.**

---

## 3. The fix that was first proposed (and forced `MIN_FLOW`)

**Proposal:** add a surface-leveling term to the RHS on free-surface cells only. For each surface cell, average the fill of its horizontal liquid neighbours; if that average exceeds the cell's own fill, add `STIFF_H · (avg − fill)` to `div`. Implemented in a subclass that *also* lowered `MIN_FLOW` from the base `0.5` to `0.01`.

**It passed its own suite** (flat puddle `[62,62,11,62,62] → [51,51,51,51,51]`, plus U-tube/regression). But it was faulty in three compounding ways, and the flaws hid each other:

### 3a. `avg` is a discrete Laplacian → incomplete on slopes
`avg(neighbours) − self` is a discrete Laplacian, which is **zero on any linear gradient**. So the term flattens sharp dips but leaves a smooth residual tilt, and **the residual grows with puddle width**. Measured residual spread (lower = flatter):

| case | `avg` @MF=0.5 | `avg` @MF=0.01 |
|---|---|---|
| 5-cell dip | 10.2 | 0.2 |
| linear ramp | 58.5 | 1.2 |
| 9-cell 2-dip | 24.4 | 7.8 |
| 25-cell dip | 40.2 | 15.3 |

The bench hid this behind a loose `assert spread < 15` on the 9-cell case; the 25-cell case (residual 15) was never tested.

### 3b. The `MIN_FLOW = 0.01` change was undisclosed **and** load-bearing
Notice the table: `avg` only levels acceptably at `MF=0.01`, not at the engine's real deadzone. The fix silently depended on a 50× reduction of the flux deadzone. That deadzone is not cosmetic — it exists to stop sub-threshold churn on settled water (see the `MIN_FLOW`/`eff_flux` history in `BUG_LOG.md`).

### 3c. Low `MIN_FLOW` reintroduces churn on settled bodies
Steady-state per-tick movement of a *settled* U-tube vs `MIN_FLOW` (should be ~0):

| `MIN_FLOW` | base projection | with fix |
|---|---|---|
| 0.5 | 0.0000 | 0.0000 |
| 0.1 | 0.0000 | 0.0000 |
| 0.05 | 0.3819 | 0.3397 |
| 0.01 | 0.2357 | 0.2401 |

At `MF ≤ 0.05` a settled U-tube churns ~0.24–0.38/tick **forever**. In-engine that means the changelog never quiets and chunks never sleep — a real performance regression. (Note it's **identical for base and fix**, so the churn is a latent property of the base projection that a tiny deadzone merely *exposes* — not something the leveling term introduces.)

**Net:** the proposal "works" only by choosing a weak operator (`avg`, which under-levels) and papering over it with a hidden `MIN_FLOW=0.01` that itself churns. Do not ship it as-is.

---

## 4. What we tried instead, and why each did not work

### 4a. Surface-slope as a provisional *flux* (through the projection) — FAILED
Idea: the missing force is the surface slope, so add a horizontal provisional flux `u*_x = −G_H·∇(y + fill/255)` between surface cells and let the projection correct it. Result: **worse than the avg patch everywhere** (5-cell 65, ramp 98).
Why: the projection **cancels provisional flux** — it removes exactly the divergence the leveling flux creates. A source is a *demand* the projection satisfies; a provisional flux is a *supply* it deletes. This is the flip side of the root-cause insight.

### 4b. Connected-component mean as an RHS *source* — FAILED
Idea: strengthen the source by targeting the true level — `source = STIFF_H·(component_surface_mean − fill)` — so the driving force stays strong across the whole width instead of attenuating like a Laplacian. Result: **ramp 94, 25-cell 184** (worse).
Why: same root cause. The component mean is a better *target*, but it is still injected as an RHS source into a solve whose `P` is pinned to ≈0 by the free-surface BC. The stronger target does not help when the solve can't act on it.

### 4c. Integer surface height-leveling — PARTIAL
Idea: level the surface layer with a *direct* integer fill exchange (bypassing the solve), which should self-terminate with no deadzone. Result: gap-merge works, but leveling stalls (5-cell 8, ramp stuck at `[112,116,120,124,128]`).
Why: `int(diff · 0.2)` truncates to **0 whenever `diff < 5`** — integer quantization just trades the flux deadzone for an *integer-truncation* floor. Right idea (leave the projection), wrong quantization.

---

## 5. What worked — two viable fixes (and a heavy exact option)

There is a genuine theorem lurking here: **no purely-local, threshold-limited mechanism can reach *exactly* level** — there is always a floor (flux deadzone, integer truncation, or a pairwise stop-threshold), which leaves a small width-dependent residual slope. Exact, width-independent leveling requires a **non-local** step. So there are three tiers; pick per how flat lakes need to be.

Residual spread at each candidate's intended settings (lower = flatter):

| approach | needs low `MIN_FLOW`? | 5-cell | ramp | 9-cell | 25-cell | gap-merge / dry-spread | settled churn | new mechanism |
|---|---|---|---|---|---|---|---|---|
| **`max` source, `MF=0.1`** (in-projection) | no (0.1) | 1.2 | 2.7 | 8.2 | 17.2 | ✗ | none | none — RHS tweak |
| **fractional height-leveling pass, `MF=0.5`** | **NO** | 1.4 | 4.0 | 6.2 | 10.9 | ✓ | none | one extra pass |
| **connected-component mean (direct, non-local)** | no | exact | exact | exact | exact | ✓ | none | flood-fill + reduce |

### 5a. Corrected in-projection fix — `max` source + `MIN_FLOW = 0.1`
The minimal correction to the original proposal: use **`max` neighbour, not `avg`** (max levels ramps and wide puddles far better — see the table; it responds to the peak rather than the local curvature), and use **`MIN_FLOW = 0.1`, not `0.01`** (the churn-free floor from §3c). Read the numbers together: `max@MF=0.1` matches `avg@MF=0.01` on leveling **but is churn-free**. So switching `avg → max` *buys back* the churn-free deadzone regime at equal quality — same cost, one operator change.
This lives in `flat_puddle_test.py` now and passes the full regression suite (5-cell → 1.2, wider → 8.2, settled-U-tube churn 0.0000, U-tube/settled-partial/conservation all green).
**Limits (accept or escalate):** wide flat puddles keep a gentle residual slope (25-cell ≈ 17); cannot spread into dry cells or merge two puddles across a dry gap (those need materialization into air, which sits outside the pressure domain).

### 5b. RECOMMENDED — a dedicated fractional surface height-leveling pass
A small, conservative, surface-elevation-driven diffusion applied **directly to the surface layer**, run **alongside** the projection (which keeps ownership of vertical/connected hydrostatics). Per pass, for each horizontal pair of surface-layer cells `a,b` with `|fill_a − fill_b| ≥ LEVEL_MIN`, move `(fill_a − fill_b)·LEVEL_RATE` from high to low (fractional, conservative). Defaults: `LEVEL_RATE = 0.20`, `LEVEL_MIN = 1.0`. Implemented in `height_level_prototype.py`.

Measured at the engine's **real `MIN_FLOW = 0.5`**: 5-cell 1.4, ramp 4.0, 9-cell 6.2, 25-cell 10.9; even lake self-terminates (churn 0.000); dry gap merges (centre fills to 98); U-tube unaffected (imbalance 0.10, mass exact).

**Why it works — in detail:**
- It **bypasses the pressure solve for flat leveling**, so the `P`-pinning of §2 simply doesn't apply. It is a direct fill exchange, not a flux the solve can cancel or a source the pinning can defeat.
- It is **decoupled from `MIN_FLOW`** because it does not go through the `eff_flux` deadzone path at all — its own `LEVEL_MIN` threshold controls the residual, independently. That is exactly why it levels at `MF=0.5` where every in-projection approach needs `MF=0.01`.
- It is **fractional, not integer**, so it avoids the §4c truncation floor; and it **self-terminates** below `LEVEL_MIN` (a settled/even surface produces no moves), so there is no churn — the thing the low deadzone caused.
- It is **physically correct and targeted**, not the old crude `head()` push: it moves water down the actual surface-elevation gradient, and it only fires on horizontally-adjacent surface-layer cells. In the U-tube the arms are one cell wide, so it never fires there — hence zero interference (verified).
- It is **conservative by construction** (every move is a paired −/+ on two cells) and, by treating an empty floor cell as elevation `y` (fill 0), it also **spreads into dry area and merges across gaps** — capabilities no RHS-source term has.

**Costs / caveats:** it is a *second* mechanism (its own GLSL pass, and its own red-black or Jacobi ordering to avoid write races — the prototype uses a simple Jacobi-over-`+x`-faces sweep). Its residual is still width-dependent (a gentle slope on very large lakes) because a pairwise stop-threshold is still "local" per §5's theorem. Order it **after** the projection+transport in the tick.

### 5c. Exact option — non-local connected-component mean (direct)
If perfectly flat large lakes are required: each tick, flood-fill connected liquid, compute the mean fill over each component's surface cells, and redistribute each surface cell **directly** toward that mean (fractional, conservative, stop below a small threshold). Because the target is the global mean, the driving force is width-independent, so leveling is **exact** regardless of size and needs no low deadzone. Cost: a per-connected-surface reduction every tick (non-local). The engine already has connected-component/flood-fill machinery for rigid-body detachment (see `ARCHITECTURE.md`), so there is precedent, but it is the heaviest option. Deferred unless gameplay demands mirror-flat lakes.

---

## 6. Recommendation & porting notes

**Recommended: 5b (fractional surface height-leveling pass).** It is the only approach that meets the "no near-zero `MIN_FLOW`" requirement cleanly, self-terminates (no churn), and throws in gap-merge/dry-spread. Keep 5a (`max` + `MF=0.1`) as the fallback if a second GLSL pass is undesirable and the residual/no-spread limits are acceptable. Use 5c only if exact flat lakes become a requirement.

**For the GLSL port of 5b:**
- New pass after Accelerate→Transport (and after any projection passes): read `voxels_a`, write the changelog like every other sim — it must remain the single-writer discipline; do the fill exchange as changelog entries, not direct `voxels_a` writes.
- Surface-cell test = "liquid with the `+Y` neighbour not the same liquid" (air or solid above); use world position for red-black coloring so it's consistent across chunk boundaries.
- **Chunk-boundary rule is the same as everywhere else in this design:** a withheld/unavailable neighbour page is a **sealed wall** for the leveling pass (skip that face), never treated as air/empty — otherwise you leak or spuriously spread at the dispatch-range edge (this is the same `page < 0 → 0.0` discipline already in `fluid_flux.comp`).
- Expose `LEVEL_RATE` and `LEVEL_MIN` as tunables (like `damp`/`g_dt`).

**Acceptance gates for whichever fix is ported (build these in-engine, judge by per-cell spread + profile, NOT height):**
1. Flat uneven puddle `[62,62,11,62,62]` → spread ≤ 2, mass exact.
2. Linear-ramp and wide-puddle → residual within the documented bound for the chosen approach (don't accept a loose `<15`).
3. **Settled U-tube churn gate:** steady-state `max_move/tick ≈ 0` at the chosen `MIN_FLOW`.
4. Even flat lake → dead quiet (no churn), and pressure-null at rest still holds.
5. U-tube still equalizes (per-arm mass balanced) — the new mechanism must not interfere.
6. Conservation exact throughout.

Everything above is reproducible: `python flat_puddle_test.py`, `python height_level_prototype.py`, `python flat_puddle_adversarial.py`.

---

## 7. Post-handoff verification pass (independent re-run, found & fixed 3 test-quality bugs)

A separate agent independently re-ran every bench in this folder and read the source rather than trusting the printed output, per this project's standing discipline ("the test passes" is not the same as "the thing being tested is true"). It found three real issues, since fixed:

1. **`flat_puddle_adversarial.py`'s `flat_row()` silently defaulted to `MIN_FLOW = 0.01`** — the exact deadzone §3c documents as unshippable (settled-U-tube churn). Four of its seven sections ([1] ramp, [2] wide-9, [6] checkerboard, [7] wide-25) called it without passing `min_flow`, so they were quietly re-benchmarking the rejected config instead of the shipped `MIN_FLOW = 0.1`. This made the width-dependent residual look much smaller than it really is (ramp: printed 0.3, real 2.7; wide-9: printed 0.8, real 8.2; wide-25: printed 1.8, real 17.2 — all now match §5's table after the fix). **Fixed:** `flat_row()`'s default is now `min_flow=None`, which leaves whatever `MIN_FLOW` the class under test actually ships with untouched, so every scenario reflects the real shipped config unless a test is deliberately comparing deadzones (as §[3] already did correctly).
2. **`flat_puddle_test.py`'s `test_single_column_drains`** built a column with no ledge/gap for water to fall through (just three equal-fill cells on solid floor, air on all sides), asserted only mass conservation, and unconditionally printed "thin column drains (no pinning)" even though the column never moved. **Fixed and renamed** to `test_single_column_compresses`: now asserts the fill pattern actually changes and that it compresses toward full cells (128,128,128 → ~255,129), which is the real (vertical-settling, not draining) behavior this geometry exercises.
3. **`height_level_prototype.py`'s no-churn check** only tested an already-flat lake (which never moves regardless of whether the leveling pass is correctly quiescent or just never triggered), so it couldn't distinguish "settles and stays quiet" from "trivially had nothing to settle." **Fixed:** added a second check that perturbs a real puddle, lets it fully settle (3000 ticks), and *then* measures per-tick movement — confirmed genuinely `0.000` post-transient, so the underlying mechanism was fine; only the test was too weak to prove it.

All benches re-verified green after the fixes; the corrected adversarial numbers now match §5's summary table. **Takeaway for whoever picks this up next:** keep doing exactly what caught these — rerun the benches yourself and read the scenario-builder code, don't take a green suite at face value, especially where a helper function has its own independent default that could silently diverge from the class it's testing.
