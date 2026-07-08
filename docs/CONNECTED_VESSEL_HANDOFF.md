# Connected-Vessel Hydrostatics (the "U-tube test") — Handoff

**Status (2026-07-07):** unsolved. Two serious attempts have been tried and set aside. `master` currently runs the original flux-only system, which is known — and now precisely understood — to be structurally unable to pass this test. This document is the single entry point for whoever picks this up next: what the test actually requires, everything tried so far and why each attempt failed, why the *current* shipped system can't pass it either, and what a solution actually needs to be able to do.

---

## 1. The test

Build (or use `world.gd`'s `KEY_U` → `build_utube_test_scene()`) a sealed U-shaped vessel: two vertical shafts connected by a horizontal passage at the bottom. Pour a known volume of water into one arm.

**Passing requires two things, not one:**
1. **First-fill equalization.** The water finds its level — both arms end up at the same height, with per-arm *mass* balanced (not just matching surface height — a thin smear across a wide arm can reach the same height as a dense column in a narrow one while holding far less water; judge by `sum(fill)/FILL_CAP` per arm, never by height alone).
2. **Reactivity after settling.** Once the system has already equalized once, pour *more* water into either arm. The other arm must react and rise to find the new shared level. **This second requirement was never actually verified in either serious attempt** — both were judged solely on first-fill equalization, which is necessary but not sufficient. A system that only equalizes once, on first fill, hasn't modeled a connected vessel; it's modeled a one-shot pour.

Both criteria must hold **while remaining fast enough to iterate on interactively** — drop water, watch it settle, tweak something, repeat, without a multi-second stall or unplayable framerate. This turned out to be load-bearing: a design that's directionally correct but too slow to test quickly is very hard to actually finish, because every iteration cycle costs minutes instead of seconds.

---

## 2. What's been tried

### 2a. The original flux-only system (currently shipped, `master`)

Each voxel face computes a provisional flux from `head(p, mat)` — a **strictly local, straight-up-only** column scan (walks upward from `p`, summing consecutive full same-liquid cells; stops at the first partial cell or non-matching material). Adjacent cells exchange flux based on the *local* difference in their own `head()` values. No global solve — this is a pure local diffusion process, cheap and fast, and it's what gives the system its "wave-like" feel: drop water and watch fill propagate outward one hop at a time, visually coherent and fun to play with.

**Never solved the U-tube.** See §3 for the precise, now-confirmed mechanism.

### 2b. "Build B" — persistent piezometric-head relaxation field (`phead`), reverted

Replaced `head()`'s `dh` term with a new persistent field, `phead(p) = max(head(p), max(0, neighbour_max - PHEAD_DECAY))`, maintained by its own relaxation pass so pressure could (in principle) propagate further than one hop per tick. Validated in a standalone Python prototype first (vertical-column/flat-lake regression, U-tube propagation, drain-recovery all passed), then ported to GLSL.

**What broke in real in-engine testing** (not caught by the prototype or small-scale GPU tests):
- Severe mass loss at scale — total fill collapsed >99% across a few ticks in one test (`972188 → 261744 → 27711 → 5155 → 1919`). Root cause: a real, independent dispatch-range leak in `compute_s_out`/`compute_s_in` (a 2-hop read against a 1-hop dispatch guarantee) that `phead`'s much wider propagation triggered constantly. **This leak fix was kept** after the revert — it's a real bug unrelated to Build B, see `docs/BUG_LOG.md`'s "BUG (FIXED, 2026-07-08...)" entry.
- Even with that leak fixed: water visibly "boiling" (voxels launched upward, falling, sticking, re-launching, cyclically), continued real mass loss, and the actual U-tube goal still failed — water sprayed out one arm instead of equalizing.
- Working hypothesis at the time (never confirmed): `phead`'s legitimately much larger magnitude for real connected bodies (potentially thousands of fill-units, vs. `head()`'s low hundreds) overdrove the existing Accelerate/Transport tuning (`g_dt`, `gravity_bias`, `axis_cap`), which had been calibrated against `head()`'s narrow range.

Full write-up: `docs/BUG_LOG.md` → `### REVERTED (2026-07-04): "Build B" connected-vessel hydrostatics`. **Do not rebuild anything in this family** — a follow-up 2D bench (see §2c) later proved *why* it structurally cannot work, independent of the specific instability found here.

### 2c. Pressure projection (Chorin), fully ported, set aside after real in-engine testing

Designed and validated in a 2D Python bench first — quasi-static per-tick pressure projection (provisional weight-scaled gravity flux → divergence RHS with a free-surface-masked density-correction term → iterative `∇²P=RHS` solve → `F = u* − ∇P` correction → the existing conservative transport, unchanged). This is the standard fluid-sim technique (Chorin projection / the same family as MAC-grid Navier-Stokes solvers) — genuinely non-local, genuinely correct in principle, and it's what four *cheaper* falsified approaches (including a more careful version of Build B's own family) were measured against and lost to. Full bench design + all four falsified predecessors: `docs/BUG_LOG.md` → `### DESIGN (2026-07-04): connected-vessel hydrostatics — SOLVED IN A 2D BENCH via pressure projection`.

It was then **fully ported to `shaders/fluid_flux.comp`** (`BuildRHS`/`solve_red`/`solve_black`/`Correct`, red-black Gauss-Seidel since serial SOR can't run on GPU), plus a dedicated `LevelSurface` pass to fix a discovered blind spot (flat, unevenly-filled puddles have zero divergence everywhere and so read as falsely at rest — pressure projection alone can't level them). Six real bugs were found and fixed in `LevelSurface` over the course of the session, each properly bench-validated before porting — the project's own debugging discipline held up well throughout.

**It passed every bench invariant it was built against.** It also genuinely solved first-fill equalization in-engine — a real, monotonic pressure gradient measured source→middle→target across a live U-tube, arms balancing to matched mass. **But real in-engine play kept finding problems the bench suite structurally couldn't see:**
- **Never verified against requirement 2 above, and when finally tested, failed it** — pouring more water into an already-settled tube did not make the far arm react.
- Settled columns could land in physically-uneven "staircase" configurations (e.g. a column reading `51 / 255 / 19 / 255 / 7 / 255 / 255` bottom-to-top) that then sit **numerically** stable — every face's corrective flux below the `MIN_FLOW` deadzone — and never resolve, even though nothing about that configuration is at real hydrostatic rest.
- Materially worse performance and feel than the flux system it was meant to replace — before/after, the flux system ran fast enough to genuinely play with (drop things, watch waves move, tweak values, repeat); the projection system did not, which mattered as much for *being able to tell whether a fix was working* as for shipping.
- A full investigation cycle (with a fresh agent, Fable 5, given a complete handoff) was spent chasing what looked like a real `LevelSurface`-caused bug (a "one-sided mass drain") that turned out to be a **bench geometry bug** — the repro's domain was accidentally fully sealed, giving the solver no free-surface pressure anchor. A genuinely useful process win in isolation, but it illustrates the cost of the bench/engine fidelity gap below.
- **The Python benches used to validate this design were not a close enough match to the real engine**, and that mismatch cost real debugging time in both directions (chasing bench artifacts as if they were engine bugs, and — the bigger risk — the bench structurally could not have caught bugs that only exist because of *how* the real thing is built): 2D vs. the engine's real 3D (several real bugs, like a 4-neighbour flux-normalization miss, can only ever manifest with a third axis), serial row-major Gauss-Seidel vs. the GPU's actual red-black checkerboard ordering (confirmed to matter by direct comparison), and general looseness around chunk-boundary/dispatch-range behavior that only exists because the real engine is sparse and paged.

**Decision:** rather than keep iterating on a design that passed its own bench but kept finding new failure modes in play, the whole effort was preserved (nothing lost — full implementation, all fixes, the Fable 5 investigation, a from-scratch rewind/snapshot debug tool) on branch **`pressure-projection-wip`** (pushed to `origin`), and `master` was reset to the last commit before any of it existed. Full outcome write-up: `docs/BUG_LOG.md` → `### DESIGN OUTCOME (2026-07-07): pressure projection — fully ported, bench-validated, and set aside`. Deepest technical detail on the LevelSurface investigation specifically: `tools/fluid_prototypes/LEVELSURFACE_TRANSIENT_HANDOFF.md` **on that branch** (deleted from `master`, still fully intact there).

---

## 3. Why the currently-shipped flux system can't pass this test

This isn't a tuning problem — it's structural, and it was proven directly in the 2D bench before pressure projection was even designed (`docs/BUG_LOG.md`'s DESIGN entry, falsification-ladder item 1):

**A full horizontal pipe has zero head gradient anywhere along its length.** `head(p, mat)` (`shaders/fluid_flux.comp`) only ever looks *straight up* from `p`. A voxel in the middle of the horizontal connecting passage — not directly beneath either vertical arm — has rock above it, not open shaft, so `head()` there equals just that voxel's own fill: `255` once the passage is full. Every voxel along the entire middle stretch of the connector reads the exact same `255`, **regardless of how different the two arms above it are.** Since flux only ever forms from a *local* difference between immediately-adjacent cells, and every adjacent pair in that stretch reads an identical head, **no flux can ever form across the connector once it's full** — not slowly, not eventually, not ever, no matter how long the simulation runs. The cells directly beneath each arm *do* correctly read that arm's real weight (head correctly walks up into the shaft there) — but that correct information has no path to cross the flat middle to reach the other side.

This is why "pour more water into the source tube, the target tube doesn't react" was observed under the flux system too, historically — it's the expected, fully-explained behavior, not a mystery. Any fix has to give the horizontal connector genuine visibility into what each end is connected to, not just what's directly above each individual voxel.

---

## 4. What a solution actually needs to do

Distilled from everything above — a design should be evaluated against **all** of these, not a subset:

1. **Genuine non-local pressure awareness across an arbitrarily long, fully-saturated horizontal connector.** Purely local, single-hop-per-tick diffusion (§3) structurally cannot do this — proven twice now (the original flux system, and Build B's `phead`, which tried to add reach but destroyed the gradient it was trying to propagate; see the falsification ladder's item 3 for the precise reason `max`-propagation fails). Something must communicate "what's the total weight/pressure at each end of this connected body" without flattening the difference between the two ends.
2. **Reactive, not just settling.** Must correctly respond to *new* disturbances after already having reached equilibrium once — not just first-fill. This was never properly tested until the very end of the pressure-projection effort and turned out to fail; test it explicitly and early for whatever's tried next.
3. **No false-rest states.** Must not settle into a configuration that's numerically stable (every local delta below some threshold) while still being physically uneven (the "staircase" problem). If a deadzone/threshold mechanism is used anywhere (and one almost certainly needs to exist, to prevent perpetual churn — see `MIN_FLOW`'s history in `docs/BUG_LOG.md`), it needs to not be able to lock in a physically-wrong configuration as if it were correct.
4. **Fast enough to develop against.** Needs to run near the flux system's own interactive speed, or close enough that iterating on it (drop water, observe, tweak, repeat) doesn't cost minutes per cycle. This is a real design constraint, not a nice-to-have — pressure projection's Tier 1 (the only tier actually reached) wasn't fast enough, and Tier 2/3 optimizations (coarse grids, temporal amortization, graph pipe-mode — see the DESIGN entry) were never reached to find out if they'd close the gap.
5. **Ideally preserves the flux system's local, wave-like character** where it already works well (settling flat lakes, falling water, general "feel") — the ask isn't necessarily to replace that system everywhere, just to give it (or something layered with it) real connected-vessel awareness. A hybrid was floated but not explored: keep flux driving cheap, local, wave-like horizontal movement as-is, and find something cheaper than a full per-tick Poisson solve to inject just enough non-local awareness for the connected-vessel case specifically.

**If a Python bench is built for whatever's tried next, it must be a direct 1:1 port of `shaders/fluid_flux.comp` as it stands at that time** — real 3D, real red-black solver ordering if a pressure solve is involved, real chunk/dispatch-range discipline — not a from-scratch model of the design under test. See `tools/fluid_prototypes/README.md` for the specifics of how the previous benches diverged and what that cost.

---

## Reference index

- `docs/BUG_LOG.md` (on `master`, current) → `DESIGN (2026-07-04)` entry (the falsification ladder, all four dead ends, the working design), `REVERTED (2026-07-04)` entry (Build B), `DESIGN OUTCOME (2026-07-07)` entry (pressure projection's in-engine outcome).
- `ARCHITECTURE.md` (on `master`, current) §5 Current To-Do — the connected-vessel entry, kept up to date with the latest status.
- Branch `pressure-projection-wip` (pushed to `origin`) — the complete pressure-projection + `LevelSurface` implementation, untouched, including `docs/PRESSURE_PROJECTION_PORT_PLAN.md`, `tools/fluid_prototypes/FLAT_PUDDLE_HANDOFF.md`, and `tools/fluid_prototypes/LEVELSURFACE_TRANSIENT_HANDOFF.md`.
- `world.gd`'s `KEY_U` (`build_utube_test_scene()`) — the actual in-engine acceptance rig, sealed glass so it can be watched from outside. Enlarged 2026-07-07 (taller arms, more water) for more headroom to see gradients before the system settles.
