# LevelSurface One-Sided Drain — Findings & Handoff

> **RESOLVED 2026-07-07 (Fable 5, second read): NOT a LevelSurface bug — the repro bench's domain was accidentally sealed (zero air cells), giving the pressure solve no free-surface anchor.** `build_drop_on_puddle()` carved `AIR` at `y=1` then overwrote those exact cells with `LIQUID`; everything above stayed `SOLID`. Pure-Neumann solve + incompatible STIFF source → warm-started `P` drifted unboundedly (−2078 → −3150 over ticks 3–10, still falling), and fix-6's pair damping consumed those meaningless gradients. The one-sided *direction* was seeded by the tower sitting at `W//2=15` vs. the 28-cell puddle's true center of 14.5 (a centered tower in an odd-width domain stays mirror-exact for 700+ ticks; the drain side follows the tower's offset — §4's "boundary effect" hypothesis was the right one). With a real air headspace, v3/v3-RB spread the tower **exactly symmetrically** (`mirror_err=0.0`, `max|P|=0.0`, exact conservation) and stall at the known fix-5 quantization floor (~10-fill/cell pyramid slope). The per-pair audit confirmed every pair delta cancels exactly at every instant — fix 5 holds; no lopsided transfer ever occurs. **No GLSL change made or needed for this symptom.** Bench fixed (air headspace + free-surface assert), regression `open_drop_symmetry` added to `level_surface_v3_check.py`. Full write-up: `docs/BUG_LOG.md` (2026-07-07 entry), which also records (a) the residual design note that fix-6 damping is undefined-ish inside genuinely sealed pockets (vacuum-lock regions), and (b) that the original in-engine "ring at fill 1" symptom is now unconfirmed and needs an in-engine retest — the LevelSurface-free open baseline reproduces a full core + fill-1..3 ring + untouched puddle, strikingly close to the in-engine description. The sections below are preserved as-written for history.

**Audience:** whoever picks up the investigation next (this handoff is written for a fresh agent — Fable 5 — to get a second read on the mechanism).
**Scope:** a bug found *after* `LevelSurface` (Checkpoint 6, the flat-puddle leveling pass — see `FLAT_PUDDLE_HANDOFF.md` for its original design) shipped to GLSL with six bug fixes already applied this session. In-engine testing of the sixth fix (pressure-difference damping) surfaced a **new, confirmed, one-sided mass-consolidation bug** that survives every attempt to explain it away as a bench artifact. Root cause is still unknown. All claims below are backed by runnable benches in this folder.

Related files:
- `shaders/fluid_flux.comp` — the shipped GLSL, `params.pass == 6` block (search `LevelSurface`). Currently has all 6 fixes from this session; unchanged since the bug below was found (no GLSL edit has been attempted yet — bench-first discipline, see `CLAUDE.md`).
- `simulations/fluid_simulation.gd` — GDScript wiring (`proj_LEVEL_RATE`, `proj_LEVEL_MIN`, `proj_LEVEL_PRESSURE_MIN`, `use_surface_leveling`).
- `docs/BUG_LOG.md` — full history of all 6 LevelSurface fixes this session, each with symptom/root-cause/fix/rule-to-remember.
- `docs/PRESSURE_PROJECTION_PORT_PLAN.md` §9 — Checkpoint 6's living design doc.
- `tools/fluid_prototypes/utube_projection.py` — the validated base projection sim (`Sim`, serial row-major Gauss-Seidel pressure solve).
- `tools/fluid_prototypes/utube_projection_redblack.py` — pre-existing red-black-ordering variant (`SimRB`), from the Checkpoint 1 GPU-parity bench.
- `tools/fluid_prototypes/level_surface_v3.py` — current `LevelSurface` design (`SimLevelSurfaceV3`): per-pair pressure-difference damping, mirrors the shipped GLSL exactly.
- `tools/fluid_prototypes/level_surface_v3_check.py` — v3's own regression suite (buried-seam, genuine-height-difference, full legacy suite). **All passing.**
- `tools/fluid_prototypes/level_surface_v3_redblack.py` — **new this handoff**, combines v3's `LevelSurface` logic with `SimRB`'s red-black pressure solve.
- `tools/fluid_prototypes/drop_settle_transient.py` — **new this handoff**, the actual repro bench.

---

## 1. Context — what shipped and why it should have been safe

`LevelSurface` is a dedicated GLSL pass (`fluid_flux.comp`, `pass == 6`) that does small, conservative, lateral fill exchanges between adjacent surface-layer cells, bypassing the pressure solve entirely (the pressure solve structurally can't see flat-layer unevenness — see `FLAT_PUDDLE_HANDOFF.md` §2). Six bugs were found and fixed in it this session, in order:

1. **`LEVEL_DEGREE=4.0` normalization** — unnormalized summing let an exposed cell lose up to 4× the intended rate per tick.
2. **Classify-by-what's-actually-there** — a dry (air) cell asking about a liquid neighbour must not gate on its own (air) material.
3. **`SUBMERGED_MIN` threshold** — "submerged" means fill-above *exceeds* a threshold, not merely nonzero.
4. **Continuous damping instead of a boolean gate** — a boolean re-evaluated on noisy input flickered a cell's whole contribution on/off every tick; replaced with a continuous `[0,1]` factor.
5. **Per-pair integer rounding** (`roundEven`, not aggregate-then-round) — fixed a real, structural conservation leak that appeared once a cell had 2+ active neighbours.
6. **Pressure-difference damping** (`clamp(1 - |P_self - P_neighbour| / PRESSURE_MIN, 0, 1)`), replacing the vertical-fill-based "submerged" proxy — fixed a design gap where a fully-submerged cell with genuinely open lateral neighbours never spread to them, because the old single per-cell damping value vetoed *all* neighbours at once instead of judging each pair independently.

Each fix was validated in a Python bench mirroring the GLSL exactly (`level_surface_v3.py`), then ported. Full write-ups: `docs/BUG_LOG.md`.

After fix 6 shipped, in-engine retesting showed the originally-reported symptoms resolved (submerged cell now spreads correctly, no more flicker). But a **new symptom** appeared: dropping water from height onto an existing puddle, the pool "dries up except for a ring around the landing location, the ring is at fill 1... and the stream of water never ends" — i.e. mass isn't spreading into the puddle at all, it's consolidating somewhere and never redistributing outward.

---

## 2. Repro bench — `drop_settle_transient.py`

`build_drop_on_puddle(sim_cls, W=30, H=20)`: a flat puddle 28 cells wide at `fill=60` (well below full, deliberately uneven-capable), with a 6-tall fully-full column (`fill=255` × 6) dropped onto its horizontal center — mirrors "drop water from height onto an existing puddle."

Run: `python drop_settle_transient.py`

### 2a. Baseline (`utube_projection.Sim`, no `LevelSurface` at all)

The tower's mass stays almost entirely parked in its own column (`fill≈254`), the rest of the puddle stays essentially flat and untouched (`~52–53` everywhere). This is the *expected*, already-documented flat-puddle blind spot (`FLAT_PUDDLE_HANDOFF.md` §1) — the whole reason `LevelSurface` exists. Not a bug.

```
baseline bottom row BEFORE: [60]*28
baseline bottom row @700:   [36, 52, 53, 52, 53, ..., 52, 254, 52, 53, ..., 53, 52, 53, 77]
```

### 2b. With `LevelSurface` (`level_surface_v3.SimLevelSurfaceV3`, current shipped design)

Liquid-cell count drops from 34 to 23 over ~370 ticks and **never recovers** within 3000 ticks. The bottom-row fill profile is not "spread out" — it's a **monotonic one-sided ramp**:

```
RB bottom row @700: [20, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, 242, 252, 208, 177, 148, 117, 88, 66, 56, 46, 36, 26, 24, 34, 25]
                      ^--- drains to ~zero ---^  ^-- tower --^  ^------- decreasing ramp, one direction only -------^
```

Conservation is **exact** throughout (`cons_drift = 0.000` in every run) — this is a redistribution bug, not a mass-loss bug. Mass is being funneled entirely to one side instead of spreading symmetrically outward from the drop point.

---

## 3. Ruled out: bench solver-ordering artifact

`utube_projection.Sim.solve_pressure()` uses a **serial, row-major** Gauss-Seidel sweep (`for y: for x:`), which has a small inherent directional bias — visible even in the LevelSurface-free baseline above (`36` on the left puddle edge vs `77` on the right, a ~2× asymmetry, though tiny compared to what follows). The real GLSL engine instead uses **red-black checkerboard** passes (`solve_red`/`solve_black`), which cancel this kind of bias by construction. So before trusting the bench, we swapped in the already-existing red-black solver (`utube_projection_redblack.SimRB`, originally built for the Checkpoint 1 GPU-parity bench) and re-ran the exact same scenario.

`level_surface_v3_redblack.py` combines `SimLevelSurfaceV3`'s leveling logic with `SimRB`'s red-black `solve_pressure()`. (Note for whoever reads this: plain multiple inheritance from both classes silently picked the *wrong* `solve_pressure` — `level_surface_v3.py` and `utube_projection_redblack.py` each do their own independent `importlib` load of `utube_projection.py`, so their respective `Sim` base classes are *different objects* despite identical source, which breaks C3 MRO precedence. Fixed by single-inheriting from `SimRB` and borrowing only the leaf methods that don't call `super()`.)

First, sanity-checked the swap didn't change anything about `LevelSurface`'s own correctness — reran `level_surface_v3_check.py`'s legacy regression cases through it:

```
[flat_uneven_puddle/RB] spread=9.0 cons=0.000   (identical to serial-solve v3)
[even_lake_no_churn/RB] max_move=0
[gap_merge/RB]          gap_fill=76 cons=0.000
```

Then reran the drop-on-puddle scenario:

```
RB bottom row @700: [20, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, 242, 252, 208, 177, 148, 117, 88, 66, 56, 46, 36, 26, 24, 34, 25]
```

**Essentially identical to the row-major result.** This rules out the row-major-sweep-artifact theory: the one-sided drain is very likely a genuine bug in `LevelSurface`'s own logic (or its interaction with the still-settling vertical solve), not a bench-fidelity gap.

---

## 4. What's confirmed vs. still unknown

**Confirmed:**
- The asymmetry is real, reproducible, and solver-ordering-independent.
- Conservation is exact — no mass is created or destroyed, only misallocated.
- `LevelSurface`'s own per-pair mechanics are symmetric *by construction* (fix 5's per-pair `roundEven` rounding guarantees exact pairwise cancellation — `level_surface_v3_check.py`'s exact-conservation tests already prove this holds for static/simple scenarios).
- The bug only manifests in this "settling tower" scenario — every previously-validated static scenario (flat uneven puddle, buried seam, genuine height difference, even lake, gap merge, U-tube) still passes exactly.

**Not yet known — the actual root cause:**
- *Why* a per-pair-symmetric mechanism produces a one-sided steady state when coupled with a still-settling vertical solve.
- Leading (**unconfirmed**) hypothesis: `LevelSurface`'s pressure-difference damping (fix 6) is evaluated fresh each tick against `P`, which is itself still evolving asymmetrically while the tower drains (the `STIFF` density-correction term's divergence source is inherently directional while the stack above any given cell is uneven — see `utube_projection.py` line ~82-83). If `LevelSurface` "locks in" whatever partial asymmetry exists in `P` at each instant rather than converging around a stable target, repeated ticks could compound a small transient bias into a large permanent one. **This has not been traced or proven** — it's the most promising thread pulled so far, not a conclusion.
- Alternative hypotheses not yet ruled out:
  - A boundary effect specific to how `build_drop_on_puddle` positions the tower relative to the puddle edges (untested: does the ramp's direction/magnitude change if the tower is placed asymmetrically, or does it always ramp the same direction regardless of position — the latter would point at an algorithmic bug rather than a boundary artifact).
  - An interaction between `LevelSurface`'s classify/donor-mat logic (dry→liquid materialization) and the order cells transition from `LEVEL_NB_DRY` to `LEVEL_NB_SURFACE` as the wavefront advances — since a newly-materialized cell only starts participating in leveling *the following tick*, there could be an asymmetric one-tick lag effect that compounds directionally.
  - Something in how `PRESSURE_MIN`'s per-pair damping interacts with `LEVEL_MIN`'s per-pair transfer-floor threshold (fix 5's known accepted quantization floor — see `docs/BUG_LOG.md`) during a transient where `P` differences are themselves small and noisy.

---

## 5. What we're asking for

A fresh pass at root-causing *why* the one-sided drain happens — ideally by instrumenting `level_surface_v3.py`'s `level_surface()` to dump per-pair `diff`/`p_self`/`p_nb`/`pair_damping`/`pair_delta` for a few ticks around the transition zone (roughly ticks 5–30, where `count traj` in `drop_settle_transient.py` shows the collapse actually happening: `[34, 34, 34, 24, 23, ...]`), to see whether the *per-pair* deltas are actually symmetric at each instant (as designed) and the asymmetry only emerges from an accumulation effect across many ticks, or whether there's an instant where a specific pair computes a genuinely lopsided transfer.

Reproduce with: `python drop_settle_transient.py` (baseline vs. v3 comparison), then `python level_surface_v3_redblack.py` (sanity check) and manually re-run the drop scenario against `SimLevelSurfaceV3RB` (see §3's snippet) to confirm on red-black before/while investigating.

**Explicitly out of scope for this handoff** (lower priority, deferred by the user): a separate, much smaller bug — a permanent, exact ±1-unit flux oscillation observed in-engine between two specific voxels straddling a chunk boundary (identical `fill lost: 1` / `fill gained: 1` repeating forever, never damping to zero). Believed to be in `BuildRHS`/`Correct`'s flux rounding rather than `LevelSurface`, and not yet investigated. Mentioned here only so it isn't rediscovered and conflated with the bug above — they are almost certainly unrelated mechanisms.

**Per this project's standing discipline (`CLAUDE.md`):** do not attempt a GLSL fix without validating the mechanism and the fix in a Python bench first. If you find the root cause, add a regression test to `level_surface_v3_check.py` (or a new file) that fails on the old behavior and passes on the fix, mirroring the pattern already used for fixes 1–6.
