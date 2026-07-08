# Fluid Sim Python Test Suite — Connected-Vessel Hydrostatics

A **faithful 1:1 Python port** of the master-branch voxel fluid simulation, built so
the Chorin-style connected-vessel fixes can be prototyped, benchmarked, and
regression-tested in seconds instead of minutes-per-iteration inside the Godot GPU
engine.

This suite exists because the previous 2D Python benches (now deleted) diverged from
the real engine and wasted investigation time. See `../README.md` for the post-mortem.
**This port mirrors the implementation, not just the design.**

---

## 1. What it reproduces

| Engine piece | Ported here |
|---|---|
| `shaders/fluid_flux.comp` — Pass A (Accelerate) | `fluid_engine.py::FluidSim._pass_accelerate` |
| `shaders/fluid_flux.comp` — Pass B (Transport, fused conservation guard) | `fluid_engine.py::FluidSim._pass_transport` |
| `fluid_flux.comp::head()` / `eff_flux()` / `compute_s_out()` / `compute_s_in()` | `FluidSim.head / eff_flux / compute_s_out / compute_s_in` |
| `fluid_simulation.gd` — sparse flux paging, dispatch-range build, per-tick page (un)map/restore, idle-guard eviction | `FluidSim.step` + page helpers |
| `fluid_simulation.gd` — conservation check + gain tracker + dirty bitmask sleep | `FluidSim._pass_transport` return values + `step` active-set update |
| `WorldOrchestrator` voxel buffer / chunk table | `VoxelWorld` |
| `material_db.gd` material → SimulationClass table | `VoxelWorld.material_sim` |
| `world.gd::build_utube_test_scene` (KEY_U rig) | `scenes.build_utube` |

Materials present: `AIR=0, ROCK=1, SAND=2, WATER=3, GLASS=4, FIRE=5, SMOKE=6, OIL=7`.
Only `LIQUID` (water, oil) flows; SAND/ROCK/GLASS are treated as static solids (no
granular solver — out of scope per request).

---

## 2. Fidelity rules (why this bench can be trusted)

1. **3D**, same axes as the engine. `(X,Y,Z)` = `(axis 0,1,2)`; gravity is `-Y`.
2. **Exact integer math.**
   - GLSL `int(x)` truncates toward zero → Python `int(x)`.
   - GLSL `round()` tie-breaking is implementation-defined; we use round-half-away-from-zero
     (`gl_round`) so `.5` is deterministic. This only affects half-integer flux magnitudes.
3. **Same dispatch discipline.** A chunk participates only if it is in *this tick's*
   `flux_list` (active chunk + its 6 face neighbours that are loaded). `flux_page_table`
   reflects *this tick's* membership, not "ever had a page". The `page<0` guards in
   `compute_s_out`/`compute_s_in` and the Accelerate neighbour-page check are reproduced
   exactly — these are the 2026-07-02 dispatch-range leak fixes.
4. **head() etc. copied verbatim.** `head()` is the strictly local straight-up column
   scan — which is *why* the base flux system cannot cross a saturated horizontal
   connector (the U-tube failure). That limitation is reproduced on purpose.
5. **Conservative transport** is identical: `delta_f` is built from pure `eff_flux`
   reads, scaled by `s_out(donor) * s_in(receiver)`, applied once per tick from a
   changelog. Clamping at `0`/`255` is the only (documented) source of small mass
   gain/loss during violent splashing — same as the real shader.

Chunk size is a parameter. Tests default to `CHUNK_SIZE = 8` for tractable run times;
the algorithm is **identical** at `32` (the real engine default). Scenes auto-load
every chunk the rig touches so water never "falls off the edge" of unloaded space.

---

## 3. Files

```
tools/fluid_prototypes/python/
├── fluid_engine.py             # VoxelWorld + FluidSim: the faithful 3D replica + hook
├── scenes.py                   # build_utube, build_flat_lake, build_falling_block, ...
├── run_tests.py                # acceptance harness; `python run_tests.py phi` runs the fix
├── phi_flood.py                # PhiFloodFluidSim — THE FIX (2026-07-08): piezometric
│                               #   max-flood w/ EPS_H hydraulic slope + fill-deficit
│                               #   suction + pressure-wave chunk wake. Flips U-tube
│                               #   criteria #1 and #2 to PASS. See its docstring.
├── diagnose_utube.py           # root-cause dump: head/phi/flux along the connector
├── test_staircase_recovery.py  # criterion #3: gross staircase must self-correct
├── probe_staircase.py          # long-run staircase/sleep probe
├── bench_large.py              # larger rigs: cs=16 tall arms; cs=8 long connector
├── test_pile_ripple.py         # OPEN BUG regression test: a single voxel piled on a
│                               #   settled flat lake must not radiate across the whole
│                               #   lake. Currently un-passable — see phi_flood.py's
│                               #   module docstring and docs/BUG_LOG.md.
└── README.md                   # this file
```

**Status 2026-07-08:** with `PhiFloodFluidSim` the U-tube flips to PASS on both
first-fill equalization and post-settling reactivity, with no staircase above
the substrate's rest tolerance, conservation unchanged, and lake/falling-block
regressions green — see the session report / `docs/BUG_LOG.md`. Two harness
metric bugs were also fixed in `run_tests.py` (fill-units-vs-voxel-count
threshold; inverted staircase comparison). **Since ported to GLSL and
confirmed working in-editor on the real `KEY_U` rig** (K raised 16→40 for the
taller real arms) — see `docs/BUG_LOG.md`'s "PORTED & VERIFIED IN-EDITOR"
entry. **One open bug found in broader in-editor testing:** the "flat lake
ripple" (`test_pile_ripple.py`) — a fix was attempted and reverted because it
broke the U-tube mechanism itself; see that file's module docstring and
`docs/BUG_LOG.md` for the full analysis and candidate direction.

`fluid_engine.py` key API:

- `VoxelWorld(chunk_size, nc)` — chunked voxel store centered at origin.
  - `load_chunk(cp)`, `read_a(gpos)`, `write_a(gpos, u32)`, `edit_box(...)`, `edit_sphere(...)`.
  - `total_fill(mat)`, `count_voxels(mat)`.
- `FluidSim(world, FluxParams(...))` — the solver.
  - `bootstrap_active_from_world()` — mark liquid-bearing chunks active.
  - `add_active(cp)` — wake a chunk after an external edit.
  - `step()` — one tick (Accelerate → optional correction → Transport), returns #changed chunks.
  - `run(ticks)` — convenience loop.
  - `last_conservation_delta`, `last_gain`, `last_dirty_chunks` — per-tick diagnostics.
  - **`extra_correction_pass()`** — `no-op` hook. **This is where a Chorin / phi-flood /
    component-equalizer fix plugs in.** Override it in a subclass to modify `self.flux_buf`
    between Accelerate and Transport (see §6).
- `FluxParams` — `damp=0.97, g_dt=0.15, gravity_bias=3.0, settle_threshold=3,
  steps_per_tick=1` (the master defaults).

---

## 4. Running the harness

```bash
cd tools/fluid_prototypes/python
python run_tests.py
```

Pure stdlib, no external dependencies. Output (current, on the base flux replica):

```
=== U-TUBE ===
  FIRST-FILL EQUALIZATION: FAIL (expected on base flux)   (arms don't equalize)
  REACTIVITY (far arm rose): FAIL (expected on base flux)  (second pour ignored)
  STAIRCASE columns flagged: 0
  max |conservation delta| over run = 6
=== FLAT LAKE ===
  FLAT-LAKE SETTLE: PASS   (surface range <= 4 voxels, max|cons| = 0)
=== FALLING BLOCK ===
  CONSERVATION: True  (contained basin, loss within clamp tolerance)
```

### Acceptance criteria (from `docs/CONNECTED_VESSEL_HANDOFF.md` §4)

| # | Criterion | Base flux result |
|---|---|---|
| 1 | First-fill equalization (per-arm mass + level) | **FAIL** — connector blind spot |
| 2 | Reactivity after settling (second pour) | **FAIL** — far arm doesn't react |
| 3 | No false-rest / staircase | PASS (0 flagged) |
| 4 | Mass conservation (delta ~ 0) | PASS (clamp loss only) |
| 5 | Preserves local wave/feel | PASS (lake settles, block falls) |

The U-tube **FAIL is expected and is the proof of fidelity** — it reproduces master's
structurally-incapable behaviour. A correct Chorin fix must flip #1 and #2 to PASS
*without* regressing #3/#4/#5.

---

## 5. Tests reference

`run_tests.py` defines helper metrics (`arm_mass`, `arm_surface`, `staircase_count`)
and three scene tests:

- **U-tube** (`scenes.build_utube`): sealed glass U (`arm_half_xz=1`, `arm_h=chunk_size`,
  `gap=chunk_size//2`, `wall=2`). Water poured into arm A; after `ticks1` ticks it checks
  equalization; then more water is added to arm A and after `ticks2` ticks it checks the
  far arm rose (reactivity).
- **Flat lake** (`scenes.build_flat_lake`): rock basin + water slab; checks it settles
  flat and conserves.
- **Falling block** (`scenes.build_falling_block`): water block dropped inside a rock
  basin; checks it falls, stays contained, and conserves (allowing documented clamp loss).

All scenes auto-load the chunks they touch so nothing escapes the simulated region.

---

## 6. How to plug in a Chorin / hydrostatic fix

The port is intentionally a clean base. A fix subclass only needs to inject a
non-local pressure signal into `flux_buf` (the same buffer Accelerate wrote) before
Transport reads it. Minimal skeleton:

```python
from fluid_engine import FluidSim, FluidParams

class ChorinFluidSim(FluidSim):
    def extra_correction_pass(self):
        # 1. Build RHS from provisional flux divergence (per liquid voxel).
        # 2. Solve pressure (red-black Gauss-Seidel / phi-flood / component equalizer).
        # 3. Add the pressure gradient as an extra accel term into self.flux_buf
        #    lanes, OR write a directed correction flux along the connector path.
        #    Keep the SAME compute_s_out/compute_s_in conservation guards.
        ...
```

The harness already reports the exact numbers the fix must change, so validation is:
run `run_tests.py` with the subclass swapped in and watch U-tube #1/#2 flip to PASS
while #3/#4/#5 stay green.

Candidate approaches and their ranking live in `docs/POSSIBLE_SOLUTIONS.md`
(phi-flood ranked #1, settled-component equalizer #2, surface-height field #3,
amortized projection #4, …). The pressure-projection design itself is preserved,
untouched, on branch `pressure-projection-wip` (full shader + `LevelSurface` pass +
bench history) — revive-and-fix is a legitimate path (ranked #4/#7).

---

## 7. Known limitations of this bench

- **Performance:** pure Python, not GPU. Use `CHUNK_SIZE=8` and modest tick counts
  (defaults: U-tube 200+120, lake 250, falling 150) — a full run is ~25–30 s. Larger
  rigs (chunk_size 32, full KEY_U dimensions) are *correct* but slower; validate
  algorithm there before trusting only the small rig.
- **No granular solver.** SAND is a static solid; fluid↔granular coupling is not modelled.
- **Rounding tie-break** is round-half-away-from-zero, not the engine's
  implementation-defined choice. Differences only appear at exactly-half-integer flux
  magnitudes and are negligible for acceptance.
- **No rendering.** This is a numeric/diagnostic harness, not a visual one. The in-engine
  `KEY_U` rig (`world.gd`) remains the visual acceptance test.

---

## 8. Quick-start for a fix implementer

1. Read `docs/CONNECTED_VESSEL_HANDOFF.md` (the problem + failed attempts) and
   `docs/POSSIBLE_SOLUTIONS.md` (the ranked options).
2. Run `python run_tests.py` to see the baseline (U-tube FAIL is expected).
3. Subclass `FluidSim`, implement `extra_correction_pass`, and wire it into the three
   scene tests in `run_tests.py`.
4. Iterate until U-tube #1 (equalization) and #2 (reactivity) PASS with conservation
   and no staircase.
5. Sanity-check against a larger rig before declaring done.
