# Fluid prototypes — connected-vessel hydrostatics bench

Standalone 2D Python emulators used to design and falsify approaches to the U-tube /
connected-vessel hydrostatics problem **before** touching `shaders/fluid_flux.comp`.
They faithfully mirror the engine's owned-face flux model and its conservative
`s_out`/`s_in` transport, plus a strict invariant suite (per-arm mass balance, full
columns, exact conservation, pipe-length independence, thin-film drain, no-churn).

Run any file directly: `python utube_projection.py` → prints `ALL TESTS PASSED`.

| File | What it is | Running it |
|---|---|---|
| `utube_projection.py` | **The validated winner.** Chorin pressure projection (provisional weight-scaled gravity flux → divergence RHS + surface-masked density correction → SOR solve → `−∇P` correction → conservative transport). Passes the full invariant suite. This is the design being ported. | prints `ALL TESTS PASSED` |
| `utube_pressure.py` | **Falsified (expected to fail equalization):** `P` replaces `head()` via a max-with-cost relaxation field. Equilibrium-correct but dynamically dead — `max`-propagation makes pressure uniform-at-elevation, flattening the lateral gradient, so arms freeze lopsided. Same failure class as "Build B". | prints `FALSIFIED AS DOCUMENTED` (a self-demonstration, not a pass) |
| `utube_emulator.py` | **Falsified:** narrow static-lift, then deficit-toward-surface-elevation. Can't route/climb; smears thin. | raises `AssertionError: THIN SMEAR` (the intended demonstration) |

> **⚠️ Metric pitfall — do not judge equalization by `column_top`.** A lopsided *thin smear* reaches the same column HEIGHT in both arms at very different DENSITIES, so `column_top` (and "median submerged fill over both arms") reports "equalized" while the arms hold very different amounts of water. The only trustworthy equalization metric is **per-arm mass** (`sum(fill) / FILL_CAP` per arm column). `utube_pressure.py` originally shipped with `column_top`-based tests that printed a **false** "ALL TESTS PASSED" and misled a reviewer into thinking max-relaxation worked; the strict per-arm-mass check shows imbalance ~2 (a broken smear). Any future port must judge the U-tube by per-arm mass + a fill-profile dump, never by surface height alone.

**Design rationale, the full falsification ladder, and the GPU-port plan:**
- `docs/BUG_LOG.md` → *"DESIGN (2026-07-04): connected-vessel hydrostatics — SOLVED IN A 2D BENCH via pressure projection"*
- `docs/PRESSURE_PROJECTION_PORT_PLAN.md` → staged implementation plan

Keep these as a regression rig: the 3D / red-black-Gauss-Seidel port must reproduce
these same invariants. Note the bench uses **serial SOR**, which cannot run on a GPU —
the port must use red-black GS or Jacobi and re-measure iteration counts.
