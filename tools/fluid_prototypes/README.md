# Fluid prototypes — connected-vessel hydrostatics bench (reset 2026-07-07)

**The previous 2D Python emulators in this folder were deleted.** They were used to design
and validate the Chorin pressure-projection approach to the U-tube / connected-vessel
problem, which was then ported to `shaders/fluid_flux.comp` — but the in-engine result
under-delivered badly enough (didn't react to water added after initial settling,
staircases that never level out, worse feel/performance than the flux system it replaced)
that the whole effort was set aside. See branch `pressure-projection-wip` for the full
implementation, its docs, and everything that was learned — nothing was lost, just parked.

**The known root cause of the mismatch (read this before writing a new bench):** the 2D
Python benches were a *conceptual* model of the design (owned-face flux, conservative
`s_out`/`s_in` transport), not a faithful mirror of the actual GLSL. They differed from the
real engine in at least: dimensionality (2D vs. 3D — several real bugs, like the
`LEVEL_DEGREE` 4-neighbor normalization miss, could only ever manifest with a 3rd axis),
solver ordering (serial row-major Gauss-Seidel vs. the GPU's red-black checkerboard —
confirmed to matter, see `LEVELSURFACE_TRANSIENT_HANDOFF.md` on the WIP branch), gravity
axis convention (top-down vs. bottom-up), and general handwaviness around chunk-boundary
and dispatch-range behavior that only exists because the real engine is sparse/paged. Each
mismatch cost real debugging time chasing a "bug" that was actually a bench artifact, or
missing a real bug the bench structurally couldn't expose.

**If a Python bench is built again, it must be a direct 1:1 port of
`shaders/fluid_flux.comp` as it stands at that time** — same dimensionality (3D), same
solver ordering (red-black, not serial), same axis conventions, same chunk/dispatch-range
discipline — not a from-scratch reimplementation of the *design*. A bench that models the
idea but not the implementation will eventually diverge from what's actually shipping, and
by construction can't catch the class of bug that only exists because of *how* the real
thing is built (parallel ordering, 3D neighbor counts, sparse paging), only bugs in the
underlying algorithm.

**Still useful, in-engine, independent of which fluid system is used:**
`world.gd`'s `KEY_U` → `build_utube_test_scene()` — builds a sealed glass U-tube rig
(two vertical shafts + a connecting passage) with a known, exact volume of water poured
into one arm, pauses immediately so the scene can be inspected before running. This is the
actual acceptance test, not a Python stand-in for it — use it directly.
