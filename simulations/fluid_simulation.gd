class_name FluidSimulation extends VoxelSimulation

# GPU dispatcher for liquid simulation (water, oil) — Phase 7e persistent-flux pipeline.
# Two passes per sub-step, both in fluid_flux.comp: Accelerate -> Transport. (Was four
# passes -- Accelerate -> Scale-Outflow -> Scale-Inflow -> Transport -- until Phase 7e
# Step 9, 2026-07-06: Transport now recomputes each cell's conservation-scale factor
# inline as a pure function of already-synced flux_buf/voxels_a data instead of reading it
# back from a buffer a prior pass wrote, so the two Scale passes and their rewritten-lane
# race-avoidance proof are gone entirely. See fluid_flux.comp's header comment.)
# No CAS, no voxels_b scratch buffer — Transport writes the changelog directly, same
# single-writer discipline as every other simulation.
#
# Phase 7e Step 7 (2026-07-01): the legacy vertical-CAS shader (fluid_sim.comp) has been
# deleted. It was dispatched every sub-step BEFORE Pass A/Scale/Transport and its
# try_compress still contained a destructive "snap remainder below 32 to zero" hack from
# before this redesign existed (a patch for a problem Scale's conservation guarantee now
# solves properly). With both systems running, vertical CAS was silently eating fill every
# time it compressed a low-fill cell created by Transport — confirmed via step_once() +
# debug_dump_fluid_stats(): a 255-fill droplet decayed 255->255->220->220->165->102->51->1
# over 7 ticks, losses correlating exactly with the growing count of small-fill voxels
# Transport was creating. Y (vertical) flow is now handled entirely by fluid_flux.comp's
# Y-axis flux (head-driven, same mechanism as X/Z) — see ARCHITECTURE.md Step 7.

var _compute
var _flux_pipeline: Dictionary = {}   # fluid_flux.comp — Accelerate / Scale / Transport
var _flux_params_buf: RID             # 52-byte params for fluid_flux.comp (incl. pass, damp, g_dt, gravity_bias, settle_threshold)
var _flux_chunk_list_buf: RID         # active + 6 face neighbours — flux dispatch range
var _flux_chunk_list_capacity: int = 0
var _dirty_chunks_buf: RID
# Phase 7e Step 1 — privately owned by FluidSimulation (was WorldOrchestrator._flux_buf).
# Nothing outside fluid ever reads flux, so it never belongs on the Orchestrator's
# shared buffer table.
#
# Phase 7e Step 2 — sparse paging. _flux_buf is no longer sized to the full voxel-slot
# capacity; it holds one "page" (vpc entries) per chunk that has ever been a fluid
# candidate, allocated from a freelist and indexed indirectly through _flux_chunk_table.
# A chunk with no page is a zero-flux boundary, not an error — most loaded chunks never
# touch fluid and should cost zero flux storage.
var _flux_buf: RID
var _flux_page_capacity: int = 0        # total pages currently backing _flux_buf
var _flux_freelist: Array = []          # free page indices (int), pop_back() to allocate
var _flux_chunk_table: Dictionary = {}  # chunk_pos → page index (CPU bookkeeping)

# Pressure-projection port (2026-07-04, Checkpoint 2 — see docs/PRESSURE_PROJECTION_PORT_PLAN.md).
# _pressure_buf/_div_buf ride the EXACT SAME page allocation as _flux_buf (same
# _flux_chunk_table/_flux_freelist/_flux_page_table_buf) — the pressure domain is a
# subset of the flux-active region, so no separate paging subsystem is needed. 1
# float32/voxel each (vs. _flux_buf's 3 lanes/voxel), grown/zeroed in lockstep with it
# in _ensure_flux_buf_capacity()/_allocate_flux_page()/_free_flux_page(). Gated entirely
# behind use_projection — when false, these are allocated but never read or written by
# any shader pass, so they cannot affect the current (Accelerate → Transport) behavior.
var _pressure_buf: RID
var _div_buf: RID

# Feature flag for the pressure-projection connected-vessel hydrostatics port. Default
# false: current baseline (Accelerate → Transport, no U-tube support) stays exactly as
# shipped. When true, step() dispatches BuildRHS → (SolveRed → SolveBlack)×N_ITER →
# Correct → Transport instead. See docs/PRESSURE_PROJECTION_PORT_PLAN.md.
@export var use_projection: bool = false

# Pressure-solve tuning. G/STIFF still starting points from Checkpoint 1's red-black-GS
# bench re-measurement (tools/fluid_prototypes/utube_projection_redblack.py), NOT the
# original bench's serial SOR values (G=64/STIFF=0.5/OMEGA=1.7/N_ITER~60), which don't
# transfer directly. OMEGA/N_ITER updated 2026-07-04 after in-editor testing found
# OMEGA=1.6 causes a genuine, non-decaying low-level oscillation at settle (small but
# real fill jitter every tick, never landing on a silent "nothing changed" tick) --
# over-relaxation intentionally overshoots each iteration for faster convergence, and it
# was overshooting into a stable limit cycle instead of settling. OMEGA=1.0 (plain
# Gauss-Seidel, no overshoot) confirmed genuinely silent settling (identical output
# repeated 60+ ticks in a row) at N_ITER=24. Re-tune further against the acceptance scene
# (Checkpoint 6) if convergence speed becomes a problem -- prefer raising N_ITER over
# raising OMEGA back up, given what this cost to find.
@export var proj_G: float = 64.0        # provisional gravity flux magnitude per open vertical face
@export var proj_STIFF: float = 0.5     # density-correction strength (submerged, under-full cells)
@export var proj_OMEGA: float = 1.0     # over-relaxation factor for red-black GS (1.0 = plain GS)
# Diagnostic toggle (2026-07-04) — M1's design is quasi-static (pressure_buf zeroed fresh
# every tick, see the port plan). Default false = that design, unchanged. When true, skip
# the per-tick zero and let solve_red/solve_black keep refining last tick's pressure_buf
# instead of re-solving from scratch -- testing whether the low-level fill jitter/
# checkerboard artifact seen at settle is a convergence-consistency artifact (a fresh
# solve landing in a slightly different place near the rest point each tick) rather than a
# deeper issue with the STIFF density-correction term itself. This is exactly what M2
# (temporal amortization) would do anyway; testing it early here is diagnostic only.
@export var proj_warm_start: bool = false
@export_range(1, 200, 1) var proj_N_ITER: int = 24  # red-black GS sweeps per tick (2 dispatches each)

# Checkpoint 6 (2026-07-05, FLAT_PUDDLE_HANDOFF.md section 5b) — surface height-leveling
# pass. BuildRHS's divergence only ever comes from VERTICAL stacking, so a single flat
# uneven layer has div==0 everywhere and the pressure solve can never move it (P is pinned
# to ~0 by the free-surface Dirichlet BC at every cell). This pass sidesteps the solve
# entirely with a small, conservative, lateral fill exchange applied directly to the
# surface layer, dispatched after Transport. Bench-validated in
# tools/fluid_prototypes/height_level_prototype.py; defaults match that bench exactly.
# Gated separately from use_projection since it's a standalone fix for a projection-only
# gap — default false until verified in-engine (Checkpoint 6 acceptance gates).
@export var use_surface_leveling: bool = false
@export var proj_LEVEL_RATE: float = 0.20  # fraction of surface-elevation diff moved per tick
@export var proj_LEVEL_MIN: float = 1.0    # stop below this |diff| -> clean termination, no churn
# BUG (found 2026-07-06, in-engine): "submerged" originally meant ANY nonzero same-liquid
# fill above (matching BuildRHS's own mask), so a thin residual film (e.g. fill=1) left on
# top of an otherwise-full reservoir permanently locked it out of LevelSurface, even with
# room to spread laterally -- confirmed reproducible in the bench (thin_film_check.py) and
# fixed there: only treat a cell as submerged if the fill directly above exceeds a threshold.
# SUPERSEDED (2026-07-06, see docs/BUG_LOG.md): the vertical-fill "submerged" proxy itself
# turned out wrong whenever a buried low-fill seam sits inside an otherwise-uniformly-stacked
# blob -- confirmed in-engine (a fully-submerged 255-fill cell with genuinely open ~11-fill
# lateral neighbours never spread to them at all). Replaced with a PER-PAIR pressure-
# difference test: |P_self - P_neighbour| beyond this threshold means the pressure solve
# already owns this pair's redistribution (defer); near-zero means the projection has no
# signal there (act), regardless of raw vertical fill. Bench-validated in
# tools/fluid_prototypes/level_surface_v3.py / level_surface_v3_check.py. 2.0 chosen the
# same way as before -- well below a genuine column-height gradient (~3.5 in the bench) and
# above the buried-seam blind-spot case's converged residual (~0.3).
@export var proj_LEVEL_PRESSURE_MIN: float = 2.0
# Idle-guard for pages allocated purely as flux-dispatch neighbours (chunk-boundary fix,
# 2026-07-02) that never actually became active — chunk_pos → idle ticks. Separate from
# _active_chunks's own idle tracking, which only covers chunks that were simulating.
var _flux_page_idle: Dictionary = {}
# GPU-side mirror of _flux_chunk_table — a shader can't read a GDScript Dictionary.
# Dense int array, indexed exactly like WorldOrchestrator._chunk_table (grid-flat index),
# value = page index or -1 (no page / zero-flux boundary). Sized to grid_span, which is
# fixed after WorldOrchestrator._init_grid() — unlike _flux_buf, this never needs to grow.
var _flux_page_table_buf: RID
const FLUX_PAGE_INITIAL_CAPACITY := 64

# Automatic conservation check (2026-07-02) — one signed int, atomicAdd'd by Transport
# with every real (new_fill - old_fill) it applies. Zero after a tick means mass
# conserved exactly; nonzero is a direct, quantified leak/gain, no manual F6 diffing
# needed. See step()'s Transport block for the read/reset side.
var _conservation_buf: RID
# Location tracker (2026-07-02) — the stale-page fix didn't resolve the leak, so this
# pinpoints WHERE the single largest gain each tick happens instead of guessing further.
# 8 bytes: [flat_idx: s32, amount: s32]. Plain (non-atomic) write on the shader side.
var _gain_tracker_buf: RID
var _orchestrator: WorldOrchestrator
var _chunk_manager: ChunkManager
var _active_chunks: Dictionary = {}
var _frame: int = 0

# Voxel watch (2026-07-02) — GainTracker only ever shows the CURRENT tick's single
# worst offender, which can jump between correlated cells and never reveals one cell's
# actual trajectory over time. This watches one fixed (chunk, local) voxel every
# sub-step and prints its fill + all 3 flux lanes, to see oscillation shape directly
# instead of inferring it from aggregate conservation numbers. Exported (not a function
# call) so it can be toggled/edited live from the remote Inspector while playing —
# no debugger console access needed. Defaults to the recurring offender found in the
# resting-hole-water leak (chunk (0,-3,-1) local (1,28,30)); zero extra GPU work when
# watch_enabled is false.
@export var watch_enabled: bool = false
@export var watch_chunk: Vector3i = Vector3i(0, -3, -1)
@export var watch_local: Vector3i = Vector3i(1, 28, 30)

# Automatic conservation checker toggle (default false, added 2026-07-03) — gates a
# GPU readback that forces a sync every sub-step; see the in-step() comment for why this
# was left on by default during the leak investigation and needs to be off otherwise.
@export var debug_conservation_check: bool = false

# Full per-voxel diagnostic dump (fill, own 3 owned lanes, 3 incoming neighbour-owned
# lanes) — factored out (2026-07-03) so both the fixed watch target AND the
# conservation checker's per-tick "biggest gain" location can share it. Watching a
# fixed cell only helps if the leak happens to recur there; the gain-tracker's flagged
# location moves every tick (confirmed 2026-07-03: manually retargeting watch_chunk/
# watch_local after the fact kept missing the actual leaking cell, since the location
# reported one tick was rarely still the worst offender when the user could react to
# it), so _debug_print_conservation() below calls this directly on the CURRENT tick's
# flagged voxel instead of relying on the user to catch up to a moving target.
func _debug_dump_voxel(label: String, chunk_pos: Vector3i, local: Vector3i) -> void:
	var slot: int = _orchestrator._chunk_to_slot.get(chunk_pos, -1)
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var lx := local.x
	var ly := local.y
	var lz := local.z
	var fill_str := "n/a"
	var mat_str := "n/a"
	if slot >= 0:
		var byte_idx := (slot * vpc + lz * cs * cs + ly * cs + lx) * Chunk.VOXEL_STRIDE
		var vdata: PackedByteArray = _compute.download_range(_orchestrator.get_voxel_buf(), byte_idx, 4)
		mat_str = str(vdata.decode_u8(0))
		fill_str = str(vdata.decode_u8(1))
	var page: int = _flux_chunk_table.get(chunk_pos, -1)
	var flux_str := "n/a (no page)"
	if page >= 0:
		var local_idx := lz * cs * cs + ly * cs + lx
		var lane_base := (page * vpc * 3 + local_idx * 3) * 4
		var fdata: PackedByteArray = _compute.download_range(_flux_buf, lane_base, 12)
		flux_str = "(fx=%d, fy=%d, fz=%d)" % [fdata.decode_s32(0), fdata.decode_s32(4), fdata.decode_s32(8)]

	# Incoming neighbour-owned lanes (2026-07-03) — the voxel's OWN 3 owned lanes above
	# only tell half the story: Transport's delta also reads the -axis neighbour's owned
	# face pointing INTO this voxel (delta += eff_flux(flux_buf[nlane])). A voxel whose
	# own 3 lanes read all-zero can still have its fill change every tick if a neighbour's
	# owned face is nonzero — exactly the pattern that motivated this addition (found
	# 2026-07-03: a watched voxel's fill oscillated 236<->240 forever while fx/fy/fz all
	# printed 0, meaning the real driver was invisible until now).
	var world_pos := chunk_pos * cs + local
	var axis_dir := [Vector3i(1, 0, 0), Vector3i(0, 1, 0), Vector3i(0, 0, 1)]
	var axis_name := ["X", "Y", "Z"]
	var incoming_str := ""
	for axis in range(3):
		var nb_world: Vector3i = world_pos - axis_dir[axis]
		var nb_chunk := _chunk_manager.voxel_to_chunk_position(nb_world)
		var nb_local := _chunk_manager.global_to_local(nb_world)
		var nb_page: int = _flux_chunk_table.get(nb_chunk, -1)
		var val_str := "n/a"
		if nb_page >= 0:
			var nb_local_idx := nb_local.z * cs * cs + nb_local.y * cs + nb_local.x
			var nb_lane_base := (nb_page * vpc * 3 + nb_local_idx * 3 + axis) * 4
			var ndata: PackedByteArray = _compute.download_range(_flux_buf, nb_lane_base, 4)
			val_str = str(ndata.decode_s32(0))
		incoming_str += " in_%s(from %s)=%s" % [axis_name[axis], nb_chunk, val_str]

	print("[%s] chunk %s local (%d,%d,%d): mat=%s fill=%s flux=%s%s" % [label, chunk_pos, lx, ly, lz, mat_str, fill_str, flux_str, incoming_str])


func _debug_print_watch() -> void:
	if not watch_enabled:
		return
	_debug_dump_voxel("FluidWatch", watch_chunk, watch_local)

# Sub-steps per simulation tick — higher = faster fluid, more GPU cost.
# Set to 1 for slow lava-like materials; 2-3 for water.
@export var steps_per_tick: int = 1

# Phase 7e Step 4 tuning — live-editable on the running instance via the Godot remote
# Inspector (select the FluidSimulation node under Remote in the Scene dock while playing).
# damp: fraction of last tick's flux retained before this tick's acceleration is added.
# Lower = more viscous (settles fast, little slosh). Higher = more wave/ripple, longer-lived.
@export_range(0.0, 1.0, 0.001) var damp: float = 0.97
# g_dt: acceleration applied per unit of hydrostatic head difference. Head ranges 0–255
# for a single cell (up to ~6120 for a deep column), and FILL_CAP clamps a single tick's
# flux to 255 — so g_dt needs to be small: a value of 6.0 made a head diff of just ~43
# saturate flux to the maximum possible transfer, meaning a full droplet could empty
# itself in one tick instead of building pressure gradually. Keep this well under 1.0.
#
# Range capped at 0.5 (was 2.0, found 2026-07-02): gravity_bias multiplies this on the Y
# axis by up to its own max — at the old bounds, g_dt=2.0 * gravity_bias=8.0 gives an
# EFFECTIVE Y multiplier of 16, saturating flux for any dh >= 16 (very common, even a
# shallow puddle). Saturating isn't itself the bug (FILL_CAP always clamps to +-255) — the
# bug is that once MULTIPLE competing directions all hit that SAME ceiling, they lose the
# relative distinction gravity_bias exists to create, and Scale divides the budget roughly
# evenly among them again — the exact symmetric-spreading problem gravity_bias fixed,
# reintroduced by oversaturation. Combined with damp carrying a saturated value forward
# each tick, this can oscillate without net progress, which reads as water freezing solid
# mid-air. Reported symptom: raising g_dt froze standing full-fill (255) columns in place;
# resetting g_dt to 0 let stored flux cleanly decay via *damp alone, so ramping back up
# started clean and worked again — until enough ticks/new water pushed it back into the
# saturating regime. This range keeps g_dt*gravity_bias inside the region that stays
# proportional instead of collapsing to the clamp ceiling for typical head differences.
@export_range(0.0, 0.5, 0.01) var g_dt: float = 0.15

# gravity_bias: multiplier on the (dh - elevation) term, Y-axis only (see
# fluid_flux.comp's elevation-head fix, 2026-07-04). Without this, Pass A's formula is
# fully symmetric across axes — a cell's push strength toward an exposed neighbour is
# identical whether that neighbour is above, below, or to the side, producing
# octahedral/spherical expansion instead of coherent falling. 1.0 = no bias (original
# symmetric behavior, useful for regression testing). Range capped at 5.0 (was 8.0) for
# the same oversaturation reason as g_dt above — see its comment.
#
# gravity_force (a separate flat downward constant this variable used to be paired
# with) was removed 2026-07-04 — it broke hydrostatic equilibrium (see docs/BUG_LOG.md
# "ROOT CAUSE FOUND (2026-07-03)"): head() is pressure head only, so a settled column
# always has a nonzero head difference across every vertical face (255, not 0), and a
# flat constant only cancels that by coincidence of tuning. Fixed by using elevation
# head instead (Y-axis dh is now `dh - FILL_CAP`, computed directly in the shader), which
# makes rest a genuine fixed point and gives a floating mass its baseline pull for free.
@export_range(1.0, 5.0, 0.1) var gravity_bias: float = 3.0

# settle_threshold: minimum |applied fill change| (see fluid_flux.comp Transport) for a
# voxel's change to count toward keeping its chunk in _active_chunks next tick. Separate
# from and coarser than MIN_FLOW in the shader (which governs conservation correctness —
# untouched by this). A large settled body of water can slosh at 1-3 units/tick forever
# (flux_buf intentionally persists across ticks so waves don't instantly damp), which
# with the old "any nonzero change keeps the chunk active" rule meant a big lake never
# fully slept even once visually and physically settled. Raising this lets FluidSimulation
# treat sub-threshold sloshing as "settled enough to sleep" — after 4 idle ticks under
# this softer rule the chunk's flux page is freed AND zeroed (see _free_flux_page()), so
# sleep and clearing residual momentum happen together. 1 = old behavior (any nonzero
# change keeps the chunk awake indefinitely).
@export_range(1, 20, 1) var settle_threshold: int = 3

const DIRTY_BUF_BYTES := 64


func _ready() -> void:
	_orchestrator  = get_node("/root/world/WorldOrchestrator")
	_chunk_manager = get_node("/root/world/ChunkManager")
	_compute = preload("res://Utils/compute_helper.gd").new()
	if _compute.has_device():
		_flux_pipeline = _compute.compile_shader(_load_flux_shader(), 4)
		if _flux_pipeline.is_empty():
			push_error("FluidSimulation: fluid_flux.comp compile failed")
		var fp := PackedByteArray(); fp.resize(76); fp.fill(0)  # ...settle_threshold, proj_G, proj_STIFF, proj_OMEGA, proj_LEVEL_RATE, proj_LEVEL_MIN, proj_LEVEL_PRESSURE_MIN
		_flux_params_buf = _compute.buffer_upload(fp)
		_flux_chunk_list_capacity = 256
		var fl := PackedByteArray(); fl.resize(_flux_chunk_list_capacity * 12); fl.fill(0)
		_flux_chunk_list_buf = _compute.buffer_upload(fl)
		var db := PackedByteArray(); db.resize(DIRTY_BUF_BYTES); db.fill(0)
		_dirty_chunks_buf = _compute.buffer_upload(db)
		var cb := PackedByteArray(); cb.resize(4); cb.fill(0)
		_conservation_buf = _compute.buffer_upload(cb)
		var gb := PackedByteArray(); gb.resize(8)
		gb.encode_s32(0, -1); gb.encode_s32(4, 0)  # flat_idx=-1 (no candidate yet), amount=0
		_gain_tracker_buf = _compute.buffer_upload(gb)
	var runtime := get_parent()
	if runtime.has_method("register_simulation"):
		runtime.register_simulation(self)

	Performance.add_custom_monitor("Voxel/fluid_active_chunks", func(): return _active_chunks.size())


func _exit_tree() -> void:
	if Performance.has_custom_monitor("Voxel/fluid_active_chunks"):
		Performance.remove_custom_monitor("Voxel/fluid_active_chunks")
	for rid in [_flux_params_buf, _flux_chunk_list_buf, _dirty_chunks_buf, _flux_buf, _flux_page_table_buf, _conservation_buf, _gain_tracker_buf, _pressure_buf, _div_buf]:
		if rid.is_valid():
			_compute.free_rid(rid)


func get_simulation_class() -> int:
	return MaterialDef.SimulationClass.LIQUID


func add_candidates(chunk_positions: Array[Vector3i]) -> void:
	_ensure_flux_page_table()
	for cp in chunk_positions:
		_active_chunks[cp] = 0
		if _flux_page_table_buf.is_valid():
			_allocate_flux_page(cp)


func has_work() -> bool:
	return not _active_chunks.is_empty()


func get_active_chunk_positions() -> Array[Vector3i]:
	var out: Array[Vector3i] = []
	for cp in _active_chunks:
		out.append(cp)
	return out


# Debug tool (world.gd binds F7): downloads the privately-owned flux buffer for every
# allocated page and prints total |flux| + nonzero-lane count per chunk, plus a grand
# total, to console. On-demand only — full page-buffer readback, never call per-frame.
# Complements WorldOrchestrator.debug_dump_fluid_stats() (which shows fill, i.e. what's
# there): this shows flux, i.e. what's about to move — including on cells that no longer
# hold any fill, which is exactly the signature of the "stale flux survives voxel death"
# bug (a chunk reporting 0 liquid voxels but nonzero |flux| here means exactly that).
func debug_dump_flux_stats() -> void:
	if not _flux_buf.is_valid():
		print("[FluxDebug] flux buffer not ready")
		return
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var lanes_per_page := vpc * 3
	var total_bytes := _flux_page_capacity * lanes_per_page * 4
	var data: PackedByteArray = _compute.download_range(_flux_buf, 0, total_bytes)

	var grand_abs := 0
	var grand_nonzero := 0
	print("[FluxDebug] === snapshot: %d page(s) allocated ===" % _flux_chunk_table.size())
	for cp in _flux_chunk_table:
		var page: int = _flux_chunk_table[cp]
		var base := page * lanes_per_page * 4
		var chunk_abs := 0
		var chunk_nonzero := 0
		for i in range(lanes_per_page):
			var v := data.decode_s32(base + i * 4)
			if v != 0:
				chunk_nonzero += 1
				chunk_abs += abs(v)
		if chunk_nonzero > 0:
			print("[FluxDebug]   chunk %s (page %d): total_|flux|=%d  nonzero_lanes=%d" % [cp, page, chunk_abs, chunk_nonzero])
		grand_abs += chunk_abs
		grand_nonzero += chunk_nonzero
	print("[FluxDebug] TOTAL |flux|=%d across %d nonzero lane(s)" % [grand_abs, grand_nonzero])


# Pressure-projection port debug tool (2026-07-04, Checkpoint 3 — see
# docs/PRESSURE_PROJECTION_PORT_PLAN.md). One-shot BuildRHS dispatch over the CURRENT
# active+neighbour flux list, for manually verifying div_buf against known test scenes
# BEFORE the pass is wired into the real step() dispatch loop (that wiring is
# Checkpoint 5, gated on use_projection). Safe to call regardless of use_projection's
# value or how step() is currently running — this only writes div_buf, which nothing
# else reads yet. Bound to a debug key in world.gd, alongside debug_dump_div().
# Shared setup for the one-shot projection debug dispatches: builds the active+neighbour
# flux list, allocates any missing pages, uploads the chunk list. Returns the flux list
# (empty if not ready/nothing active) and the uniforms array for the caller to dispatch
# with. Does NOT upload params — each pass differs only in params.pass, which callers set.
func _debug_prep_projection_dispatch() -> Dictionary:
	if _flux_pipeline.is_empty() or not _orchestrator.is_grid_valid():
		return {"flux_list": [], "uniforms": []}
	var active_list := _active_chunks.keys()
	var flux_list := _build_flux_chunk_list(active_list)
	if flux_list.is_empty():
		return {"flux_list": [], "uniforms": []}
	_ensure_flux_page_table()
	for cp in flux_list:
		if _flux_page_table_buf.is_valid() and not _flux_chunk_table.has(cp):
			_allocate_flux_page(cp)
	_upload_flux_chunk_list(flux_list)
	return {"flux_list": flux_list, "uniforms": _build_flux_uniforms()}


func debug_dispatch_build_rhs() -> void:
	var prep := _debug_prep_projection_dispatch()
	var flux_list: Array = prep["flux_list"]
	if flux_list.is_empty():
		print("[ProjDebug] not ready or no active chunks to dispatch BuildRHS over")
		return
	var grid_min  := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()
	_upload_flux_params(grid_min, grid_span, _frame, 2)  # BuildRHS
	_compute.run(_flux_pipeline, prep["uniforms"], flux_list.size() * 4, 4, 4)
	_compute.submit()
	_compute.sync()
	print("[ProjDebug] BuildRHS dispatched over %d chunk(s)" % flux_list.size())


# Pressure-projection port debug tool (2026-07-04, Checkpoint 4). One-shot BuildRHS +
# proj_N_ITER red-black GS sweeps, for manually verifying pressure_buf against known test
# scenes before Checkpoint 5 wires this into the real step() loop. Zeroes pressure_buf
# fully before solving (this debug command has no warm-start/persistence concept yet —
# that's a Checkpoint 5+/M2 concern), so each call is a clean, reproducible solve from
# scratch, matching the bench's per-tick "drop stale P in non-liquid cells" reset just
# extended to a full zero for repeatable manual testing.
func debug_dispatch_solve_pressure() -> void:
	var prep := _debug_prep_projection_dispatch()
	var flux_list: Array = prep["flux_list"]
	if flux_list.is_empty():
		print("[ProjDebug] not ready or no active chunks to dispatch solve over")
		return
	var grid_min  := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()
	var uniforms: Array = prep["uniforms"]

	if _pressure_buf.is_valid() and not proj_warm_start:
		var cs := _chunk_manager.chunk_size
		var vpc := cs * cs * cs
		var zero := PackedByteArray(); zero.resize(_flux_page_capacity * vpc * 4); zero.fill(0)
		_compute.buffer_update(_pressure_buf, 0, zero)

	_upload_flux_params(grid_min, grid_span, _frame, 2)  # BuildRHS
	_compute.run(_flux_pipeline, uniforms, flux_list.size() * 4, 4, 4)
	_compute.submit(); _compute.sync()

	for _i in range(proj_N_ITER):
		_upload_flux_params(grid_min, grid_span, _frame, 3)  # solve_red
		_compute.run(_flux_pipeline, uniforms, flux_list.size() * 4, 4, 4)
		_compute.submit(); _compute.sync()
		_upload_flux_params(grid_min, grid_span, _frame, 4)  # solve_black
		_compute.run(_flux_pipeline, uniforms, flux_list.size() * 4, 4, 4)
		_compute.submit(); _compute.sync()

	print("[ProjDebug] BuildRHS + %d red-black sweep(s) dispatched over %d chunk(s)" % [proj_N_ITER, flux_list.size()])


# Checkpoint 6 debug tool (2026-07-05) — one-shot LevelSurface dispatch, for manually
# verifying the surface-leveling pass against known test scenes (e.g. build_utube_test_scene
# or a hand-placed flat puddle) independent of use_projection/use_surface_leveling and the
# real step() loop. Unlike debug_dispatch_build_rhs()/debug_dispatch_solve_pressure() (which
# only write div_buf/pressure_buf, harmless to call standalone), LevelSurface writes real
# changelog entries — this resets and applies the changelog itself so the effect is
# immediately visible in voxels_a, exactly like a single real dispatch in step().
func debug_dispatch_level_surface() -> void:
	var prep := _debug_prep_projection_dispatch()
	var flux_list: Array = prep["flux_list"]
	if flux_list.is_empty():
		print("[ProjDebug] not ready or no active chunks to dispatch LevelSurface over")
		return
	var grid_min  := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()
	_orchestrator.reset_changelog()
	_upload_flux_params(grid_min, grid_span, _frame, 6)  # LevelSurface
	_compute.run(_flux_pipeline, prep["uniforms"], flux_list.size() * 4, 4, 4)
	_compute.submit()
	_compute.sync()
	_orchestrator.apply_changelog()
	print("[ProjDebug] LevelSurface dispatched + applied over %d chunk(s)" % flux_list.size())


# Prints div_buf's value at a specific (chunk, local) voxel — call after
# debug_dispatch_build_rhs(). Reuses the same page-lookup pattern as _debug_dump_voxel().
func debug_dump_div(chunk_pos: Vector3i, local: Vector3i) -> void:
	var page: int = _flux_chunk_table.get(chunk_pos, -1)
	if page < 0:
		print("[ProjDebug] chunk %s has no flux/pressure page" % chunk_pos)
		return
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var local_idx := local.z * cs * cs + local.y * cs + local.x
	var byte_off := (page * vpc + local_idx) * 4
	var data: PackedByteArray = _compute.download_range(_div_buf, byte_off, 4)
	print("[ProjDebug] div at chunk %s local %s = %f" % [chunk_pos, local, data.decode_float(0)])


# Prints pressure_buf's value at a specific (chunk, local) voxel — call after
# debug_dispatch_solve_pressure().
func debug_dump_pressure(chunk_pos: Vector3i, local: Vector3i) -> void:
	var page: int = _flux_chunk_table.get(chunk_pos, -1)
	if page < 0:
		print("[ProjDebug] chunk %s has no flux/pressure page" % chunk_pos)
		return
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var local_idx := local.z * cs * cs + local.y * cs + local.x
	var byte_off := (page * vpc + local_idx) * 4
	var data: PackedByteArray = _compute.download_range(_pressure_buf, byte_off, 4)
	print("[ProjDebug] P at chunk %s local %s = %f" % [chunk_pos, local, data.decode_float(0)])


func step() -> void:
	if _active_chunks.is_empty() or _flux_pipeline.is_empty():
		return
	if not _orchestrator.is_grid_valid():
		return

	# Safety net: add_candidates() normally allocates pages, but if it ran before the
	# grid was valid (e.g. very first tick), retry here before the dispatch needs them.
	_ensure_flux_page_table()
	for cp in _active_chunks:
		if _flux_page_table_buf.is_valid() and not _flux_chunk_table.has(cp):
			_allocate_flux_page(cp)

	var grid_min  := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()

	var active_list := _active_chunks.keys()

	# Flux dispatch covers active + all 6 face neighbours so cross-chunk flux reads/writes
	# (Scale's neighbour-owned-face lookups, Transport's changelog application) land within
	# a single dispatch.
	var flux_list := _build_flux_chunk_list(active_list)
	_upload_flux_chunk_list(flux_list)
	var flux_count := flux_list.size()

	# Every chunk in the flux dispatch range needs a page, not just _active_chunks — Pass A
	# gates its entire per-axis loop on my_page (2026-07-02 fix made air cells compute their
	# own owned faces, but that computation is still gated on the CURRENT voxel's own page).
	# Positive-direction flow crossing a chunk boundary stores on the DONOR's face (already
	# paged, since the donor's chunk is active) and works fine either way. Negative-direction
	# flow — including straight down, gravity is -Y — stores on the RECEIVER's own owned
	# face: if that neighbour chunk has never held water, it has no page, so Pass A returns
	# immediately for every voxel there and nothing is ever computed, let alone materialized
	# — there's no changelog entry to trigger discovery, so this doesn't self-heal like the
	# positive-direction case does. Allocating pages for the full neighbour set up front
	# closes that hole.
	for cp in flux_list:
		if _flux_page_table_buf.is_valid() and not _flux_chunk_table.has(cp):
			_allocate_flux_page(cp)

	# Restore the GPU page-table entry for every dispatched chunk this tick — a previous
	# tick's exclusion loop below may have temporarily unmapped it (written -1) even
	# though the CPU-side page itself was never freed. Pass A's new neighbour-page check
	# (2026-07-02) depends on flux_page_table accurately reflecting THIS TICK's dispatch
	# membership, not just "ever had a page".
	for cp in flux_list:
		var restore_page: int = _flux_chunk_table.get(cp, -1)
		if restore_page >= 0:
			_write_flux_page_table_entry(cp, restore_page)

	# Zero flux data AND unmap the page table entry for any paged chunk NOT in this
	# tick's dispatch range (2026-07-02). Originally this only zeroed flux data, based on
	# the theory that stale-but-paged data was the leak — that fix did NOT resolve the
	# observed leak (confirmed by user testing). The GainTracker diagnostic subsequently
	# found the real mechanism: Pass A's per-axis loop reads a neighbour's REAL voxel
	# data via read_a() regardless of paging/dispatch status, and previously only checked
	# flux_page_table for "has a page" (true for a chunk still in its idle-guard grace
	# period, even though it isn't part of THIS tick's shader dispatch and therefore never
	# runs Transport to pay its side of any transfer). Un-mapping the page table entry
	# (not just zeroing flux data) makes flux_page_table accurately reflect "in this
	# tick's dispatch" — which Pass A's new neighbour-page check (fluid_flux.comp) now
	# relies on to refuse computing flux against an un-dispatched neighbour at all. The
	# CPU-side page allocation itself (_flux_chunk_table) is untouched — this is a
	# per-tick logical unmap, restored every tick a chunk re-enters flux_list (see the
	# restore loop above), not a free. Idle-guard eviction (actual page freeing) is
	# unaffected, still governed separately below.
	var in_dispatch_range := {}
	for cp in flux_list:
		in_dispatch_range[cp] = true
	for cp in _flux_chunk_table.keys():
		if not in_dispatch_range.has(cp):
			var page: int = _flux_chunk_table[cp]
			var cs := _chunk_manager.chunk_size
			var vpc := cs * cs * cs
			var zero := PackedByteArray(); zero.resize(vpc * 3 * 4); zero.fill(0)
			_compute.buffer_update(_flux_buf, page * vpc * 3 * 4, zero)
			_write_flux_page_table_entry(cp, -1)

	var merged_bitmask := PackedByteArray(); merged_bitmask.resize(DIRTY_BUF_BYTES); merged_bitmask.fill(0)
	var zero_dirty     := PackedByteArray(); zero_dirty.resize(DIRTY_BUF_BYTES); zero_dirty.fill(0)
	var zero4          := PackedByteArray(); zero4.resize(4); zero4.fill(0)
	var reset_gain     := PackedByteArray(); reset_gain.resize(8)
	reset_gain.encode_s32(0, -1); reset_gain.encode_s32(4, 0)

	var flux_uniforms := _build_flux_uniforms()

	for _s in range(steps_per_tick):
		_frame += 1

		# Pass A (accelerate) -> Transport (fused conservation guard + apply, writes
		# changelog). Persistent: flux_buf is never zeroed between ticks (see
		# fluid_flux.comp header). Transport reads flux_buf and voxels_a that Accelerate
		# already finished writing, a genuine cross-workgroup dependency, so it still needs
		# its own submit+sync after Accelerate — but that's the only cross-dispatch
		# dependency left (Phase 7e Step 9, 2026-07-06: the old Scale-Outflow/Scale-Inflow
		# passes are gone; Transport now recomputes their scale factors inline per-cell
		# instead of reading them back from a buffer a prior pass wrote — see
		# fluid_flux.comp's compute_s_out()/compute_s_in()). Transport is the only pass
		# that touches the changelog, so it gets its own reset_changelog()/
		# apply_changelog() pair immediately around its dispatch. Guard: flux_count can be
		# 0 if all active chunks were unloaded since last step().
		var flux_bitmask := zero_dirty  # default: no flux changes this sub-step
		if flux_count > 0:
			# Pressure-projection port, Checkpoint 5 (docs/PRESSURE_PROJECTION_PORT_PLAN.md).
			# use_projection swaps Accelerate for BuildRHS -> zero pressure_buf -> N red-black
			# sweeps -> Correct. Transport below is completely unchanged either way -- it
			# only ever reads flux_buf, never cares which pass populated it this tick.
			if use_projection:
				_upload_flux_params(grid_min, grid_span, _frame, 2)  # BuildRHS
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()

				# Quasi-static (M1): pressure is recomputed fresh every tick, never
				# persisted -- see the port plan's M1 milestone note on why (no
				# warm-start/inertia yet, that's a separate, later M2 concern).
				# Diagnostic override: proj_warm_start skips this zero, letting
				# solve_red/solve_black refine last tick's pressure_buf instead --
				# testing whether the observed settle-time jitter/checkerboard is a
				# convergence-consistency artifact of re-solving from scratch every
				# tick, see the export's own comment above.
				if _pressure_buf.is_valid() and not proj_warm_start:
					var cs := _chunk_manager.chunk_size
					var vpc := cs * cs * cs
					var zero_p := PackedByteArray(); zero_p.resize(_flux_page_capacity * vpc * 4); zero_p.fill(0)
					_compute.buffer_update(_pressure_buf, 0, zero_p)

				for _iter in range(proj_N_ITER):
					_upload_flux_params(grid_min, grid_span, _frame, 3)  # solve_red
					_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
					_compute.submit(); _compute.sync()
					_upload_flux_params(grid_min, grid_span, _frame, 4)  # solve_black
					_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
					_compute.submit(); _compute.sync()

				_upload_flux_params(grid_min, grid_span, _frame, 5)  # Correct
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()
			else:
				_upload_flux_params(grid_min, grid_span, _frame, 0)  # Accelerate
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()

			_orchestrator.reset_changelog()
			_compute.buffer_update(_dirty_chunks_buf, 0, zero_dirty)
			_upload_flux_params(grid_min, grid_span, _frame, 1)  # Transport
			_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
			_compute.submit()
			_compute.sync()

			# Checkpoint 6 (2026-07-05, FLAT_PUDDLE_HANDOFF.md section 5b) — LevelSurface,
			# dispatched after Transport (and after any projection passes above). Appends
			# to the SAME changelog Transport just wrote to (no reset_changelog() call
			# here) so both are applied together by the single apply_changelog() below.
			# Gated separately from use_projection: it's a standalone fix for a
			# projection-only gap and has no effect when use_projection is false.
			if use_projection and use_surface_leveling:
				_upload_flux_params(grid_min, grid_span, _frame, 6)  # LevelSurface
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()

			flux_bitmask = _compute.download_range(_dirty_chunks_buf, 0, DIRTY_BUF_BYTES)
			_orchestrator.apply_changelog()

			_debug_print_watch()

			# Automatic conservation check (2026-07-02) — GPU-side, no manual F6 diffing
			# needed. Transport atomicAdds every real (new_fill - old_fill) it applies into
			# _conservation_buf; if the simulation conserves mass perfectly, that sum is
			# exactly 0 for a tick fully contained within the dispatch range (every unit
			# lost by one voxel is gained by another). A nonzero total is a direct,
			# quantified leak/gain measurement — only printed when it actually happens, so
			# this stays silent during normal correct operation.
			#
			# Gated behind debug_conservation_check (default false, added 2026-07-03) —
			# the download_range() calls below are a GPU readback that forces a sync every
			# single sub-step regardless of whether anything gets printed, which measurably
			# hurt performance once the leak investigation was done and this was left
			# running unconditionally.
			#
			# BUG (found 2026-07-03): the reset below used to live INSIDE this gate too —
			# Transport's atomicAdd into _conservation_buf runs every tick regardless of
			# this toggle, so leaving the toggle off let the buffer silently accumulate,
			# unread and unreset, for however long it stayed off. The instant it was
			# flipped back on, the first read dumped the ENTIRE accumulated backlog as if
			# it were a single tick's loss (observed: a reported "45814 lost this tick"
			# that was actually the sum of many quiet ticks, not one real event). Fix: the
			# reset must always run every tick, independent of whether we're reading/
			# printing this tick's value — only the (expensive) read+print is gated.
			if debug_conservation_check:
				var cons: PackedByteArray = _compute.download_range(_conservation_buf, 0, 4)
				var cons_delta := cons.decode_s32(0)
				if cons_delta != 0:
					print("[Conservation] fill %s this tick: %d" % ["gained" if cons_delta > 0 else "lost", abs(cons_delta)])

					# Location tracker (2026-07-02) — the stale-page fix didn't resolve the leak,
					# so pinpoint WHERE the single largest gain happened instead of guessing
					# further. Decode the flat voxel index back into chunk position + local xyz.
					var gt: PackedByteArray = _compute.download_range(_gain_tracker_buf, 0, 8)
					var gain_flat := gt.decode_s32(0)
					var gain_amount := gt.decode_s32(4)
					if gain_flat >= 0:
						var cs := _chunk_manager.chunk_size
						var vpc := cs * cs * cs
						var slot := gain_flat / vpc
						var local := gain_flat % vpc
						var lz := local / (cs * cs)
						var ly := (local % (cs * cs)) / cs
						var lx := local % cs
						var chunk_pos: Variant = _orchestrator._slot_to_chunk.get(slot, null)
						print("[Conservation]   biggest single gain: %d at chunk %s local (%d,%d,%d)" % [gain_amount, chunk_pos, lx, ly, lz])
						# Auto-dump the flagged voxel itself (2026-07-03) — its location moves
						# every tick, so retargeting watch_chunk/watch_local by hand after
						# reading a previous tick's log line always lags a step behind the
						# actual leaking cell. This grabs it live, same tick it's reported.
						if chunk_pos != null:
							_debug_dump_voxel("GainVoxel", chunk_pos, Vector3i(lx, ly, lz))
			_compute.buffer_update(_conservation_buf, 0, zero4)
			_compute.buffer_update(_gain_tracker_buf, 0, reset_gain)

		# OR dirty bits across sub-steps.
		for i in range(DIRTY_BUF_BYTES / 4):
			var a := merged_bitmask.decode_u32(i * 4)
			var c := flux_bitmask.decode_u32(i * 4)
			merged_bitmask.encode_u32(i * 4, a | c)

	var changed: Array[Vector3i] = _decode_dirty_bitmask(merged_bitmask)
	if not changed.is_empty():
		for cp in changed:
			var chunk: Chunk = _chunk_manager.chunks.get(cp)
			if chunk:
				chunk.mark_simulation_active(MaterialDef.SimulationClass.LIQUID)
		_orchestrator.chunks_committed.emit(changed)

	var changed_set: Dictionary = {}
	for cp in changed:
		changed_set[cp] = true
	var still_active: Dictionary = {}
	for cp in _active_chunks:
		if changed_set.has(cp):
			still_active[cp] = 0
		else:
			var idle: int = _active_chunks[cp] + 1
			if idle < 4:
				still_active[cp] = idle
	for cp in changed:
		if not still_active.has(cp):
			still_active[cp] = 0

	# Free flux pages for chunks that just fell out of the active set — eviction rides
	# the same idle-guard counter that already governs GranularSimulation's chunk sleep.
	for cp in _active_chunks:
		if not still_active.has(cp):
			_free_flux_page(cp)

	_active_chunks = still_active

	# Evict pages allocated purely as flux-dispatch neighbours (chunk-boundary fix,
	# 2026-07-02) that never actually became active — without this, sparse paging's core
	# promise ("most loaded chunks never touch fluid and pay zero flux storage") breaks
	# permanently for every chunk adjacent to any water body. A few ticks of grace before
	# freeing: a chunk that materialized its first voxel THIS tick won't show up in
	# _active_chunks until next tick's discovery runs, so freeing immediately would yank
	# its page out from under it mid-flow.
	var still_page_idle: Dictionary = {}
	for cp in _flux_chunk_table.keys():
		if _active_chunks.has(cp):
			continue  # active chunks keep their page via the eviction loop above
		var idle: int = _flux_page_idle.get(cp, 0) + 1
		if idle < 4:
			still_page_idle[cp] = idle
		else:
			_free_flux_page(cp)
	_flux_page_idle = still_page_idle


# --- Internal ---

# Allocates the GPU-side page-table mirror, sized to grid_span (chunk_pos → page index,
# -1 = no page). grid_span is fixed once WorldOrchestrator._init_grid() runs and never
# grows afterward (only the voxel slot pool grows via buffer_grow) — same invariant
# WorldOrchestrator._chunk_table already relies on — so this never needs to grow either.
func _ensure_flux_page_table() -> void:
	if _flux_page_table_buf.is_valid():
		return
	if not _orchestrator.is_grid_valid():
		return
	var span := _orchestrator.get_grid_span()
	var total := span.x * span.y * span.z
	if total <= 0:
		return
	var data := PackedByteArray(); data.resize(total * 4); data.fill(0xFF)  # -1 as s32
	_flux_page_table_buf = _compute.buffer_upload(data)


# Grows/creates the page-indexed flux buffer so it can hold at least min_pages pages.
# Phase 7e Step 4: each page now holds 3 signed int lanes per voxel (fx, fy, fz), not 1 —
# stride 3, interleaved — since every liquid voxel owns 3 persistent flux faces.
func _ensure_flux_buf_capacity(min_pages: int) -> void:
	var cs  := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	if not _flux_buf.is_valid():
		var initial = max(FLUX_PAGE_INITIAL_CAPACITY, min_pages)
		var bytes = initial * vpc * 3 * 4
		var data := PackedByteArray(); data.resize(bytes); data.fill(0)
		_flux_buf = _compute.buffer_upload(data)
		_flux_page_capacity = initial
		for i in range(initial - 1, -1, -1):
			_flux_freelist.append(i)
		# _pressure_buf/_div_buf ride the same page count as _flux_buf, 1 float32/voxel
		# each (vs. flux's 3 lanes/voxel) — see the field declarations above.
		var scalar_bytes = initial * vpc * 4
		var pdata := PackedByteArray(); pdata.resize(scalar_bytes); pdata.fill(0)
		_pressure_buf = _compute.buffer_upload(pdata)
		var ddata := PackedByteArray(); ddata.resize(scalar_bytes); ddata.fill(0)
		_div_buf = _compute.buffer_upload(ddata)
		return
	if min_pages > _flux_page_capacity:
		var old_bytes := _flux_page_capacity * vpc * 3 * 4
		var new_capacity := _flux_page_capacity * 2
		while new_capacity < min_pages:
			new_capacity *= 2
		var new_bytes := new_capacity * vpc * 3 * 4
		_flux_buf = _compute.buffer_grow(_flux_buf, old_bytes, new_bytes)
		var old_scalar_bytes := _flux_page_capacity * vpc * 4
		var new_scalar_bytes := new_capacity * vpc * 4
		_pressure_buf = _compute.buffer_grow(_pressure_buf, old_scalar_bytes, new_scalar_bytes)
		_div_buf = _compute.buffer_grow(_div_buf, old_scalar_bytes, new_scalar_bytes)
		for i in range(new_capacity - 1, _flux_page_capacity - 1, -1):
			_flux_freelist.append(i)
		_flux_page_capacity = new_capacity


# chunk_pos → grid-flat index, identical formula to WorldOrchestrator._chunk_flat_idx.
func _flux_table_idx(chunk_pos: Vector3i) -> int:
	var grid_min  := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()
	var l := chunk_pos - grid_min
	if l.x < 0 or l.x >= grid_span.x or l.y < 0 or l.y >= grid_span.y or l.z < 0 or l.z >= grid_span.z:
		return -1
	return l.x + l.y * grid_span.x + l.z * grid_span.x * grid_span.y


func _write_flux_page_table_entry(chunk_pos: Vector3i, page: int) -> void:
	if not _flux_page_table_buf.is_valid():
		return
	var idx := _flux_table_idx(chunk_pos)
	if idx < 0:
		return
	var b := PackedByteArray(); b.resize(4); b.encode_s32(0, page)
	_compute.buffer_update(_flux_page_table_buf, idx * 4, b)


func _allocate_flux_page(chunk_pos: Vector3i) -> void:
	if _flux_chunk_table.has(chunk_pos):
		return
	if _flux_freelist.is_empty():
		_ensure_flux_buf_capacity(_flux_page_capacity + 1)
	if _flux_freelist.is_empty():
		return  # buffer alloc failed (no compute device) — leave as zero-flux boundary
	var page: int = _flux_freelist.pop_back()
	_flux_chunk_table[chunk_pos] = page
	_write_flux_page_table_entry(chunk_pos, page)


func _free_flux_page(chunk_pos: Vector3i) -> void:
	if not _flux_chunk_table.has(chunk_pos):
		return
	var page: int = _flux_chunk_table[chunk_pos]
	_flux_chunk_table.erase(chunk_pos)
	_flux_freelist.append(page)
	_write_flux_page_table_entry(chunk_pos, -1)
	# Zero the vacated page so a future occupant doesn't inherit stale momentum
	# (Step 6b — flux invalidation applies to page reuse as well as cell reuse).
	var cs  := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var zero := PackedByteArray(); zero.resize(vpc * 3 * 4); zero.fill(0)
	_compute.buffer_update(_flux_buf, page * vpc * 3 * 4, zero)
	# A future occupant must not inherit stale pressure/RHS any more than stale flux.
	var scalar_zero := PackedByteArray(); scalar_zero.resize(vpc * 4); scalar_zero.fill(0)
	_compute.buffer_update(_pressure_buf, page * vpc * 4, scalar_zero)
	_compute.buffer_update(_div_buf, page * vpc * 4, scalar_zero)


func _decode_dirty_bitmask(bitmask: PackedByteArray) -> Array[Vector3i]:
	var slot_to_chunk := _orchestrator._slot_to_chunk
	var out: Array[Vector3i] = []
	var words := bitmask.size() / 4
	for i in range(words):
		var word := bitmask.decode_u32(i * 4)
		if word == 0:
			continue
		for bit in range(32):
			if word & (1 << bit):
				var slot := i * 32 + bit
				var cp: Variant = slot_to_chunk.get(slot, null)
				if cp != null:
					out.append(cp)
	return out


func _build_flux_uniforms() -> Array:
	var u := func(binding: int, rid: RID) -> RDUniform:
		var un := RDUniform.new()
		un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		un.binding = binding
		un.add_id(rid)
		return un
	return [
		u.call(0, _orchestrator.get_voxel_buf()),
		u.call(1, _orchestrator.get_chunk_table()),
		u.call(2, _flux_params_buf),
		u.call(3, _orchestrator.get_material_sim_buf()),
		u.call(4, _orchestrator.get_changelog_buf()),
		u.call(5, _flux_chunk_list_buf),   # active + 6 face neighbours
		u.call(6, _dirty_chunks_buf),
		u.call(7, _flux_buf),
		u.call(8, _flux_page_table_buf),
		u.call(9, _conservation_buf),
		u.call(10, _gain_tracker_buf),
		u.call(11, _pressure_buf),  # pressure-projection port, Checkpoint 2 — unread until Checkpoint 4
		u.call(12, _div_buf),       # pressure-projection port, Checkpoint 2 — unread until Checkpoint 3
	]


func _upload_flux_params(grid_min: Vector3i, grid_span: Vector3i, frame: int, pass_idx: int) -> void:
	var cs := _chunk_manager.chunk_size
	var p := PackedByteArray(); p.resize(76)
	p.encode_s32(0,  cs)
	p.encode_s32(4,  grid_min.x)
	p.encode_s32(8,  grid_min.y)
	p.encode_s32(12, grid_min.z)
	p.encode_s32(16, grid_span.x)
	p.encode_s32(20, grid_span.y)
	p.encode_s32(24, grid_span.z)
	p.encode_s32(28, frame)
	p.encode_s32(32, pass_idx)
	p.encode_float(36, damp)
	p.encode_float(40, g_dt)
	p.encode_float(44, gravity_bias)
	p.encode_s32(48, settle_threshold)
	# Pressure-projection port (Checkpoint 2) — unread by any shader pass until Checkpoints
	# 3-5 implement build_rhs/solve_red/solve_black/correct; harmless to upload now.
	p.encode_float(52, proj_G)
	p.encode_float(56, proj_STIFF)
	p.encode_float(60, proj_OMEGA)
	# Checkpoint 6 (surface-leveling pass, params.pass == 6) — unread by any other pass.
	p.encode_float(64, proj_LEVEL_RATE)
	p.encode_float(68, proj_LEVEL_MIN)
	p.encode_float(72, proj_LEVEL_PRESSURE_MIN)
	_compute.buffer_update(_flux_params_buf, 0, p)


func _build_flux_chunk_list(active_list: Array) -> Array:
	# Active chunks + all 6 face neighbours so cross-chunk flux reads/writes (Scale's
	# neighbour-owned-face lookups, Transport's changelog application) all land within a
	# single dispatch. Vertical neighbours were added when Pass A folded the Y axis into
	# flux (Phase 7e Step 4) — before that, only horizontal neighbours mattered.
	var seen := {}
	var out  := []
	var offsets: Array[Vector3i] = [
		Vector3i( 0,  0,  0),
		Vector3i( 1,  0,  0), Vector3i(-1,  0,  0),
		Vector3i( 0,  0,  1), Vector3i( 0,  0, -1),
		Vector3i( 0,  1,  0), Vector3i( 0, -1,  0),
	]
	for cp: Vector3i in active_list:
		for offset in offsets:
			var nb: Vector3i = cp + offset
			if not seen.has(nb) and _chunk_manager.chunks.has(nb):
				seen[nb] = true
				out.append(nb)
	return out


func _upload_flux_chunk_list(chunk_positions: Array) -> void:
	var count := chunk_positions.size()
	if count > _flux_chunk_list_capacity:
		if _flux_chunk_list_buf.is_valid():
			_compute.free_rid(_flux_chunk_list_buf)
		_flux_chunk_list_capacity = max(count, _flux_chunk_list_capacity * 2)
		var blank := PackedByteArray(); blank.resize(_flux_chunk_list_capacity * 12); blank.fill(0)
		_flux_chunk_list_buf = _compute.buffer_upload(blank)
	var data := PackedByteArray(); data.resize(count * 12)
	for i in count:
		var cp: Vector3i = chunk_positions[i]
		data.encode_s32(i * 12,     cp.x)
		data.encode_s32(i * 12 + 4, cp.y)
		data.encode_s32(i * 12 + 8, cp.z)
	_compute.buffer_update(_flux_chunk_list_buf, 0, data)


func _load_flux_shader() -> String:
	var f := FileAccess.open("res://shaders/fluid_flux.comp", FileAccess.READ)
	if f: return f.get_as_text()
	push_error("FluidSimulation: failed to load fluid_flux.comp")
	return ""
