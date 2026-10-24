class_name MPMSimulation extends Node

# Production MLS-MPM node (tools/fluid_prototypes/MPM_EVALUATION_BRIEF.md §35-37). Deliberately
# NOT a VoxelSimulation: that base class's add_candidates(chunk_positions)/has_work() contract
# assumes chunk-granularity CA dispatch woken by a per-chunk material-class bitmask, which is
# ALMOST the right shape — see below — but its dispatch model (one shared shader invocation
# over a chunk_list, reading the sparse world buffer directly) doesn't fit MPM's need for a
# contiguous per-region background grid (P2G/G2P scatter/gather needs real float accumulator
# arrays, not just per-voxel material IDs). SimulationRuntime ticks this directly (tick_all()),
# not via register_simulation()/get_simulation_class() — that path assumes one sim instance per
# SimulationClass bit, and MPM owns several materials across GRANULAR+LIQUID at once.
#
# CROSS-CHUNK CONTINUITY (2026-08-13 — this replaces the per-chunk-window model).
# The original design gave EVERY active chunk its own independent 32³ grid, and its own header
# flagged the consequence as an unsolved seam: "a particle exactly on a chunk boundary doesn't
# hand off cleanly to the neighboring chunk's grid... Fine for now (isolated ponds/piles)."
# It was not fine — it was three separate walls, all in the shaders:
#   1. p2g.comp  bounds-checked nodes out of the cube (`node >= ng → continue`), so a particle
#      within one cell of the boundary lost part of its quadratic B-spline kernel: mass vanished
#      into the void and the particle saw a phantom free surface.
#   2. g2p.comp  did the same on the gather, so velocity/affine were only partially weighted.
#   3. g2p.comp  clamped position to [0, ng) — particles literally could not cross at all.
#   4. grid_op.comp damped a 3-cell band (P.bound) at every edge, i.e. a soft wall.
#
# The fix is a SHARED GRID: one region covers the bounding box of all active chunks, so chunk
# boundaries that fall inside it are now interior nodes with no special handling at all, and the
# four walls above only exist at the true edge of the simulated domain — where they're correct.
# Grids are no longer cubes (`dims: Vector3i`), which is the whole shader-side change.
#
# TRADEOFF, stated rather than hidden: a single region is the bounding box of ALL active chunks,
# so two far-apart puddles waste grid between them. With max_active_chunks small and contiguous
# material (a lake, a waterfall) this is nothing; MAX_REGION_NODES bounds the pathological case
# by refusing to grow rather than allocating gigabytes. The refinement — one region per
# CONNECTED component (union-find over face-adjacency) — is deliberately deferred: it multiplies
# the bookkeeping (merge/split on activation) without changing anything a contiguous lake sees.
#
# Uses Utils/compute_helper.gd like every other GPU system in this project — its
# ComputeHelper.new() shares the single static RenderingDevice all of them run on, so this
# reads/writes the SAME buffer space as the raymarcher and collision.
#
# Per-material physics (model/density/stiffness/viscosity/friction) live in
# MaterialDB.mpm_buffer_rid, bound directly into p2g.comp/g2p.comp/grid_op.comp — NOT
# duplicated as scalar params here (§35.6 M1).
#
# Grid is LOCKED to real voxel resolution: dx is always 1.0 (one MPM cell = one voxel) — the
# "no snap" design (brief §15-18, Part III): the MPM background grid IS the voxel grid.

# Chunk edge length in cells. Must equal ChunkManager.chunk_size — every chunk→cell conversion
# below uses this, and activation/readout assume chunks tile the grid exactly.
@export var n_grid: int = 32
@export var gravity: float = 9.8

# SIMULATED SECONDS ADVANCED PER REAL TICK. Substep dt is derived from this and the substep
# count, so there is ONE pacing knob instead of two that must agree.
#
# This replaces a hardcoded `sim_dt = 1.0e-4` with 24 substeps: that advanced only 2.4e-3 s per
# tick, i.e. 0.144 s of simulation per real second — water fell at roughly a SEVENTH of real
# speed, which is the "falls really slowly" report. At 1/60 with 24 substeps the substep dt is
# 6.94e-4, still far under the MPM CFL limit for this port (sound speed ≈ sqrt(E/ρ) ≈ 89, so
# dt_max ≈ 1.1e-2), so real time is comfortably stable. Raise it for slow-motion, lower it if a
# violent impact starts blowing up.
@export var sim_seconds_per_tick: float = 1.0 / 60.0

# TIMESCALE PROBE. Cycles sim_seconds_per_tick through a ladder so the stability threshold can be
# found in one session instead of one build per guess (KEY_T in world.gd).
#
# Why this exists: the "falls too slowly" fix multiplied sim_dt by 6.94x (1.0e-4 -> 6.94e-4 by
# making a tick advance real time). An MPM fluid integrated too coarsely does not just look wrong,
# it EXPANDS — the J-model pressure impulse over-shoots, J drifts away from 1, and the body pushes
# itself apart until it fills whatever volume it is given. That is indistinguishable from a region
# growth bug by looking at the region: material genuinely reaches every wall, at genuine speed, so
# every growth criterion we have tried fires legitimately. If the fluid is meta-stable, the region
# machinery is being blamed for a solver problem.
const TIMESCALE_LADDER := [
	1.0 / 60.0,   # real time — the current default, and the suspected unstable one
	1.0 / 120.0,  # 2x finer
	1.0 / 240.0,  # 4x finer
	1.0 / 480.0,  # 8x finer — near the original hand-tuned value
	1.0 / 30.0,   # deliberately too coarse, for contrast
]
var _timescale_index: int = 0

# Particle capacity for the WHOLE region (not per chunk). One full 32³ chunk of water is 32768
# voxels, so the old 4000 could not represent even an eighth of a single chunk — and the
# promotion scan fills in ascending y, so overflow truncated the TOP of a poured sphere. 64K
# holds two full chunks; raising it costs ~7 MB of particle buffers per region.
@export var max_particles: int = 65536
# Cap on how many chunks may be active at once (bounds the region's grid). 8 was too tight for
# anything but an isolated puddle: water reaching the edge of an un-simulated chunk sees a hard
# wall and piles up it, which is the "climbs up that wall" report. 16 doubles the reach without
# letting the bounding box get big enough to tank the per-substep dispatch cost — the real
# driver of cost is the BOX, not the chunk count, so this is a balance point, not a free win.
@export var max_active_chunks: int = 16

# PROMOTION SAMPLING STRIDE. 1 = one particle per voxel (full fidelity). N = one particle per
# NxNxN voxel block: N^3 fewer particles, each carrying N^3 times the mass, on the SAME grid
# resolution. This is the "make each particle bigger so we use fewer" knob — but note what it
# does and does not save: the grid passes (clear/P2G/grid_op/G2P) all still run over the same
# node count, so it cuts only the per-particle work, and it does so by SAMPLING, so the surface
# gets correspondingly blockier.
#
# Left at 1 deliberately: the cost curve has not been measured yet, and this project's own rule
# (ARCHITECTURE.md §7.0) is measure before cutting. Use the `mpm_perf` / `mpm_stress` modes to
# see whether particles or grid nodes dominate before spending fidelity here.
@export var promote_stride: int = 1

@export_group("Particle density")
# PARTICLES PER VOXEL, per axis. 1 = one particle per voxel per axis (the current default, and
# MPM's minimum — the reference implementation this port follows uses 2, i.e. EIGHT particles per
# cell). Total mass and volume are conserved: each sub^3 group shares the voxel's mass.
#
# Why this exists as an experiment: with one particle per cell, a particle at a free surface has
# only part of its 3x3x3 gathering stencil populated. Pressure is scattered into nodes weighted by
# mass, so the empty half of the stencil contributes gravity but no pressure — the surface
# particle's velocity is diluted toward "no pressure", and the body spreads into a film instead of
# holding depth. That is the leading explanation for the pancake, and this knob tests it directly:
# if subdiv 2 stops the filming, the container/window was never the problem.
#
# COST: sub^3 times the particles (2 = 8x). Particle work is the cheaper half of the loop — the
# grid passes are unchanged — but max_particles must be raised to match, or seeding truncates.
@export_range(1, 3, 1) var particle_subdiv: int = 1

# MINIMUM region size, in chunks. This is a FLOOR, not the size: the region starts at the extent
# of the edit that created it (never smaller than this) and then grows on evidence.
#
# 1x1x1 = 32^3 = 32,768 nodes ≈ 0.3 ms. A single sphere inside one chunk therefore costs ~0.3 ms
# rather than the 2.4 ms that allocating a 2x2x2 window "in case it spreads" would cost. Raise
# this only if you want a guaranteed margin around new material.
@export var region_chunks: Vector3i = Vector3i(1, 1, 1)

@export_group("Region growth")
# Grow the window when material touches a face — a chunk the material has actually reached becomes
# part of the simulated volume. This is the CA-style "add the chunk the material moved into",
# expressed at region granularity because MPM needs one contiguous grid.
#
# GROWTH IS DRIVEN BY THE SIMULATION, not by the edit: readout.comp reports per-chunk occupancy
# each tick, and _growth_for() grows toward whichever faces hold material.
#
# BOUNDED by max_axis_chunks and MAX_REGION_NODES. That ceiling is what the original growth
# implementation lacked, and its absence (not growth itself) produced the runaway.
@export var grow_window: bool = true
# Ceiling per axis. 2 = up to 64^3 (262,144 nodes, ~2.4 ms); 3 = up to 96^3 (884,736, ~5 ms).
@export var max_axis_chunks: int = 2

@export_group("Water physics (live)")
# LIVE overrides for WATER (id 3). Edit these in the REMOTE Inspector while the game runs —
# Godot: Debugger > Remote Scene Tree > WorldOrchestrator/SimulationRuntime/MPMSimulation — and
# you get real sliders against the running fluid, no relaunch.
#
# stiffness and viscosity are read straight out of the material table every substep, so they take
# effect immediately. density feeds particle MASS at seed time, so changing it forces a region
# rebuild (state preserved; seed-time mass re-derived).
@export_range(0.05, 8.0, 0.05) var water_density: float = 1.0:
	set(v):
		water_density = v
		_apply_water_override(true)
@export_range(50.0, 200000.0, 50.0) var water_stiffness: float = 8000.0:
	set(v):
		water_stiffness = v
		_apply_water_override(false)
@export_range(0.0, 50.0, 0.1) var water_viscosity: float = 0.0:
	set(v):
		water_viscosity = v
		_apply_water_override(false)


# Per-particle rest volume, and the fixed-point scale derived from it.
#
# MUST be recomputed whenever promote_stride or particle_subdiv changes. p2g.comp computes its
# pressure impulse as -dt * 4E * P.p_vol * (J-1) * inv_dx^2, using THIS value — so if particles are
# subdivided and only their MASS is divided, each particle still claims a full voxel's worth of
# pressure response while carrying 1/sub^3 of the mass. Total pressure momentum then comes out
# sub^3 times too high against the same total mass, and the fluid behaves as though its stiffness
# were multiplied by sub^3 (subdiv 2 turned E=8000 into an effective 64,000: over-pressurised to
# the point where the max_speed clamp engaged and the body froze mid-air).
#
# Dividing p_vol by sub^3 keeps mass, volume and stiffness consistent: each particle owns 1/sub^3
# of the voxel's volume and the matching share of both its mass and its pressure response.
# fp_scale compensates automatically — g_mass ends up as density*weight*NODE_BUDGET regardless.
func _refresh_scales() -> void:
	var st := maxi(promote_stride, 1)
	var sub := maxi(particle_subdiv, 1)
	p_vol = pow(dx * 0.5, 3.0) * float(st * st * st) / float(sub * sub * sub)
	fp_scale = NODE_BUDGET / p_vol


func _apply_water_override(reseed: bool) -> void:
	# Setters fire during scene load, before _ready has resolved the material DB.
	if not is_setup or not is_instance_valid(_material_db):
		return
	_material_db.override_mpm(3, water_density, water_stiffness, water_viscosity)
	if reseed:
		hard_rebuild()
# Hard ceiling on region grid nodes. 1M nodes ≈ 60 MB of grid buffers (60 B/node) and ≈ 15.6k
# workgroups per pass. Exceeded → the chunk that would cross it is refused, not allocated.
const MAX_REGION_NODES := 1048576
# Width of the velocity-damping band grid_op applies inside each region face, in cells. This is
# an artificial wall: material stops `BOUND` cells short of the true edge, which both looks wrong
# (water hovering above an invisible floor) and used to hide the edge from face-presence detection
# (readout.comp now tests the band rather than the plane). 1 keeps the wall as thin as it can be
# while still stopping material from leaving.
@export var bound: int = 1
const NMAT := 8  # must match clear_grid.comp/p2g.comp's NMAT — the render mass-breakdown slot count
const DIRTY_BUF_BYTES := 64  # matches GranularSimulation's own constant/convention
# Region-grid chunk occupancy bitmask: 512 chunks' worth, far more than a bounding box under
# MAX_REGION_NODES can hold. Must be a multiple of 4 (u32 words).
const OCCUPANCY_BYTES := 64
@export var idle_threshold: int = 4
# Target magnitude for a busy grid node's fixed-point accumulator (fp_scale is derived from
# this, not hardcoded — §36.3 found a fixed magic constant silently rounds low-density
# materials to zero at one dx and silently overflows int32 at another).
const NODE_BUDGET := 1.0e7

class ChunkRegion:
	# Grid extent in CELLS and its world-cell origin. dims is a multiple of n_grid per axis and
	# covers the bounding box of `chunks`, so more than one chunk can live in one grid.
	var dims: Vector3i = Vector3i.ZERO
	var origin: Vector3i = Vector3i.ZERO
	# The chunks this region was built for (snapshot of the active set at build time).
	var chunks: Array[Vector3i] = []
	# Cheap fingerprint of `chunks`, so _ensure_region() can tell "same chunk set" from "same
	# count but different members" without comparing arrays (Vector3i has no natural order to
	# sort by). Mismatch on either dims/origin or signature forces a rebuild.
	var signature: int = 0

	var n_particles: int = 0
	var groups_particles: int = 0
	var groups_grid: int = 0
	var idle_ticks: int = 0

	var params_buf: RID
	var pos: RID; var vel: RID; var C: RID; var J: RID
	var mat: RID; var mass: RID; var alive: RID; var F: RID
	var g_mom: RID; var g_mass: RID; var g_vel: RID; var g_mass_mat: RID
	var voxel_params_buf: RID; var voxel_buf_a: RID; var chunk_table: RID; var dirty_chunks_buf: RID
	# 6-bit "material touched this face plane" mask written by readout.comp, downloaded each tick
	# and consumed by _growth_for(). Reset per readout.
	var face_mask_buf: RID
	var face_mask: int = 0
	# 1 bit per chunk of this region's grid: does the chunk contain material? Written by
	# readout.comp, downloaded each tick, consumed by _growth_for() for growth and by the Y dump.
	var occupancy_buf: RID
	var occupied: Dictionary = {}  # Vector3i chunk coord -> true

	var set_clear: RID; var set_p2g: RID; var set_grid_op: RID; var set_g2p: RID; var set_readout: RID

# The one live region, or null when nothing is being simulated.
var _region: ChunkRegion = null
# Chunks that currently WANT to be simulated. Region geometry is always derived from this set,
# never edited directly — _ensure_region() is the only thing that builds grids.
var _active_chunks: Dictionary = {}  # Vector3i -> true
var _warned_budget := false
var _warned_region_cap := false
var _warned_capacity := false
# Ticks to wait between window moves. A rebuild reallocates every grid and particle buffer, so a
# centroid hovering on a chunk boundary must not trigger one per tick.
var _recenter_cooldown := 0
# Substep dt actually uploaded to the shaders; recomputed from sim_seconds_per_tick.
var sim_dt: float = 1.0e-4
var _applied_sim_dt: float = -1.0

# Shared across the region (one grid resolution).
var dx: float
var inv_dx: float
var p_vol: float
var fp_scale: float

var _material_db: MaterialDB
var _compute
var rd: RenderingDevice
var _orchestrator: WorldOrchestrator
var _chunk_manager: ChunkManager

var _clear: Dictionary
var _p2g: Dictionary
var _grid_op: Dictionary
var _g2p: Dictionary
var _readout: Dictionary

var is_setup: bool = false
# Why setup() failed, if it did — surfaced by world.gd's Y dump so a dead MPM says which shader
# or which precondition, instead of the misleading "not bound".
var setup_error: String = ""


func _ready() -> void:
	_material_db = get_node("/root/world/MaterialDB")
	_compute = preload("res://Utils/compute_helper.gd").new()
	rd = _compute.rd
	if not _compute.has_device():
		push_error("MPMSimulation: no RenderingDevice available")
		return
	# MPMSimulation sits under WorldOrchestrator/SimulationRuntime, which fully readies
	# before MaterialDB (a later sibling under world) even starts — same ordering hazard
	# SimulationRuntime._ready() already hits and fixes with call_deferred().
	call_deferred("_verify_material_table")


func _verify_material_table() -> void:
	if not _material_db.mpm_buffer_rid.is_valid():
		push_error("MPMSimulation: MaterialDB.mpm_buffer_rid not built — check init order")


# Compiles shared shader pipelines, resolves the real WorldOrchestrator/ChunkManager, and
# connects to the signals that drive chunk activation. Call once. Allocates NO grid buffers
# here — those come from _ensure_region() as chunks actually need them.
func setup() -> bool:
	setup_error = ""
	dx = 1.0
	inv_dx = 1.0
	_refresh_scales()

	_clear   = _compute.compile_shader(_load_shader("res://shaders/mpm/clear_grid.comp"))
	_p2g     = _compute.compile_shader(_load_shader("res://shaders/mpm/p2g.comp"))
	_grid_op = _compute.compile_shader(_load_shader("res://shaders/mpm/grid_op.comp"))
	_g2p     = _compute.compile_shader(_load_shader("res://shaders/mpm/g2p.comp"))
	_readout = _compute.compile_shader(_load_shader("res://shaders/mpm/readout.comp"))
	var failed: Array[String] = []
	if _clear.is_empty():   failed.append("clear_grid.comp")
	if _p2g.is_empty():     failed.append("p2g.comp")
	if _grid_op.is_empty(): failed.append("grid_op.comp")
	if _g2p.is_empty():     failed.append("g2p.comp")
	if _readout.is_empty(): failed.append("readout.comp")
	if not failed.is_empty():
		# Name the file(s). A shader compile failure only surfaces as "MPM is not running",
		# which reads like a solver-selection problem and costs a debugging round trip; the
		# actual GLSL error is push_error'd by compile_shader() and got lost in scrollback
		# (2026-08-13: a forward reference to `fill` in readout.comp did exactly this).
		setup_error = "shader compile failed: " + ", ".join(failed)
		push_error("MPMSimulation: " + setup_error)
		return false

	var orch: WorldOrchestrator = get_node("/root/world/WorldOrchestrator")
	var cm: ChunkManager = get_node("/root/world/ChunkManager")
	if not orch.is_grid_valid():
		setup_error = "WorldOrchestrator grid not valid yet"
		push_error("MPMSimulation.setup(): " + setup_error)
		return false
	_orchestrator = orch
	_chunk_manager = cm

	if not orch.voxels_placed.is_connected(_on_voxels_placed):
		orch.voxels_placed.connect(_on_voxels_placed)
	if not orch.chunk_loaded.is_connected(_on_chunk_loaded):
		orch.chunk_loaded.connect(_on_chunk_loaded)
	# R-key regenerate (player_controller) calls ChunkManager.regenerate_chunks(), which emits
	# slab_invalidated -> WorldOrchestrator._init_grid(). That FREES AND RECREATES
	# _voxel_buf_a, _chunk_table and _changelog_buf with new RIDs, and a live region binds all
	# three directly (w.voxel_buf_a / w.chunk_table and the readout uniform set). Without this
	# the region keeps pointing at freed handles and readout writes stale particles into a dead
	# buffer. Tear the region down whenever the grid is rebuilt.
	if not cm.slab_invalidated.is_connected(Callable(self, "reset")):
		cm.slab_invalidated.connect(Callable(self, "reset"))

	is_setup = true
	# Push exported water physics into the material table, so values set in the Inspector before
	# launch are honoured (the setters intentionally no-op until is_setup).
	_apply_water_override(false)
	scan_loaded_chunks_for_existing_material()
	return true


func is_bound() -> bool:
	return is_instance_valid(_orchestrator)


# Is there live material being simulated right now? (False when nothing activated, or the region
# has slept.) Used so perf averages don't dilute real ticks with idle ones.
func is_simulating() -> bool:
	return _region != null and _region.n_particles > 0


# Throw the region away and rebuild it from voxels on the next tick, keeping the requested set.
# Used when something OUTSIDE the simulation state changes what the shaders must bind — e.g. a
# material parameter edit rebuilds MaterialDB's mpm buffer, which would leave the region's uniform
# sets pointing at a freed handle.
func hard_rebuild() -> void:
	_destroy_region()


# Step the timescale ladder. Returns a human-readable summary for the console.
func cycle_timescale(substeps: int) -> String:
	_timescale_index = (_timescale_index + 1) % TIMESCALE_LADDER.size()
	sim_seconds_per_tick = TIMESCALE_LADDER[_timescale_index]
	_applied_sim_dt = -1.0  # force a param re-upload on the next tick
	var dt := sim_seconds_per_tick / maxf(float(substeps), 1.0)
	return "timescale " + str(_timescale_index + 1) + " of " + str(TIMESCALE_LADDER.size()) \
		+ " — " + String.num(sim_seconds_per_tick, 4) + " s per tick, substep dt " \
		+ String.num(dt, 8) + " (1 = real time; lower = finer and slower)"


# Drop ALL simulation state: the region (and with it every particle), the active-chunk set, and
# every one-shot warning flag. Called when the world itself is rebuilt (R key -> slab_invalidated
# -> WorldOrchestrator._init_grid), because the region's buffers are bound to RIDs that get
# freed and recreated underneath it. Also usable as a manual hard reset.
func reset() -> void:
	_active_chunks.clear()
	_destroy_region()
	_warned_budget = false
	_warned_region_cap = false
	_warned_capacity = false
	_recenter_cooldown = 0
	_applied_sim_dt = -1.0


# --- chunk → cell helpers -------------------------------------------------------------

func _chunk_of(world_voxel: Vector3i) -> Vector3i:
	var s := n_grid
	return Vector3i(floori(float(world_voxel.x) / s), floori(float(world_voxel.y) / s), floori(float(world_voxel.z) / s))


func _region_local(world_voxel: Vector3i) -> Vector3:
	return Vector3(world_voxel - _region.origin) + Vector3(0.5, 0.5, 0.5)


func _is_mpm_material(mat_id: int) -> bool:
	if mat_id == 0:
		return false
	var def := _material_db.get_def(mat_id)
	return def != null and (def.mpm_model == MaterialDef.MPMModel.FLUID
		or def.mpm_model == MaterialDef.MPMModel.GRANULAR)


# --- activation (intent only; geometry follows) ---------------------------------------

# Player-independent discovery: checks EVERY currently loaded chunk's cheap, already-maintained
# active_simulation_mask() (Chunk.rebuild_counts_and_mask(), zero per-voxel scanning) for
# GRANULAR/LIQUID material and requests those chunks. Called once from setup() and again from
# _on_chunk_loaded() as new chunks stream in — so a persistent lake somewhere the player has
# never looked still gets discovered and simulated, matching "not player-dependent."
func scan_loaded_chunks_for_existing_material() -> void:
	const CANDIDATE_MASK := 1 | 2  # SIM_BITS[GRANULAR] | SIM_BITS[LIQUID]
	for key in _chunk_manager.chunks.keys():
		var cp: Vector3i = key
		if _active_chunks.has(cp):
			continue
		var chunk: Chunk = _chunk_manager.chunks[cp]
		if chunk.active_simulation_mask() & CANDIDATE_MASK != 0:
			_try_add_chunk(cp)


func _on_chunk_loaded(chunk_pos: Vector3i) -> void:
	if not is_setup or _active_chunks.has(chunk_pos):
		return
	var chunk: Chunk = _chunk_manager.chunks.get(chunk_pos)
	if not chunk:
		return
	const CANDIDATE_MASK := 1 | 2
	if chunk.active_simulation_mask() & CANDIDATE_MASK != 0:
		_try_add_chunk(chunk_pos)


# Would adding `cp` keep the region grid within MAX_REGION_NODES? Computed on the prospective
# bounding box, so a refusal leaves the existing region untouched.
func _would_fit(cp: Vector3i) -> bool:
	var bmin := cp
	var bmax := cp
	for key in _active_chunks:
		var c: Vector3i = key
		bmin = Vector3i(mini(bmin.x, c.x), mini(bmin.y, c.y), mini(bmin.z, c.z))
		bmax = Vector3i(maxi(bmax.x, c.x), maxi(bmax.y, c.y), maxi(bmax.z, c.z))
	var d := (bmax - bmin + Vector3i.ONE) * n_grid
	return int(d.x) * int(d.y) * int(d.z) <= MAX_REGION_NODES


# Request a chunk be simulated. Returns true if it is now part of the active set. Adding a
# chunk changes the region bounding box, so this rebuilds the region (and re-seeds particles
# from the voxel world) whenever the geometry actually changes.
func _try_add_chunk(cp: Vector3i, promote: bool = false) -> bool:
	if _active_chunks.has(cp):
		return true
	if _active_chunks.size() >= max_active_chunks:
		if not _warned_budget:
			push_warning("MPMSimulation: max_active_chunks=%d exhausted, dropping request at %s" % [max_active_chunks, cp])
			_warned_budget = true
		return false
	if not _would_fit(cp):
		if not _warned_region_cap:
			push_warning("MPMSimulation: region grid would exceed MAX_REGION_NODES=%d including %s — refusing to grow (far-apart active chunks are the cause; see class header's TRADEOFF)" % [MAX_REGION_NODES, cp])
			_warned_region_cap = true
		return false
	_active_chunks[cp] = true
	# NO incremental promotion here. readout writes every particle's voxels into the world each
	# tick, so appending a chunk's voxels as particles DUPLICATES everything already represented —
	# measured: the particle count climbed 5575 -> 8083 -> 10849 -> 28491 across a run that should
	# have conserved it, and each duplicate added mass, which pushed the body harder, which
	# activated more chunks. Fresh material is picked up by _build_region()'s promote when a region
	# is created from nothing; that is the only place it is safe.
	_ensure_region()
	return true


# Promote one chunk's pre-existing MPM material by APPENDING to the live region's particle set.
func _promote_chunk_voxels(cp: Vector3i) -> void:
	if _region == null:
		return
	var origin := cp * n_grid
	_orchestrator.sync_chunk_to_cpu(cp)  # block_data is stale-by-design (CLAUDE.md)
	var pos: Array[Vector3] = []
	var mats: Array[int] = []
	var mss: Array[float] = []
	var st := maxi(promote_stride, 1)
	var st_v := float(st)
	var st_mass := float(st * st * st)
	for ly in range(0, n_grid, st):
		for lz in range(0, n_grid, st):
			for lx in range(0, n_grid, st):
				var wp := origin + Vector3i(lx, ly, lz)
				var mat := _orchestrator.query_voxel_material(wp)
				if not _is_mpm_material(mat):
					continue
				var def := _material_db.get_def(mat)
				pos.append(Vector3(wp - _region.origin) + Vector3(0.5, 0.5, 0.5) * st_v)
				mats.append(mat)
				mss.append(def.mpm_density * p_vol * st_mass)
	_append_particles(_region, pos, mats, mss)


# FIXED-SIZE SLIDING WINDOW (2026-08-13). Replaces the growing bounding box entirely.
#
# Why the bbox had to go: growth had to be decided from contact with the region's own walls, and
# there is no local measurement that distinguishes "this fluid would keep going if the wall
# weren't there" from "this fluid is confined" — pressure against a wall IS pressure. Four
# criteria were tried (chunk changed / mass at plane / mass in band / outward velocity) and every
# one either never fired or ran away, because a pressurised fluid genuinely does reach every wall
# at genuine speed. Giving it a bigger box just gave it more room to spread into, which lit the
# next set of faces. That feedback loop is the "box of spaz", and it is not fixable by a better
# growth rule — only by not growing.
#
# So the window has a FIXED size (region_chunks) and TRANSLATES instead. Cost is decided up front
# and never changes; there is no growth policy, no release policy, and no resize-under-pressure.
# Material that leaves the window is written to voxels and frozen, which is the same deal Noita
# makes with unloaded chunks.
func _ensure_region() -> bool:
	if _active_chunks.is_empty():
		_destroy_region()
		return false
	var chunks: Array[Vector3i] = []
	for key in _active_chunks:
		var c: Vector3i = key
		chunks.append(c)

	var sig := _active_signature()

	var dims := region_chunks * n_grid
	var origin: Vector3i
	if _region == null:
		# FIRST BUILD — allocate only what the edit actually touched. region_chunks is a FLOOR, not
		# a fixed size: a single sphere inside one chunk used to allocate the whole 2x2x2 window
		# (262,144 nodes, ~2.4 ms) regardless, "in case" the material later spread. It now starts
		# at the edit's own extent and grows only on evidence (see _growth_for()).
		var bb := _bbox_chunks(chunks)
		var span := bb[1] - bb[0] + Vector3i.ONE
		var start := Vector3i(
			clampi(maxi(region_chunks.x, span.x), 1, max_axis_chunks),
			clampi(maxi(region_chunks.y, span.y), 1, max_axis_chunks),
			clampi(maxi(region_chunks.z, span.z), 1, max_axis_chunks))
		var pad := (start - span) / 2
		origin = (bb[0] - pad) * n_grid
		dims = start * n_grid
	else:
		# EXISTING REGION — either GROW toward material at a face, or SHIFT one chunk toward it.
		if _region.occupied.is_empty():
			return false
		if grow_window:
			# Material is touching a face: that chunk becomes part of the window. BOUNDED —
			# max_axis_chunks per axis and MAX_REGION_NODES overall — which is the guard that was
			# missing from the original growth rules. Those ran away because they had no ceiling
			# and kept granting the fluid more room; this one stops at a fixed budget.
			var g := _growth_for(_region)
			var gmin: Vector3i = g[0]
			var gmax: Vector3i = g[1]
			if gmin == Vector3i.ZERO and gmax == Vector3i.ZERO:
				return false
			origin = _region.origin - gmin * n_grid
			dims = _region.dims + (gmin + gmax) * n_grid
		else:
			var shift := _edge_shift(_region)
			if shift == Vector3i.ZERO:
				return false
			origin = _region.origin + shift * n_grid

	if _region != null and _region.dims == dims and _region.origin == origin:
		if _region.signature == sig:
			return false
		_region.chunks = chunks
		_region.signature = sig
		return false

	# Geometry changed. PRESERVE the particle set across the move instead of re-seeding it from
	# voxels: velocities, the affine matrix C, the volume ratio J and sand's deformation gradient
	# F all survive, and a translation is exact (the offset is a whole number of chunks).
	var prev := _region
	_region = null
	var w := _build_region(origin, dims, chunks, prev)
	w.signature = sig
	# A move must not inherit the previous region's idle streak: after rebuilding, the first
	# readout often writes nothing (the voxels already match), which looked exactly like "settled"
	# and put the region to sleep four ticks later — the reported "chunks freeze before the sphere
	# has even fallen".
	w.idle_ticks = 0
	_region = w
	if prev != null:
		_free_region_buffers(prev)
	return true


# One chunk of growth per face that the material is currently touching, subject to
# max_axis_chunks and MAX_REGION_NODES. Returns [grow_min, grow_max] as per-axis 0/1 flags.
#
# This is the dynamic-local-allocation model restored, but with the ceiling the original lacked:
# growth can never exceed a fixed node budget, so "give the fluid more room" is bounded rather
# than an invitation to expand forever.
func _growth_for(w: ChunkRegion) -> Array:
	var none := Vector3i.ZERO
	var wmin := Vector3i(w.origin.x / n_grid, w.origin.y / n_grid, w.origin.z / n_grid)
	var wmax := wmin + (w.dims / n_grid) - Vector3i.ONE
	var bb := _bbox_chunks_from(w.occupied)
	var omin := bb[0]
	var omax := bb[1]
	var gmin := Vector3i.ZERO
	var gmax := Vector3i.ZERO
	if omin.x <= wmin.x: gmin.x = 1
	if omax.x >= wmax.x: gmax.x = 1
	if omin.y <= wmin.y: gmin.y = 1
	if omax.y >= wmax.y: gmax.y = 1
	if omin.z <= wmin.z: gmin.z = 1
	if omax.z >= wmax.z: gmax.z = 1
	if gmin == Vector3i.ZERO and gmax == Vector3i.ZERO:
		return [none, none]
	# Per-axis ceiling.
	var cap := max_axis_chunks * n_grid
	var d := w.dims + (gmin + gmax) * n_grid
	if d.x > cap: gmin.x = 0; gmax.x = 0
	if d.y > cap: gmin.y = 0; gmax.y = 0
	if d.z > cap: gmin.z = 0; gmax.z = 0
	d = w.dims + (gmin + gmax) * n_grid
	if int(d.x) * int(d.y) * int(d.z) > MAX_REGION_NODES:
		return [none, none]
	return [gmin, gmax]


# ONE CHUNK toward whichever side has material at (or past) the window edge. A TRANSLATION, never
# a re-centring, so it can strand at most one chunk's worth of material instead of the whole body.
#
# An axis is skipped when the material already spans the full window in it: the same water then
# touches BOTH edges and shifting would ratchet the window back and forth forever.
func _edge_shift(w: ChunkRegion) -> Vector3i:
	var wmin := Vector3i(w.origin.x / n_grid, w.origin.y / n_grid, w.origin.z / n_grid)
	var wmax := wmin + region_chunks - Vector3i.ONE
	var bb := _bbox_chunks_from(w.occupied)
	var omin := bb[0]
	var omax := bb[1]
	var span := omax - omin + Vector3i.ONE
	var shift := Vector3i.ZERO
	if span.x < region_chunks.x:
		if omin.x <= wmin.x: shift.x = -1
		elif omax.x >= wmax.x: shift.x = 1
	if span.y < region_chunks.y:
		if omin.y <= wmin.y: shift.y = -1
		elif omax.y >= wmax.y: shift.y = 1
	if span.z < region_chunks.z:
		if omin.z <= wmin.z: shift.z = -1
		elif omax.z >= wmax.z: shift.z = 1
	return shift


func _bbox_chunks_from(src: Dictionary) -> Array[Vector3i]:
	var bmin := Vector3i(2147483647, 2147483647, 2147483647)
	var bmax := Vector3i(-2147483648, -2147483648, -2147483648)
	for key in src:
		var c: Vector3i = key
		bmin = Vector3i(mini(bmin.x, c.x), mini(bmin.y, c.y), mini(bmin.z, c.z))
		bmax = Vector3i(maxi(bmax.x, c.x), maxi(bmax.y, c.y), maxi(bmax.z, c.z))
	if bmin.x > bmax.x:
		return [Vector3i.ZERO, Vector3i.ZERO]
	return [bmin, bmax]


func _bbox_chunks(arr: Array[Vector3i]) -> Array[Vector3i]:
	var d: Dictionary = {}
	for c in arr:
		d[c] = true
	return _bbox_chunks_from(d)


# Order-independent fingerprint of the active chunk set (positional hash combine).
func _active_signature() -> int:
	var h := 0
	for key in _active_chunks:
		var c: Vector3i = key
		h = (h * 73856093) ^ (int(c.x) * 19349663) ^ (int(c.y) * 83492791) ^ (int(c.z) * 2654435761)
	return h


func _build_region(origin: Vector3i, dims: Vector3i, chunks: Array[Vector3i], prev: ChunkRegion = null) -> ChunkRegion:
	_refresh_scales()  # pick up any promote_stride / particle_subdiv change
	var w := ChunkRegion.new()
	w.origin = origin
	w.dims = dims
	w.chunks = chunks

	var nodes := int(dims.x) * int(dims.y) * int(dims.z)
	w.groups_grid = int(ceil(float(nodes) / 64.0))
	w.g_mom      = _new_zero_buf(nodes * 3 * 4)
	w.g_mass     = _new_zero_buf(nodes * 4)
	w.g_vel      = _new_zero_buf(nodes * 3 * 4)
	w.g_mass_mat = _new_zero_buf(nodes * NMAT * 4)
	w.params_buf = _new_zero_buf(48)
	w.dirty_chunks_buf = _new_zero_buf(DIRTY_BUF_BYTES)
	w.face_mask_buf = _new_zero_buf(4)
	w.occupancy_buf = _new_zero_buf(OCCUPANCY_BYTES)

	w.voxel_params_buf = _new_zero_buf(48)
	var b := PackedByteArray(); b.resize(48)
	b.encode_s32(0, _chunk_manager.chunk_size)
	var gmin := _orchestrator.get_grid_min(); var gspan := _orchestrator.get_grid_span()
	b.encode_s32(4, gmin.x); b.encode_s32(8, gmin.y); b.encode_s32(12, gmin.z)
	b.encode_s32(16, gspan.x); b.encode_s32(20, gspan.y); b.encode_s32(24, gspan.z)
	b.encode_s32(28, origin.x); b.encode_s32(32, origin.y); b.encode_s32(36, origin.z)
	b.encode_s32(40, 1)  # voxel_world_valid
	_compute.buffer_update(w.voxel_params_buf, 0, b)
	w.voxel_buf_a = _orchestrator.get_voxel_buf()
	w.chunk_table = _orchestrator.get_chunk_table()

	var cap := max_particles
	w.pos   = _new_zero_buf(cap * 3 * 4)
	w.vel   = _new_zero_buf(cap * 3 * 4)
	w.C     = _new_zero_buf(cap * 9 * 4)
	w.J     = _new_zero_buf(cap * 4)
	w.mat   = _new_zero_buf(cap * 4)
	w.mass  = _new_zero_buf(cap * 4)
	w.alive = _new_zero_buf(cap * 4)
	w.F     = _new_zero_buf(cap * 9 * 4)

	_upload_region_params(w)
	_build_region_sets(w)
	# Fresh region: seed from whatever MPM material already exists in these chunks. Resize:
	# carry the live particles over instead, so nothing is reset.
	if prev != null:
		_copy_particles_shifted(prev, w)
	else:
		promote_existing_voxels(w, chunks)
	return w


# Moves a live particle set from one region's buffers into another's, shifting positions by the
# difference in grid origin (positions are region-local). Only POSITIONS need the offset, so
# they go through the CPU; everything else is a straight GPU→GPU copy with no round trip.
#
# Position precision note: the offset is an exact multiple of n_grid, so this is exact for any
# float position — no accumulation error, no drift.
func _copy_particles_shifted(prev: ChunkRegion, w: ChunkRegion) -> void:
	var n := prev.n_particles
	if n <= 0:
		w.n_particles = 0
		w.groups_particles = 0
		return
	var shift := Vector3(prev.origin - w.origin)
	var pos: PackedFloat32Array = _compute.download(prev.pos).to_float32_array()
	# A particle left OUTSIDE the new box by a shrink is killed rather than left to drift: g2p
	# clamps every position into the region, so an outside particle would be snapped onto the
	# boundary wall and reappear as material teleported to the edge. Readout already wrote its
	# voxels before the chunk was released, so nothing is lost visually — it is simply frozen as
	# static voxels until something disturbs it again.
	# Particles must never be KILLED here. Clamp them back inside instead: killing loses material
	# permanently (the voxels are stale, and nothing re-activates a chunk that no longer changes),
	# which is what left an intact water sphere hanging motionless in mid-air. With occupancy-based
	# release the box should not shrink over material at all, so this is a safety net — but a
	# safety net that preserves mass rather than silently deleting it.
	var alive := PackedInt32Array(); alive.resize(n)
	var limit := Vector3(w.dims) - Vector3(1e-4, 1e-4, 1e-4)
	var outside := 0
	for i in n:
		var px: float = pos[i*3]     + shift.x
		var py: float = pos[i*3 + 1] + shift.y
		var pz: float = pos[i*3 + 2] + shift.z
		# DROP, do not clamp. Clamping teleported them onto whatever wall was nearest, which is
		# where the flat-axis streaks came from. A particle outside the window becomes frozen
		# voxels, which is the intended deal — but with the padding fix it should be rare, because
		# the window is now placed to contain the material rather than rounded around its midpoint.
		if px < 0.0 or py < 0.0 or pz < 0.0 or px > limit.x or py > limit.y or pz > limit.z:
			alive[i] = 0
			outside += 1
			continue
		pos[i*3] = px; pos[i*3 + 1] = py; pos[i*3 + 2] = pz
		alive[i] = 1
	_compute.buffer_update(w.pos, 0, pos.to_byte_array())
	_compute.buffer_update(w.alive, 0, alive.to_byte_array())
	_compute.buffer_copy(prev.vel,   w.vel,   0, 0, n * 3 * 4)
	_compute.buffer_copy(prev.C,     w.C,     0, 0, n * 9 * 4)
	_compute.buffer_copy(prev.J,     w.J,     0, 0, n * 4)
	_compute.buffer_copy(prev.mat,   w.mat,   0, 0, n * 4)
	_compute.buffer_copy(prev.mass,  w.mass,  0, 0, n * 4)
	_compute.buffer_copy(prev.F,     w.F,     0, 0, n * 9 * 4)
	if outside > 0:
		print("[MPM] window moved: %d particle(s) left the window and are frozen as voxels" % outside)
	w.n_particles = n
	w.groups_particles = int(ceil(float(n) / 64.0))
	_upload_region_params(w)


func _destroy_region() -> void:
	var w := _region
	_region = null
	if w:
		_free_region_buffers(w)


func _free_region_buffers(w: ChunkRegion) -> void:
	for rid in [w.params_buf, w.pos, w.vel, w.C, w.J, w.mat, w.mass, w.alive, w.F,
		w.g_mom, w.g_mass, w.g_vel, w.g_mass_mat,
		w.voxel_params_buf, w.dirty_chunks_buf, w.face_mask_buf, w.occupancy_buf]:
		if rid.is_valid():
			_compute.free_rid(rid)


# Seeds particles from whatever FLUID/GRANULAR-modeled voxels currently exist in the given
# chunks, so a newly-activated region isn't just the one voxel that triggered it — a whole
# pre-existing pond is picked up too, including ponds in the NEIGHBOURING chunks that share
# this grid (which is the point: they're one body of water now).
func promote_existing_voxels(w: ChunkRegion, chunks: Array[Vector3i]) -> int:
	var st := maxi(promote_stride, 1)
	var sub := maxi(particle_subdiv, 1)
	var pos: Array[Vector3] = []
	var mats: Array[int] = []
	var mss: Array[float] = []
	var st_v := float(st)
	for cp in chunks:
		var origin := cp * n_grid
		_orchestrator.sync_chunk_to_cpu(cp)  # block_data is stale-by-design (CLAUDE.md)
		for ly in range(0, n_grid, st):
			for lz in range(0, n_grid, st):
				for lx in range(0, n_grid, st):
					var wp := origin + Vector3i(lx, ly, lz)
					var mat := _orchestrator.query_voxel_material(wp)
					if not _is_mpm_material(mat):
						continue
					var def := _material_db.get_def(mat)
					# p_vol already carries the stride^3 / subdiv^3 factors (see _refresh_scales),
					# so this is the particle's own mass with no extra scaling.
					var mass_each := def.mpm_density * p_vol
					for sz in sub:
						for sy in sub:
							for sx in sub:
								var off := (Vector3(sx, sy, sz) + Vector3(0.5, 0.5, 0.5)) * (st_v / float(sub))
								pos.append(Vector3(wp - w.origin) + off)
								mats.append(mat)
								mss.append(mass_each)

	var n := pos.size()
	if n == 0:
		return 0
	seed_particles(w, n, func(i: int) -> Dictionary:
		return {"pos": pos[i], "mat": mats[i], "mass": mss[i]}
	)
	return n


func _load_shader(path: String) -> String:
	var f := FileAccess.open(path, FileAccess.READ)
	return f.get_as_text()


func _new_zero_buf(size: int) -> RID:
	var b := PackedByteArray()
	b.resize(size)
	b.fill(0)
	return _compute.buffer_upload(b)


# seed_fn(i) -> Dictionary { pos: Vector3 (LOCAL to the REGION origin, voxel units),
# mat: int, mass: float }
func seed_particles(w: ChunkRegion, n: int, seed_fn: Callable) -> void:
	if n > max_particles:
		push_error("MPMSimulation.seed_particles(): n=%d exceeds max_particles=%d" % [n, max_particles])
		n = max_particles
	w.n_particles = n
	w.groups_particles = int(ceil(float(n) / 64.0)) if n > 0 else 0

	var pos := PackedFloat32Array(); pos.resize(n * 3)
	var vel := PackedFloat32Array(); vel.resize(n * 3)
	var cc  := PackedFloat32Array(); cc.resize(n * 9)
	var jj  := PackedFloat32Array(); jj.resize(n)
	var mt  := PackedInt32Array();   mt.resize(n)
	var ms  := PackedFloat32Array(); ms.resize(n)
	var al  := PackedInt32Array();   al.resize(n)
	var ff  := PackedFloat32Array(); ff.resize(n * 9)
	for i in n:
		var d: Dictionary = seed_fn.call(i)
		var pp: Vector3 = d["pos"]
		pos[i*3] = pp.x; pos[i*3+1] = pp.y; pos[i*3+2] = pp.z
		jj[i] = 1.0
		mt[i] = int(d.get("mat", 0))
		ms[i] = float(d["mass"])
		al[i] = 1
		ff[i*9] = 1.0; ff[i*9+4] = 1.0; ff[i*9+8] = 1.0  # F = identity

	_compute.buffer_update(w.pos, 0, pos.to_byte_array())
	_compute.buffer_update(w.vel, 0, vel.to_byte_array())
	_compute.buffer_update(w.C, 0, cc.to_byte_array())
	_compute.buffer_update(w.J, 0, jj.to_byte_array())
	_compute.buffer_update(w.mat, 0, mt.to_byte_array())
	_compute.buffer_update(w.mass, 0, ms.to_byte_array())
	_compute.buffer_update(w.alive, 0, al.to_byte_array())
	_compute.buffer_update(w.F, 0, ff.to_byte_array())
	_upload_region_params(w)


# Continuous promotion: appends ONE particle into the region's unused tail capacity — a
# buffer_update() write at index n_particles plus incrementing it, NOT a reseed. No pipeline/
# uniform-set rebuild needed. Returns false if max_particles is exhausted.
func add_particle(w: ChunkRegion, local_pos: Vector3, mat_id: int, mass: float) -> bool:
	if w.n_particles >= max_particles:
		push_warning("MPMSimulation.add_particle(): max_particles=%d exhausted" % max_particles)
		return false
	var i: int = w.n_particles

	_compute.buffer_update(w.pos, i * 3 * 4, PackedFloat32Array([local_pos.x, local_pos.y, local_pos.z]).to_byte_array())
	_compute.buffer_update(w.vel, i * 3 * 4, PackedFloat32Array([0.0, 0.0, 0.0]).to_byte_array())
	_compute.buffer_update(w.C, i * 9 * 4, PackedFloat32Array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0]).to_byte_array())
	_compute.buffer_update(w.J, i * 4, PackedFloat32Array([1.0]).to_byte_array())
	_compute.buffer_update(w.mat, i * 4, PackedInt32Array([mat_id]).to_byte_array())
	_compute.buffer_update(w.mass, i * 4, PackedFloat32Array([mass]).to_byte_array())
	_compute.buffer_update(w.alive, i * 4, PackedInt32Array([1]).to_byte_array())
	_compute.buffer_update(w.F, i * 9 * 4, PackedFloat32Array([1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]).to_byte_array())

	w.n_particles += 1
	w.groups_particles = int(ceil(float(w.n_particles) / 64.0))
	_upload_region_params(w)
	return true


# EditDispatcher/WorldOrchestrator.voxels_placed handler. edit_apply.comp writes straight into
# _voxel_buf_a, so by the time this fires the placed voxels ARE in the voxel world — meaning a
# chunk that has to (re)activate already gets them via promote_existing_voxels(), and calling
# add_particle() on top of that would insert a duplicate. Only voxels landing in a chunk that
# was ALREADY part of the region need the incremental append.
func _on_voxels_placed(positions: Array[Vector3i], material_ids: Array[int]) -> void:
	if not is_setup:
		return

	# PASS 1 — resolve every chunk this edit touches BEFORE any geometry work. A sphere across
	# a chunk boundary used to add one chunk at a time, and each addition rebuilt the region and
	# re-ran the full promotion scan (a whole-chunk GPU readback + 32k voxel lookups). One edit
	# spanning N chunks therefore cost N rebuilds — the stutter reported on every sphere drop.
	var new_chunks: Array[Vector3i] = []
	for idx in positions.size():
		if not _is_mpm_material(material_ids[idx]):
			continue
		var cp := _chunk_of(positions[idx])
		if _active_chunks.has(cp) or new_chunks.has(cp):
			continue
		new_chunks.append(cp)

	var had_region := _region != null
	var grew := false
	for cp in new_chunks:
		if _active_chunks.size() >= max_active_chunks:
			if not _warned_budget:
				push_warning("MPMSimulation: max_active_chunks=%d exhausted — %d chunk(s) from this edit left unsimulated (they will act as walls until a slot frees)." % [max_active_chunks, new_chunks.size()])
				_warned_budget = true
			break
		if not _would_fit(cp):
			if not _warned_region_cap:
				push_warning("MPMSimulation: region grid would exceed MAX_REGION_NODES=%d including %s — refusing to grow." % [MAX_REGION_NODES, cp])
				_warned_region_cap = true
			break
		_active_chunks[cp] = true
		grew = true
	if grew:
		_ensure_region()  # exactly ONE resize for the whole edit; live particles are preserved

	# PASS 2 — append the placed voxels. If the region was just CREATED from nothing, its
	# _build_region() promoted every voxel already in these chunks (edit_apply.comp writes
	# _voxel_buf_a before this signal fires), including the ones just placed — appending on top
	# would duplicate them. A RESIZE instead preserves the existing particle set and promotes
	# nothing, so the new voxels do need appending.
	if _region == null or not had_region:
		return
	var add_pos: Array[Vector3] = []
	var add_mat: Array[int] = []
	var add_mass: Array[float] = []
	var sub := maxi(particle_subdiv, 1)
	for idx in positions.size():
		var mat_id: int = material_ids[idx]
		if not _is_mpm_material(mat_id):
			continue
		var wp: Vector3i = positions[idx]
		var cp := _chunk_of(wp)
		# "Inside the grid" is the right test, not "in the active set": the region's bounding
		# box can cover a chunk that was never explicitly activated, and readout already writes
		# into those (world_to_flat only needs the chunk to be loaded).
		if not _region_covers(cp):
			continue
		var def := _material_db.get_def(mat_id)
		var local := _region_local(wp) - Vector3(0.5, 0.5, 0.5)  # voxel corner, for subdivision
		var mass_each := def.mpm_density * p_vol  # p_vol already divided by subdiv^3
		for sz in sub:
			for sy in sub:
				for sx in sub:
					add_pos.append(local + (Vector3(sx, sy, sz) + Vector3(0.5, 0.5, 0.5)) / float(sub))
					add_mat.append(mat_id)
					add_mass.append(mass_each)
	_append_particles(_region, add_pos, add_mat, add_mass)


# Is this chunk inside the region's bounding box (chunk coordinates)?
func _region_covers(cp: Vector3i) -> bool:
	if _region == null:
		return false
	var bmin := Vector3i(_region.origin.x / n_grid, _region.origin.y / n_grid, _region.origin.z / n_grid)
	var span := Vector3i(_region.dims.x / n_grid, _region.dims.y / n_grid, _region.dims.z / n_grid)
	var bmax := bmin + span - Vector3i.ONE
	return (cp.x >= bmin.x and cp.x <= bmax.x
		and cp.y >= bmin.y and cp.y <= bmax.y
		and cp.z >= bmin.z and cp.z <= bmax.z)


# Bulk append — one buffer_update per ATTRIBUTE for the whole batch, instead of the 8 per
# particle that add_particle() costs (a 500-voxel sphere was 4000 individual GPU uploads, the
# other half of the placement stutter). add_particle() stays for single-voxel placement.
func _append_particles(w: ChunkRegion, local_pos: Array[Vector3], mats: Array[int], masses: Array[float]) -> int:
	var count := local_pos.size()
	if count == 0:
		return 0
	var room := max_particles - w.n_particles
	if count > room:
		push_warning("MPMSimulation: appending %d particle(s) but only %d slot(s) free — dropping the excess." % [count, maxi(room, 0)])
		count = room
		if count <= 0:
			return 0

	var base := w.n_particles
	var pos := PackedFloat32Array(); pos.resize(count * 3)
	var vel := PackedFloat32Array(); vel.resize(count * 3)
	var cc  := PackedFloat32Array(); cc.resize(count * 9)
	var jj  := PackedFloat32Array(); jj.resize(count)
	var mt  := PackedInt32Array();   mt.resize(count)
	var ms  := PackedFloat32Array(); ms.resize(count)
	var al  := PackedInt32Array();   al.resize(count)
	var ff  := PackedFloat32Array(); ff.resize(count * 9)
	for i in count:
		var p: Vector3 = local_pos[i]
		pos[i*3] = p.x; pos[i*3+1] = p.y; pos[i*3+2] = p.z
		jj[i] = 1.0
		mt[i] = mats[i]
		ms[i] = masses[i]
		al[i] = 1
		ff[i*9] = 1.0; ff[i*9+4] = 1.0; ff[i*9+8] = 1.0  # F = identity

	_compute.buffer_update(w.pos, base * 3 * 4, pos.to_byte_array())
	_compute.buffer_update(w.vel, base * 3 * 4, vel.to_byte_array())
	_compute.buffer_update(w.C, base * 9 * 4, cc.to_byte_array())
	_compute.buffer_update(w.J, base * 4, jj.to_byte_array())
	_compute.buffer_update(w.mat, base * 4, mt.to_byte_array())
	_compute.buffer_update(w.mass, base * 4, ms.to_byte_array())
	_compute.buffer_update(w.alive, base * 4, al.to_byte_array())
	_compute.buffer_update(w.F, base * 9 * 4, ff.to_byte_array())

	w.n_particles += count
	w.groups_particles = int(ceil(float(w.n_particles) / 64.0))
	_upload_region_params(w)
	return count


func _upload_region_params(w: ChunkRegion) -> void:
	var b := PackedByteArray(); b.resize(48)
	b.encode_s32(0, w.dims.x)
	b.encode_s32(4, w.dims.y)
	b.encode_s32(8, w.dims.z)
	b.encode_s32(12, w.n_particles)
	b.encode_float(16, sim_dt)
	b.encode_float(20, dx)
	b.encode_float(24, inv_dx)
	b.encode_float(28, gravity)
	b.encode_float(32, p_vol)
	b.encode_float(36, fp_scale)
	b.encode_s32(40, bound)
	b.encode_s32(44, 0)
	_compute.buffer_update(w.params_buf, 0, b)


func _u(binding: int, rid: RID) -> RDUniform:
	var un := RDUniform.new()
	un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	un.binding = binding
	un.add_id(rid)
	return un


func _build_region_sets(w: ChunkRegion) -> void:
	w.set_clear = _compute.create_uniform_set([
		_u(0, w.params_buf), _u(8, w.g_mom), _u(9, w.g_mass), _u(12, w.g_mass_mat)
	], _clear.shader)
	w.set_p2g = _compute.create_uniform_set([
		_u(0, w.params_buf), _u(1, w.pos), _u(2, w.vel), _u(3, w.C), _u(4, w.J), _u(5, w.mat),
		_u(6, w.mass), _u(7, w.alive), _u(8, w.g_mom), _u(9, w.g_mass), _u(11, w.F),
		_u(12, w.g_mass_mat), _u(13, _material_db.mpm_buffer_rid)
	], _p2g.shader)
	w.set_grid_op = _compute.create_uniform_set([
		_u(0, w.params_buf), _u(8, w.g_mom), _u(9, w.g_mass), _u(10, w.g_vel),
		_u(13, _material_db.mpm_buffer_rid), _u(14, w.voxel_params_buf),
		_u(15, w.voxel_buf_a), _u(16, w.chunk_table), _u(21, w.face_mask_buf)
	], _grid_op.shader)
	w.set_g2p = _compute.create_uniform_set([
		_u(0, w.params_buf), _u(1, w.pos), _u(2, w.vel), _u(3, w.C), _u(4, w.J), _u(5, w.mat),
		_u(7, w.alive), _u(10, w.g_vel), _u(11, w.F), _u(13, _material_db.mpm_buffer_rid)
	], _g2p.shader)
	w.set_readout = _compute.create_uniform_set([
		_u(0, w.params_buf), _u(9, w.g_mass), _u(12, w.g_mass_mat),
		_u(13, _material_db.mpm_buffer_rid), _u(14, w.voxel_params_buf),
		_u(15, w.voxel_buf_a), _u(16, w.chunk_table), _u(17, _orchestrator.get_changelog_buf()),
		_u(18, w.dirty_chunks_buf), _u(19, w.face_mask_buf), _u(20, w.occupancy_buf)
	], _readout.shader)


# Below this max speed (voxel-units/simulated-second), a particle counts as "at rest" for
# idle-detection purposes. Found empirically (2026-08-12): readout()'s voxel-level changed/
# unchanged signal was tried first and is TOO COARSE — a particle early in a gravity fall
# (measured max_speed ~0.12 after only ~0.012 simulated seconds, v=g*t) can occupy the same
# grid cell for several consecutive ticks without crossing a voxel boundary, producing
# identical readout output while genuinely still accelerating — which hit IDLE_THRESHOLD and
# deactivated chunks that were still actively falling, exactly the "waterfall stops" failure
# this whole design exists to prevent, just one layer deeper than player-position coupling.
# Velocity is the actual physical quantity that means "at rest," not a proxy for it.
@export var rest_speed: float = 0.02

# Public entry point — call ONCE per real simulation tick (SimulationRuntime._tick()). Runs the
# region's physics substeps + readout, then sleeps the whole region once it has been motionless
# for IDLE_THRESHOLD ticks. Idle is a REGION property now, not a per-chunk one: chunks share a
# grid, so one chunk going quiet while its neighbour still flows is not a reason to drop
# anything — that would tear the shared grid apart mid-simulation.
func tick_all(substeps_per_tick: int) -> void:
	if not is_bound():
		return
	# Rebuild on demand (hard_rebuild / material change): the region is created lazily, so a tick
	# that arrives with no region but a non-empty requested set must reconstruct it.
	if _region == null and not _active_chunks.is_empty():
		_ensure_region()
	var w := _region
	if w == null or w.n_particles == 0:
		return

	# Derive the substep dt from the single pacing knob (see sim_seconds_per_tick).
	var want_dt := sim_seconds_per_tick / maxf(float(substeps_per_tick), 1.0)
	if not is_equal_approx(want_dt, _applied_sim_dt):
		_applied_sim_dt = want_dt
		sim_dt = want_dt
		_upload_region_params(w)

	# The face-push mask is accumulated by grid_op across ALL substeps, so it is cleared here,
	# once per tick before them — not in the readout, which runs afterwards and only reads it.
	var zero_face := PackedByteArray(); zero_face.resize(4); zero_face.fill(0)
	_compute.buffer_update(w.face_mask_buf, 0, zero_face)
	w.face_mask = 0

	_run_region_substeps(w, substeps_per_tick)
	var changed := _readout_region(w)

	# REGION GEOMETRY UPDATE — the whole "which chunks are active" decision, once per tick.
	# _ensure_region() either GROWS the window toward material touching a face (grow_window) or
	# SHIFTS it one chunk that way. Rate-limited because a rebuild reallocates every grid and
	# particle buffer. If it returns true, `w` is stale.
	if _recenter_cooldown > 0:
		_recenter_cooldown -= 1
	elif _ensure_region():
		_recenter_cooldown = 10
		return  # the region was rebuilt; `w` is stale from here on

	# IDLE DETECTION — cheap signal first. An empty readout dirty-set means the grid wrote
	# nothing back this tick, which needs no extra GPU→CPU round trip because the bitmask was
	# already downloaded. The previous version called read_velocities() EVERY tick, which
	# downloads the whole particle position buffer (a per-tick stall) and requires max speed to
	# fall under REST_SPEED — a threshold MPM particles jitter above indefinitely, so the region
	# never slept. That is the "chunks stay active and I can't modify them" report. Velocity is
	# now only consulted once the voxel world has already gone quiet, as confirmation.
	if not changed.is_empty():
		w.idle_ticks = 0
		return

	w.idle_ticks += 1
	if w.idle_ticks < idle_threshold:
		return

	var max_speed := 0.0
	var vel := read_velocities(w)
	for i in w.n_particles:
		var spd := Vector3(vel[i*3], vel[i*3+1], vel[i*3+2]).length()
		max_speed = max(max_speed, spd)

	if max_speed < rest_speed:
		# Everything is at rest and already written back to voxels. Sleep the region and clear
		# the intent set; a later edit or a streamed chunk re-activates from voxels.
		_active_chunks.clear()
		_destroy_region()
	else:
		w.idle_ticks = 0


func _run_region_substeps(w: ChunkRegion, substeps: int) -> void:
	for i in substeps:
		_compute.run_with_set(_clear,   w.set_clear,   w.groups_grid)
		_compute.run_with_set(_p2g,     w.set_p2g,     w.groups_particles)
		_compute.run_with_set(_grid_op, w.set_grid_op, w.groups_grid)
		_compute.run_with_set(_g2p,     w.set_g2p,     w.groups_particles)
	_compute.submit()
	_compute.sync()


# Writes the region's current grid mass/material state into the REAL voxel world (shared
# changelog + WorldOrchestrator.apply_changelog() — same safe path GranularSimulation/
# FluidSimulation use). readout.comp maps each grid node through origin + local index and then
# through world_to_flat(), so nodes in any chunk of the region land in the right chunk slot.
# Does NOT emit WorldOrchestrator.chunks_committed (see the note in _on_voxels_placed's
# counterpart discussion in MIGRATION_MPM.md §2 — CA is off, nothing else consumes it yet).
func _readout_region(w: ChunkRegion) -> Array[Vector3i]:
	# CAPACITY INTERLOCK — the cause of "I can no longer add or modify those chunks".
	# readout writes AIR into any node whose grid mass is below threshold. A water voxel the
	# player just placed, in a region whose particle set is already full, has NO particle — so
	# readout would erase it on the very next tick, and every subsequent edit vanishes the same
	# way. When the particle set cannot represent the region, stop writing back instead of
	# corrupting the world.
	if w.n_particles >= max_particles:
		if not _warned_capacity:
			push_warning("MPMSimulation: max_particles=%d reached — readout SUSPENDED so it cannot erase voxels that have no particle. Raise max_particles or reduce active chunks." % max_particles)
			_warned_capacity = true
		var none: Array[Vector3i] = []
		return none

	var zero_dirty := PackedByteArray(); zero_dirty.resize(DIRTY_BUF_BYTES); zero_dirty.fill(0)
	_compute.buffer_update(w.dirty_chunks_buf, 0, zero_dirty)
	var zero_occ := PackedByteArray(); zero_occ.resize(OCCUPANCY_BYTES); zero_occ.fill(0)
	_compute.buffer_update(w.occupancy_buf, 0, zero_occ)
	_orchestrator.reset_changelog()

	_compute.run_with_set(_readout, w.set_readout, w.groups_grid)
	_compute.submit()
	_compute.sync()

	_orchestrator.apply_changelog()

	# 4 bytes — cheaper than the dirty bitmask download next to it.
	var face_raw: PackedByteArray = _compute.download_range(w.face_mask_buf, 0, 4)
	if face_raw.size() >= 4:
		w.face_mask = int(face_raw.decode_u32(0))
	w.occupied = _decode_occupancy(w, _compute.download_range(w.occupancy_buf, 0, OCCUPANCY_BYTES))
	var bitmask: PackedByteArray = _compute.download_range(w.dirty_chunks_buf, 0, DIRTY_BUF_BYTES)
	return _decode_dirty_bitmask(bitmask)


# Turn the occupancy bitmask into a set of chunk coords (world chunk space, same convention as
# _active_chunks) so growth can ask a direct question: does this chunk hold
# material? Bit index follows readout.comp's layout: (cz * dcx + cy) * dcx + cx over the region's
# own chunk grid.
func _decode_occupancy(w: ChunkRegion, raw: PackedByteArray) -> Dictionary:
	var out: Dictionary = {}
	var dcx := int(w.dims.x) / n_grid
	var dcy := int(w.dims.y) / n_grid
	var dcz := int(w.dims.z) / n_grid
	if dcx <= 0 or dcy <= 0 or dcz <= 0:
		return out
	var bmin := Vector3i(w.origin.x / n_grid, w.origin.y / n_grid, w.origin.z / n_grid)
	var words := raw.size() / 4
	for wi in range(words):
		var word: int = raw.decode_u32(wi * 4)
		if word == 0:
			continue
		for bit in range(32):
			if word & (1 << bit) == 0:
				continue
			var idx: int = wi * 32 + bit
			var cz: int = idx / (dcx * dcy)
			var rem: int = idx % (dcx * dcy)
			var cy: int = rem / dcx
			var cx: int = rem % dcx
			if cz >= dcz:
				continue
			out[bmin + Vector3i(cx, cy, cz)] = true
	return out


# Ported from GranularSimulation._decode_dirty_bitmask() — same bit layout, same reach into
# WorldOrchestrator's private _slot_to_chunk (GDScript doesn't enforce real privacy; matches
# the existing convention rather than inventing a public accessor unprompted).
func _decode_dirty_bitmask(bitmask: PackedByteArray) -> Array[Vector3i]:
	var slot_to_chunk = _orchestrator._slot_to_chunk
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


# --- Diagnostics / test accessors ---

# The region now spans multiple chunks, so this returns the region if `chunk_pos` is covered by
# it, else null. Kept under the old name so existing callers keep working; n_particles on the
# returned object is REGION-wide, not that chunk's share.
func get_chunk_region(chunk_pos: Vector3i) -> ChunkRegion:
	if _region == null:
		return null
	return _region if _region.chunks.has(chunk_pos) else null


func active_chunk_positions() -> Array:
	# Chunks the region's GRID actually covers — every chunk being simulated. This is NOT
	# _active_chunks (the set that was explicitly requested): the grid is the BOUNDING BOX of
	# that set, so it normally covers MORE chunks. That gap is exactly why the debug tint looked
	# like it only appeared under edits — a box-covered chunk is fully simulated whether or not
	# anything ever explicitly asked for it.
	if _region == null:
		return []
	var bmin := Vector3i(_region.origin.x / n_grid, _region.origin.y / n_grid, _region.origin.z / n_grid)
	var span := Vector3i(_region.dims.x / n_grid, _region.dims.y / n_grid, _region.dims.z / n_grid)
	var out: Array = []
	for y in range(span.y):
		for z in range(span.z):
			for x in range(span.x):
				out.append(bmin + Vector3i(x, y, z))
	return out


# The explicitly-requested set (edits / boot scan / growth). Separate from grid coverage because
# it is what drives the chunk budget and the growth decisions, not what the overlay should show.
func requested_chunk_positions() -> Array:
	return _active_chunks.keys()


func region_grid_info() -> Dictionary:
	if _region == null:
		return {}
	return {
		"origin": _region.origin,
		"dims": _region.dims,
		"nodes": _region.dims.x * _region.dims.y * _region.dims.z,
		"chunks": _region.chunks.size(),
		"requested": _active_chunks.size(),
		"particles": _region.n_particles,
		# Exposed because "is it going to sleep?" is the question this dump exists to answer:
		# the region sleeps after IDLE_THRESHOLD consecutive ticks with an empty readout
		# dirty-set AND a max particle speed under REST_SPEED.
		"idle_ticks": _region.idle_ticks,
		"capacity": max_particles,
		# Which boundary planes currently hold material (bit 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z).
		# The single most useful field for "why isn't the region following the water".
		"faces": _region.face_mask,
		"occupied": _region.occupied.size(),
		# Window extent in CHUNK coords, so the dump shows where the box actually is relative to
		# wherever material was dropped.
		"window_min": Vector3i(_region.origin.x / n_grid, _region.origin.y / n_grid, _region.origin.z / n_grid),
		"window_max": Vector3i(_region.origin.x / n_grid, _region.origin.y / n_grid, _region.origin.z / n_grid) + region_chunks - Vector3i.ONE,
		"region_chunks": region_chunks,
		"occupied_list": _region.occupied.keys(),
	}


func read_positions(w: ChunkRegion) -> PackedFloat32Array:
	return _compute.download(w.pos).to_float32_array()


func read_velocities(w: ChunkRegion) -> PackedFloat32Array:
	return _compute.download(w.vel).to_float32_array()


func read_materials(w: ChunkRegion) -> PackedInt32Array:
	return _compute.download(w.mat).to_int32_array()
