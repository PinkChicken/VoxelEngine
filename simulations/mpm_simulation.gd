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
# ACTIVE-CHUNK MODEL (2026-08-12, revised after user pushback on an earlier per-player-window
# design — "the simulation shouldn't be player-dependent; a waterfall behind the player
# shouldn't stop"): reuses the EXACT pattern GranularSimulation/FluidSimulation already use —
# _active_chunks: Dictionary[Vector3i, idle_ticks], a chunk goes active when disturbed, sleeps
# after 4 consecutive idle ticks (GranularSimulation's own threshold, matched here) — rather
# than inventing a separate "window near the player" concept. Each active chunk gets its OWN
# MPM grid+particle buffers (a ChunkRegion), sized exactly to that chunk (n_grid = chunk_size,
# origin = chunk_pos * chunk_size) — chunk-aligned, not centered on anything. Activation is
# driven by real world state: EditDispatcher.voxels_placed (a player/spell/explosion placed
# MPM-eligible material) and chunk.active_simulation_mask() at boot + as chunks stream in
# (chunk_loaded) — never by player/camera position. A continuously flowing waterfall never
# goes idle, so it never sleeps, regardless of where the player is looking.
#
# KNOWN TRADEOFF, flagged not hidden: each active chunk simulates on its OWN grid, independent
# of its neighbors. A particle exactly on a chunk boundary doesn't hand off cleanly to the
# neighboring chunk's grid the way granular/fluid's shared sparse-buffer dispatch does — this
# is a real seam, not solved here. Fine for now (isolated ponds/piles); revisit if cross-chunk
# continuity (a wide lake spanning many chunks) turns out to matter in practice.
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

@export var n_grid: int = 32  # kept equal to ChunkManager.chunk_size — see class header
@export var sim_dt: float = 1.0e-4
@export var gravity: float = 9.8
# Per-CHUNK particle capacity. Smaller than the old single-window default (20000) since each
# active chunk now gets its own buffers and there can be several simultaneously — see
# max_active_chunks below for the real resource cap.
@export var max_particles: int = 4000
# Hard cap on simultaneously active chunks (each with its own full grid+particle buffer set).
# A real resource budget, matching the project's "active tier" philosophy (brief §19.1) —
# activation requests beyond this are dropped (logged once, not spammed), not queued.
@export var max_active_chunks: int = 8
const BOUND := 3
const NMAT := 8  # must match clear_grid.comp/p2g.comp's NMAT — the render mass-breakdown slot count
const DIRTY_BUF_BYTES := 64  # matches GranularSimulation's own constant/convention
const IDLE_THRESHOLD := 4    # ticks with no readout change before a chunk sleeps — matches
							  # GranularSimulation's own threshold (granular_simulation.gd)
# Target magnitude for a busy grid node's fixed-point accumulator (fp_scale is derived from
# this, not hardcoded — §36.3 found a fixed magic constant silently rounds low-density
# materials to zero at one dx and silently overflows int32 at another).
const NODE_BUDGET := 1.0e7

# --- Per-chunk simulation state ---
class ChunkRegion:
	var n_particles: int = 0
	var groups_particles: int = 0
	var idle_ticks: int = 0

	var params_buf: RID
	var pos: RID; var vel: RID; var C: RID; var J: RID
	var mat: RID; var mass: RID; var alive: RID; var F: RID
	var g_mom: RID; var g_mass: RID; var g_vel: RID; var g_mass_mat: RID
	var voxel_params_buf: RID; var voxel_buf_a: RID; var chunk_table: RID; var dirty_chunks_buf: RID

	var set_clear: RID; var set_p2g: RID; var set_grid_op: RID; var set_g2p: RID; var set_readout: RID

var _windows: Dictionary = {}  # Vector3i chunk_pos -> ChunkRegion
var _warned_budget := false

# Shared across all windows (same n_grid for every chunk).
var dx: float
var inv_dx: float
var p_vol: float
var fp_scale: float
var _groups_grid: int = 0

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
# connects to the signals that drive chunk activation. Call once. Allocates NO per-chunk
# buffers here — those come from _activate_chunk() as chunks actually need them.
func setup() -> bool:
	dx = 1.0
	inv_dx = 1.0
	p_vol = pow(dx * 0.5, 3.0)
	fp_scale = NODE_BUDGET / p_vol
	_groups_grid = int(ceil(float(n_grid * n_grid * n_grid) / 64.0))

	_clear   = _compute.compile_shader(_load_shader("res://shaders/mpm/clear_grid.comp"))
	_p2g     = _compute.compile_shader(_load_shader("res://shaders/mpm/p2g.comp"))
	_grid_op = _compute.compile_shader(_load_shader("res://shaders/mpm/grid_op.comp"))
	_g2p     = _compute.compile_shader(_load_shader("res://shaders/mpm/g2p.comp"))
	_readout = _compute.compile_shader(_load_shader("res://shaders/mpm/readout.comp"))
	if _clear.is_empty() or _p2g.is_empty() or _grid_op.is_empty() or _g2p.is_empty() or _readout.is_empty():
		push_error("MPMSimulation: one or more shaders failed to compile")
		return false

	var orch: WorldOrchestrator = get_node("/root/world/WorldOrchestrator")
	var cm: ChunkManager = get_node("/root/world/ChunkManager")
	if not orch.is_grid_valid():
		push_error("MPMSimulation.setup(): WorldOrchestrator grid not valid yet")
		return false
	_orchestrator = orch
	_chunk_manager = cm

	if not orch.voxels_placed.is_connected(_on_voxels_placed):
		orch.voxels_placed.connect(_on_voxels_placed)
	if not orch.chunk_loaded.is_connected(_on_chunk_loaded):
		orch.chunk_loaded.connect(_on_chunk_loaded)

	is_setup = true
	scan_loaded_chunks_for_existing_material()
	return true


func is_bound() -> bool:
	return is_instance_valid(_orchestrator)


# Player-independent discovery: checks EVERY currently loaded chunk's cheap, already-maintained
# active_simulation_mask() (Chunk.rebuild_counts_and_mask(), zero per-voxel scanning) for
# GRANULAR/LIQUID material and activates a window for any that qualify, up to budget. Called
# once from setup() and again from _on_chunk_loaded() as new chunks stream in — so a persistent
# lake somewhere the player has never looked still gets discovered and simulated, matching
# "not player-dependent."
func scan_loaded_chunks_for_existing_material() -> void:
	const CANDIDATE_MASK := 1 | 2  # SIM_BITS[GRANULAR] | SIM_BITS[LIQUID]
	for cp in _chunk_manager.chunks.keys():
		if _windows.has(cp):
			continue
		var chunk: Chunk = _chunk_manager.chunks[cp]
		if chunk.active_simulation_mask() & CANDIDATE_MASK != 0:
			_activate_chunk(cp)


func _on_chunk_loaded(chunk_pos: Vector3i) -> void:
	if not is_setup or _windows.has(chunk_pos):
		return
	var chunk: Chunk = _chunk_manager.chunks.get(chunk_pos)
	if not chunk:
		return
	const CANDIDATE_MASK := 1 | 2
	if chunk.active_simulation_mask() & CANDIDATE_MASK != 0:
		_activate_chunk(chunk_pos)


# Allocates a full grid+particle buffer set for one chunk and binds its STATIC no-slip read to
# the real voxel world at origin = chunk_pos * n_grid. Immediately scans that chunk for
# existing MPM-eligible material (promote_existing_voxels) so a chunk discovered via a single
# placed grain also picks up any pre-existing water/sand already there, not just the one voxel
# that triggered activation. Returns the new ChunkRegion, or null if the budget is exhausted.
func _activate_chunk(chunk_pos: Vector3i) -> ChunkRegion:
	if _windows.size() >= max_active_chunks:
		if not _warned_budget:
			push_warning("MPMSimulation: max_active_chunks=%d exhausted, dropping activation at %s" % [max_active_chunks, chunk_pos])
			_warned_budget = true
		return null

	var w := ChunkRegion.new()
	var gnodes := n_grid * n_grid * n_grid
	w.g_mom      = _new_zero_buf(gnodes * 3 * 4)
	w.g_mass     = _new_zero_buf(gnodes * 4)
	w.g_vel      = _new_zero_buf(gnodes * 3 * 4)
	w.g_mass_mat = _new_zero_buf(gnodes * NMAT * 4)
	w.params_buf = _new_zero_buf(48)
	w.dirty_chunks_buf = _new_zero_buf(DIRTY_BUF_BYTES)

	var origin := chunk_pos * n_grid
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

	# Empty particle buffers (max_particles capacity) — seed_particles()/add_particle() below fill them.
	var cap := max_particles
	w.pos   = _new_zero_buf(cap * 3 * 4)
	w.vel   = _new_zero_buf(cap * 3 * 4)
	w.C     = _new_zero_buf(cap * 9 * 4)
	w.J     = _new_zero_buf(cap * 4)
	w.mat   = _new_zero_buf(cap * 4)
	w.mass  = _new_zero_buf(cap * 4)
	w.alive = _new_zero_buf(cap * 4)
	w.F     = _new_zero_buf(cap * 9 * 4)

	_upload_window_params(w)
	_build_window_sets(w)
	_windows[chunk_pos] = w

	promote_existing_voxels(chunk_pos, w)
	return w


func _deactivate_chunk(chunk_pos: Vector3i) -> void:
	var w: ChunkRegion = _windows.get(chunk_pos)
	if not w:
		return
	for rid in [w.params_buf, w.pos, w.vel, w.C, w.J, w.mat, w.mass, w.alive, w.F,
			w.g_mom, w.g_mass, w.g_vel, w.g_mass_mat,
			w.voxel_params_buf, w.dirty_chunks_buf]:
		if rid.is_valid():
			_compute.free_rid(rid)
	_windows.erase(chunk_pos)


# ONE-SHOT: seeds particles from whatever FLUID/GRANULAR-modeled voxels currently exist in
# this chunk, so a newly-activated window doesn't just represent the one voxel that triggered
# it — a whole pre-existing pond in that chunk gets picked up too.
func promote_existing_voxels(chunk_pos: Vector3i, w: ChunkRegion) -> int:
	var origin := chunk_pos * n_grid
	_orchestrator.sync_chunk_to_cpu(chunk_pos)  # block_data is stale-by-design (CLAUDE.md)

	var found_pos: Array[Vector3i] = []
	var found_mat: Array[int] = []
	for ly in range(n_grid):
		for lz in range(n_grid):
			for lx in range(n_grid):
				var wp := origin + Vector3i(lx, ly, lz)
				var mat := _orchestrator.query_voxel_material(wp)
				if mat == 0:
					continue
				var def := _material_db.get_def(mat)
				if def and (def.mpm_model == MaterialDef.MPMModel.FLUID
						or def.mpm_model == MaterialDef.MPMModel.GRANULAR):
					found_pos.append(wp)
					found_mat.append(mat)

	var n := found_pos.size()
	if n == 0:
		return 0
	seed_particles(w, n, func(i: int) -> Dictionary:
		var wp := found_pos[i]
		var mat_id := found_mat[i]
		var def := _material_db.get_def(mat_id)
		var local := Vector3(wp - origin) + Vector3(0.5, 0.5, 0.5)  # voxel center
		return {"pos": local, "mat": mat_id, "mass": def.mpm_density * p_vol}
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


# seed_fn(i) -> Dictionary { pos: Vector3 (LOCAL to the window's chunk origin, voxel units),
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
	_upload_window_params(w)


# Continuous promotion: appends ONE particle into a window's unused tail capacity — a
# buffer_update() write at index n_particles plus incrementing it, NOT a reseed. No pipeline/
# uniform-set rebuild needed. Returns false if this window's max_particles is exhausted.
func add_particle(w: ChunkRegion, local_pos: Vector3, mat_id: int, mass: float) -> bool:
	if w.n_particles >= max_particles:
		push_warning("MPMSimulation.add_particle(): window max_particles=%d exhausted" % max_particles)
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
	_upload_window_params(w)
	return true


# EditDispatcher/WorldOrchestrator.voxels_placed handler — activates a window for the placed
# voxel's chunk if one doesn't already exist (budget permitting), then promotes the voxel into
# a particle. Chunk-keyed, not window/player-position-keyed — placing sand/water anywhere in
# any loaded chunk works identically, near the player or not.
func _on_voxels_placed(positions: Array[Vector3i], material_ids: Array[int]) -> void:
	if not is_setup:
		return
	for idx in positions.size():
		var wp: Vector3i = positions[idx]
		var mat_id: int = material_ids[idx]
		var def := _material_db.get_def(mat_id)
		if not def or (def.mpm_model != MaterialDef.MPMModel.FLUID and def.mpm_model != MaterialDef.MPMModel.GRANULAR):
			continue

		var chunk_pos := Vector3i(floori(float(wp.x) / n_grid), floori(float(wp.y) / n_grid), floori(float(wp.z) / n_grid))
		var w: ChunkRegion = _windows.get(chunk_pos)
		if not w:
			w = _activate_chunk(chunk_pos)
			if not w:
				continue  # budget exhausted, already warned
		w.idle_ticks = 0
		var origin := chunk_pos * n_grid
		var local_pos := Vector3(wp - origin) + Vector3(0.5, 0.5, 0.5)
		add_particle(w, local_pos, mat_id, def.mpm_density * p_vol)


func _upload_window_params(w: ChunkRegion) -> void:
	var b := PackedByteArray(); b.resize(48)
	b.encode_s32(0, n_grid)
	b.encode_s32(4, w.n_particles)
	b.encode_s32(8, BOUND)
	b.encode_s32(12, 0)
	b.encode_float(16, sim_dt)
	b.encode_float(20, dx)
	b.encode_float(24, inv_dx)
	b.encode_float(28, gravity)
	b.encode_float(32, p_vol)
	b.encode_float(36, fp_scale)
	b.encode_float(40, 0.0)
	b.encode_float(44, 0.0)
	_compute.buffer_update(w.params_buf, 0, b)


func _u(binding: int, rid: RID) -> RDUniform:
	var un := RDUniform.new()
	un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	un.binding = binding
	un.add_id(rid)
	return un


func _build_window_sets(w: ChunkRegion) -> void:
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
		_u(15, w.voxel_buf_a), _u(16, w.chunk_table)
	], _grid_op.shader)
	w.set_g2p = _compute.create_uniform_set([
		_u(0, w.params_buf), _u(1, w.pos), _u(2, w.vel), _u(3, w.C), _u(4, w.J), _u(5, w.mat),
		_u(7, w.alive), _u(10, w.g_vel), _u(11, w.F), _u(13, _material_db.mpm_buffer_rid)
	], _g2p.shader)
	w.set_readout = _compute.create_uniform_set([
		_u(0, w.params_buf), _u(9, w.g_mass), _u(12, w.g_mass_mat),
		_u(13, _material_db.mpm_buffer_rid), _u(14, w.voxel_params_buf),
		_u(15, w.voxel_buf_a), _u(16, w.chunk_table), _u(17, _orchestrator.get_changelog_buf()),
		_u(18, w.dirty_chunks_buf)
	], _readout.shader)


# Below this max speed (voxel-units/simulated-second), a particle counts as "at rest" for
# idle-detection purposes. Found empirically (2026-08-12): readout()'s voxel-level changed/
# unchanged signal was tried first and is TOO COARSE — a particle early in a gravity fall
# (measured max_speed ~0.12 after only ~0.012 simulated seconds, v=g*t) can occupy the same
# grid cell for several consecutive ticks without crossing a voxel boundary, producing
# identical readout output while genuinely still accelerating — which hit IDLE_THRESHOLD and
# deactivated chunks that were still actively falling, exactly the "waterfall stops" failure
# this whole redesign exists to prevent, just one layer deeper than player-position coupling.
# Velocity is the actual physical quantity that means "at rest," not a proxy for it.
const REST_SPEED := 0.02

# Public entry point — call ONCE per real simulation tick (SimulationRuntime._tick()). Runs
# every active chunk's physics substeps + readout, then updates idle counters and drops any
# chunk that's been idle IDLE_THRESHOLD ticks in a row (matches GranularSimulation's own
# threshold) — a chunk with continuous activity (a waterfall) never idles out, independent of
# player position.
func tick_all(substeps_per_tick: int) -> void:
	if not is_bound():
		return
	var to_drop: Array[Vector3i] = []
	for chunk_pos in _windows.keys():
		var w: ChunkRegion = _windows[chunk_pos]
		if w.n_particles == 0:
			continue
		_run_window_substeps(w, substeps_per_tick)
		_readout_window(w)  # still writes voxel state every tick; return value no longer
		# drives idle-detection (see REST_SPEED above) — kept for the STATIC-guard write path.

		var vel := read_velocities(w)
		var max_speed := 0.0
		for i in w.n_particles:
			var spd := Vector3(vel[i*3], vel[i*3+1], vel[i*3+2]).length()
			max_speed = max(max_speed, spd)

		if max_speed < REST_SPEED:
			w.idle_ticks += 1
			if w.idle_ticks >= IDLE_THRESHOLD:
				to_drop.append(chunk_pos)
		else:
			w.idle_ticks = 0
	for chunk_pos in to_drop:
		_deactivate_chunk(chunk_pos)


func _run_window_substeps(w: ChunkRegion, substeps: int) -> void:
	for i in substeps:
		_compute.run_with_set(_clear,   w.set_clear,   _groups_grid)
		_compute.run_with_set(_p2g,     w.set_p2g,     w.groups_particles)
		_compute.run_with_set(_grid_op, w.set_grid_op, _groups_grid)
		_compute.run_with_set(_g2p,     w.set_g2p,     w.groups_particles)
	_compute.submit()
	_compute.sync()


# Writes the window's current grid mass/material state into the REAL voxel world (shared
# changelog + WorldOrchestrator.apply_changelog() — same safe path GranularSimulation/
# FluidSimulation use). Does NOT emit WorldOrchestrator.chunks_committed — CA
# (GranularSimulation/FluidSimulation) is currently disabled (enabled=false in world.tscn), so
# the original conflict concern (CA picking up voxels MPM just wrote) doesn't apply right now;
# left unwired rather than wired-but-untested, since re-enabling CA alongside MPM on the same
# materials is still an open question this migration hasn't answered.
func _readout_window(w: ChunkRegion) -> Array[Vector3i]:
	var zero_dirty := PackedByteArray(); zero_dirty.resize(DIRTY_BUF_BYTES); zero_dirty.fill(0)
	_compute.buffer_update(w.dirty_chunks_buf, 0, zero_dirty)
	_orchestrator.reset_changelog()

	_compute.run_with_set(_readout, w.set_readout, _groups_grid)
	_compute.submit()
	_compute.sync()

	_orchestrator.apply_changelog()

	var bitmask: PackedByteArray = _compute.download_range(w.dirty_chunks_buf, 0, DIRTY_BUF_BYTES)
	return _decode_dirty_bitmask(bitmask)


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


# --- Diagnostics / test accessors — operate on a specific chunk's window ---

func get_chunk_region(chunk_pos: Vector3i) -> ChunkRegion:
	return _windows.get(chunk_pos)


func active_chunk_positions() -> Array:
	return _windows.keys()


func read_positions(w: ChunkRegion) -> PackedFloat32Array:
	return _compute.download(w.pos).to_float32_array()


func read_velocities(w: ChunkRegion) -> PackedFloat32Array:
	return _compute.download(w.vel).to_float32_array()


func read_materials(w: ChunkRegion) -> PackedInt32Array:
	return _compute.download(w.mat).to_int32_array()
