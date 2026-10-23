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
# Hard ceiling on region grid nodes. 1M nodes ≈ 60 MB of grid buffers (60 B/node) and ≈ 15.6k
# workgroups per pass. Exceeded → the chunk that would cross it is refused, not allocated.
const MAX_REGION_NODES := 1048576
const BOUND := 3
const NMAT := 8  # must match clear_grid.comp/p2g.comp's NMAT — the render mass-breakdown slot count
const DIRTY_BUF_BYTES := 64  # matches GranularSimulation's own constant/convention
const IDLE_THRESHOLD := 4    # ticks with no motion before the whole region sleeps — matches
							 # GranularSimulation's own threshold (granular_simulation.gd)
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

	var set_clear: RID; var set_p2g: RID; var set_grid_op: RID; var set_g2p: RID; var set_readout: RID

# The one live region, or null when nothing is being simulated.
var _region: ChunkRegion = null
# Chunks that currently WANT to be simulated. Region geometry is always derived from this set,
# never edited directly — _ensure_region() is the only thing that builds grids.
var _active_chunks: Dictionary = {}  # Vector3i -> true
var _warned_budget := false
var _warned_region_cap := false
var _warned_capacity := false
# Ticks to wait before the region may grow downward again — see _grow_region_down_if_needed().
var _grow_cooldown := 0
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
	dx = 1.0
	inv_dx = 1.0
	p_vol = pow(dx * 0.5, 3.0)
	fp_scale = NODE_BUDGET / p_vol

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
	# R-key regenerate (player_controller) calls ChunkManager.regenerate_chunks(), which emits
	# slab_invalidated -> WorldOrchestrator._init_grid(). That FREES AND RECREATES
	# _voxel_buf_a, _chunk_table and _changelog_buf with new RIDs, and a live region binds all
	# three directly (w.voxel_buf_a / w.chunk_table and the readout uniform set). Without this
	# the region keeps pointing at freed handles and readout writes stale particles into a dead
	# buffer. Tear the region down whenever the grid is rebuilt.
	if not cm.slab_invalidated.is_connected(Callable(self, "reset")):
		cm.slab_invalidated.connect(Callable(self, "reset"))

	is_setup = true
	scan_loaded_chunks_for_existing_material()
	return true


func is_bound() -> bool:
	return is_instance_valid(_orchestrator)


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
	_grow_cooldown = 0
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
func _try_add_chunk(cp: Vector3i) -> bool:
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
	_ensure_region()
	return true


# (Re)builds the region if its geometry no longer matches the active set. This is the ONLY
# place grid buffers are created. Rebuilding re-seeds particles from the voxel world (which is
# the authoritative, readout-written state), so in-flight velocities are lost across a rebuild
# — acceptable because rebuilds only happen when the region's chunk set changes, which is a
# rare, structural event, not a per-tick one.
func _ensure_region() -> bool:
	if _active_chunks.is_empty():
		_destroy_region()
		return false

	var bmin := Vector3i(2147483647, 2147483647, 2147483647)
	var bmax := Vector3i(-2147483648, -2147483648, -2147483648)
	var chunks: Array[Vector3i] = []
	for key in _active_chunks:
		var c: Vector3i = key
		chunks.append(c)
		bmin = Vector3i(mini(bmin.x, c.x), mini(bmin.y, c.y), mini(bmin.z, c.z))
		bmax = Vector3i(maxi(bmax.x, c.x), maxi(bmax.y, c.y), maxi(bmax.z, c.z))

	var dims := (bmax - bmin + Vector3i.ONE) * n_grid
	var origin := bmin * n_grid
	var sig := _active_signature()

	if _region != null and _region.dims == dims and _region.origin == origin:
		if _region.signature == sig:
			return false
		# Same bbox, different membership — keep the grid, re-seed particles from the voxel
		# world so the new chunk's material is picked up.
		_region.chunks = chunks
		_region.signature = sig
		promote_existing_voxels(_region, chunks)
		return false

	_destroy_region()
	_region = _build_region(origin, dims, chunks)
	_region.signature = sig
	return true


# Order-independent fingerprint of the active chunk set (positional hash combine).
func _active_signature() -> int:
	var h := 0
	for key in _active_chunks:
		var c: Vector3i = key
		h = (h * 73856093) ^ (int(c.x) * 19349663) ^ (int(c.y) * 83492791) ^ (int(c.z) * 2654435761)
	return h


func _build_region(origin: Vector3i, dims: Vector3i, chunks: Array[Vector3i]) -> ChunkRegion:
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
	promote_existing_voxels(w, chunks)
	return w


func _destroy_region() -> void:
	var w := _region
	_region = null
	if not w:
		return
	for rid in [w.params_buf, w.pos, w.vel, w.C, w.J, w.mat, w.mass, w.alive, w.F,
		w.g_mom, w.g_mass, w.g_vel, w.g_mass_mat,
		w.voxel_params_buf, w.dirty_chunks_buf]:
		if rid.is_valid():
			_compute.free_rid(rid)


# Seeds particles from whatever FLUID/GRANULAR-modeled voxels currently exist in the given
# chunks, so a newly-activated region isn't just the one voxel that triggered it — a whole
# pre-existing pond is picked up too, including ponds in the NEIGHBOURING chunks that share
# this grid (which is the point: they're one body of water now).
func promote_existing_voxels(w: ChunkRegion, chunks: Array[Vector3i]) -> int:
	var found_pos: Array[Vector3i] = []
	var found_mat: Array[int] = []
	for cp in chunks:
		var origin := cp * n_grid
		_orchestrator.sync_chunk_to_cpu(cp)  # block_data is stale-by-design (CLAUDE.md)
		for ly in range(n_grid):
			for lz in range(n_grid):
				for lx in range(n_grid):
					var wp := origin + Vector3i(lx, ly, lz)
					var mat := _orchestrator.query_voxel_material(wp)
					if _is_mpm_material(mat):
						found_pos.append(wp)
						found_mat.append(mat)

	var n := found_pos.size()
	if n == 0:
		return 0
	seed_particles(w, n, func(i: int) -> Dictionary:
		var wp := found_pos[i]
		var mat_id := found_mat[i]
		var def := _material_db.get_def(mat_id)
		var local := Vector3(wp - w.origin) + Vector3(0.5, 0.5, 0.5)  # voxel center, region-local
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
		_ensure_region()  # exactly ONE rebuild + reseed for the whole edit

	# PASS 2 — append only voxels in chunks that were ALREADY active. A voxel in a chunk that
	# just activated is already covered by promote_existing_voxels() (edit_apply.comp writes
	# _voxel_buf_a before this signal fires), so appending it again would duplicate it.
	if _region == null:
		return
	var add_pos: Array[Vector3] = []
	var add_mat: Array[int] = []
	var add_mass: Array[float] = []
	for idx in positions.size():
		var mat_id: int = material_ids[idx]
		if not _is_mpm_material(mat_id):
			continue
		var wp: Vector3i = positions[idx]
		var cp := _chunk_of(wp)
		if new_chunks.has(cp) or not _region.chunks.has(cp):
			continue
		var def := _material_db.get_def(mat_id)
		add_pos.append(_region_local(wp))
		add_mat.append(mat_id)
		add_mass.append(def.mpm_density * p_vol)
	_append_particles(_region, add_pos, add_mat, add_mass)


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
	b.encode_s32(40, BOUND)
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
# this whole design exists to prevent, just one layer deeper than player-position coupling.
# Velocity is the actual physical quantity that means "at rest," not a proxy for it.
const REST_SPEED := 0.02

# Public entry point — call ONCE per real simulation tick (SimulationRuntime._tick()). Runs the
# region's physics substeps + readout, then sleeps the whole region once it has been motionless
# for IDLE_THRESHOLD ticks. Idle is a REGION property now, not a per-chunk one: chunks share a
# grid, so one chunk going quiet while its neighbour still flows is not a reason to drop
# anything — that would tear the shared grid apart mid-simulation.
func tick_all(substeps_per_tick: int) -> void:
	if not is_bound():
		return
	var w := _region
	if w == null or w.n_particles == 0:
		return

	# Derive the substep dt from the single pacing knob (see sim_seconds_per_tick).
	var want_dt := sim_seconds_per_tick / maxf(float(substeps_per_tick), 1.0)
	if not is_equal_approx(want_dt, _applied_sim_dt):
		_applied_sim_dt = want_dt
		sim_dt = want_dt
		_upload_region_params(w)

	_run_region_substeps(w, substeps_per_tick)
	var changed := _readout_region(w)

	# FOLLOW FALLING MATERIAL DOWNWARD — added after a real run showed a region at chunk y 0..1
	# with NO chunk below y=0 active. Chunks only ever activated from edits, chunk loads and the
	# boot scan, so nothing grew the region to follow a falling stream: water fell to the bottom
	# of the grid, g2p clamped it, and it piled up against the region's floor — the vertical
	# version of "climbs up that wall".
	if _grow_region_down_if_needed(w, changed):
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
	if w.idle_ticks < IDLE_THRESHOLD:
		return

	var max_speed := 0.0
	var vel := read_velocities(w)
	for i in w.n_particles:
		var spd := Vector3(vel[i*3], vel[i*3+1], vel[i*3+2]).length()
		max_speed = max(max_speed, spd)

	if max_speed < REST_SPEED:
		# Everything is at rest and already written back to voxels. Sleep the region and clear
		# the intent set; a later edit or a streamed chunk re-activates from voxels.
		_active_chunks.clear()
		_destroy_region()
	else:
		w.idle_ticks = 0


# If the region's BOTTOM-MOST active chunk produced readout changes this tick, material is
# active at the floor of the simulated domain — add the chunk below so it has somewhere to go.
# Returns true if the region was rebuilt (caller must not touch its ChunkRegion afterwards).
#
# Self-limiting by construction: after the rebuild the origin has dropped a chunk, so the
# material that triggered this is no longer in the bottom chunk. Resting material produces NO
# readout change (readout early-outs when new_voxel == cur_voxel), so a settled pool cannot
# ratchet the region downward one chunk per tick.
#
# KNOWN LIMITATION, and it is a visible one: _ensure_region() rebuilds by re-seeding particles
# from voxels, so a falling stream LOSES ITS VELOCITY each time it crosses a chunk boundary
# downward. Fixing that means preserving particle buffers across a resize (allocate new, copy
# the old contents with a position offset, promote only the newly added chunks) instead of
# re-seeding. Tracked in docs/MIGRATION_MPM.md §7 as the next S3 item.
func _grow_region_down_if_needed(w: ChunkRegion, changed: Array[Vector3i]) -> bool:
	# RATE LIMIT. "Bottom chunk changed this tick" is a proxy for "material is active at the
	# floor", but a settling pool jitters its fill value, which readout reports as a change every
	# tick — without this it would ratchet one chunk downward per tick until the budget died.
	if _grow_cooldown > 0:
		_grow_cooldown -= 1
		return false

	if changed.is_empty() or w.chunks.is_empty():
		return false

	var bottom := 2147483647
	for cp in w.chunks:
		bottom = mini(bottom, cp.y)

	var at_floor := false
	for cp in changed:
		if cp.y == bottom:
			at_floor = true
			break
	if not at_floor:
		return false

	# One chunk directly below any bottom chunk — it does not need to span the whole footprint,
	# it only needs to give the falling material somewhere to exist.
	var below := Vector3i(0, 0, 0)
	var found := false
	for cp in w.chunks:
		if cp.y == bottom:
			below = Vector3i(cp.x, bottom - 1, cp.z)
			found = true
			break
	if not found or _active_chunks.has(below):
		return false
	if not _chunk_manager.chunks.has(below):
		return false  # not streamed in — a genuine edge of the world
	_grow_cooldown = 15
	return _try_add_chunk(below)


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
	_orchestrator.reset_changelog()

	_compute.run_with_set(_readout, w.set_readout, w.groups_grid)
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


# --- Diagnostics / test accessors ---

# The region now spans multiple chunks, so this returns the region if `chunk_pos` is covered by
# it, else null. Kept under the old name so existing callers keep working; n_particles on the
# returned object is REGION-wide, not that chunk's share.
func get_chunk_region(chunk_pos: Vector3i) -> ChunkRegion:
	if _region == null:
		return null
	return _region if _region.chunks.has(chunk_pos) else null


func active_chunk_positions() -> Array:
	return _active_chunks.keys()


func region_grid_info() -> Dictionary:
	if _region == null:
		return {}
	return {
		"origin": _region.origin,
		"dims": _region.dims,
		"nodes": _region.dims.x * _region.dims.y * _region.dims.z,
		"chunks": _region.chunks.size(),
		"particles": _region.n_particles,
		# Exposed because "is it going to sleep?" is the question this dump exists to answer:
		# the region sleeps after IDLE_THRESHOLD consecutive ticks with an empty readout
		# dirty-set AND a max particle speed under REST_SPEED.
		"idle_ticks": _region.idle_ticks,
		"capacity": max_particles,
	}


func read_positions(w: ChunkRegion) -> PackedFloat32Array:
	return _compute.download(w.pos).to_float32_array()


func read_velocities(w: ChunkRegion) -> PackedFloat32Array:
	return _compute.download(w.vel).to_float32_array()


func read_materials(w: ChunkRegion) -> PackedInt32Array:
	return _compute.download(w.mat).to_int32_array()
