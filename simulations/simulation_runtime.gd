class_name SimulationRuntime extends Node

# Fired once per real simulation tick, whether from run()'s accumulator (unpaused) or
# step_once() (paused, manual single-step) — both funnel through _tick(). Added 2026-07-16
# so world.gd's auto-F6 toggle can fire the dump after every tick without world.gd needing
# its own tick-counter bookkeeping.
signal tick_advanced

const SIM_HZ := 60.0       # simulation ticks per second
const MAX_CATCHUP := 1     # max ticks per render frame to prevent spiral-of-death

var _chunk_manager: ChunkManager
var _material_db: MaterialDB
var _material_simulation: PackedInt32Array   # material_id → SimulationClass
var _simulations: Dictionary                 # SIM_BITS[class] → VoxelSimulation
var _active: Array[VoxelSimulation]          # simulations currently with work
var _accumulator: float = 0.0
var paused: bool = false

# --- Solver selection (2026-08-13) — EXACTLY ONE solver owns the voxel buffer per run ---
#
# CA and MPM both write `_voxel_buf_a`, through different owners: CA via per-tick changelogs
# -> patch_apply.comp, MPM via readout.comp. Before this gate all three writers ran every tick
# simultaneously, which is the maximal-seam configuration and the documented source of most
# multi-session bugs in docs/BUG_LOG.md (brief §37: "one solver removes the cross-solver seam
# ... rather than managing it"). The gate makes "one solver at a time" structural instead of a
# convention nobody enforces.
#
# CA  = GranularSimulation + FluidSimulation (the working baseline, and the reference oracle
#       MPM water is validated against while it is still at pilot scale).
# MPM = MPMSimulation only; the CA sims are never woken, so they can't compete for the buffer.
#
# Default is CA deliberately: MPM is currently pilot-scale (4000 particles/chunk, 8 active
# chunks) and would visibly degrade the game if made the only solver today. Flipping the
# cutover is a one-line change — set this export to MPM in world.tscn, or change the default
# below — and should happen only once MPM reaches parity (see docs/MIGRATION_MPM.md).
enum SolverMode { CA = 0, MPM = 1 }
@export_enum("CA", "MPM") var solver_mode: int = SolverMode.CA

# MLS-MPM migration (2026-08-12, brief §37) — NOT a VoxelSimulation (see mpm_simulation.gd's
# header for why), so it doesn't go through register_simulation()/_active; ticked directly
# here instead. mpm_simulation is a CHILD of this node, so unlike _material_db above (a later
# SIBLING of this node's own ancestor WorldOrchestrator, not ready yet at this point) it's
# already ready by the time this runs — Godot readies children before their own parent.
var mpm_simulation: MPMSimulation
# Substeps of MPM physics per real 60Hz tick. 24 mirrors the retired evaluation's own tuned
# default (brief §32.3) — a starting point, not a value re-derived for this game's specific
# performance profile; tune directly if pacing looks wrong in play.
@export var mpm_substeps_per_tick: int = 24


func _ready() -> void:
	_chunk_manager = get_node("/root/world/ChunkManager")
	_material_db   = get_node("/root/world/MaterialDB")
	mpm_simulation = get_node_or_null("MPMSimulation")
	var orch := get_node("/root/world/WorldOrchestrator")
	orch.chunks_committed.connect(_on_chunks_committed)
	orch.chunk_loaded.connect(func(cp: Vector3i): _on_chunks_committed([cp]))
	_resolve_solver_mode()
	call_deferred("_build_material_cache")


# Command-line override, applied once at boot so benches/tests can force a solver without
# editing the scene: `-- mpm` or `-- ca`. The MPM integration test implies MPM.
# Resolved here (not in world.gd) so this node stays the single source of truth for the mode,
# and so the print below always reflects the mode actually in effect.
func _resolve_solver_mode() -> void:
	var args := OS.get_cmdline_user_args()
	if args.has("mpm") or args.has("mpm_multi_chunk_test"):
		solver_mode = SolverMode.MPM
	elif args.has("ca"):
		solver_mode = SolverMode.CA
	var mode_name := "MPM" if is_mpm_mode() else "CA"
	print("[SimRuntime] solver mode = %s (one solver owns the voxel buffer this run)" % mode_name)


func is_mpm_mode() -> bool:
	return solver_mode == SolverMode.MPM


func _build_material_cache() -> void:
	_material_simulation.resize(256)
	_material_simulation.fill(MaterialDef.SimulationClass.NONE)
	for def in _material_db.materials:
		_material_simulation[def.id] = def.simulation


func register_simulation(sim: VoxelSimulation) -> void:
	var sc := sim.get_simulation_class()
	if sc != MaterialDef.SimulationClass.NONE:
		_simulations[MaterialDef.SIM_BITS[sc]] = sim


func _on_chunks_committed(chunk_positions: Array[Vector3i]) -> void:
	# MPM owns the buffer this run — never wake the CA sims, so they cannot accumulate
	# candidates and cannot step. MPM does its own activation (setup()'s scan,
	# voxels_placed, chunk_loaded), so it needs nothing from this path.
	if is_mpm_mode():
		return

	# Per-simulation candidate sets (chunk positions, not voxel positions).
	# We check each committed chunk plus its vertical neighbours — grains fall
	# between chunks, so a grain in the chunk above may now be able to move down.
	var per_sim: Dictionary = {}
	for bit in _simulations:
		per_sim[bit] = {} as Dictionary  # set semantics: chunk_pos → true

	var neighbours := [
		Vector3i.ZERO,
		Vector3i( 0,  1,  0), Vector3i( 0, -1,  0),
		Vector3i( 1,  0,  0), Vector3i(-1,  0,  0),
		Vector3i( 0,  0,  1), Vector3i( 0,  0, -1),
	]
	for cp in chunk_positions:
		# Always add the committed chunk itself — mask may be stale if mark_simulation_active
		# hasn't fired yet, or if the chunk was just loaded. Shaders filter non-applicable voxels.
		if _chunk_manager.chunks.has(cp):
			for bit in per_sim:
				per_sim[bit][cp] = true
		# Neighbors still use the mask check — avoids adding every chunk near any edit.
		for offset in neighbours:
			if offset == Vector3i.ZERO:
				continue
			var neighbour: Vector3i = cp + offset
			var chunk: Chunk = _chunk_manager.chunks.get(neighbour)
			if not chunk:
				continue
			var mask := chunk.active_simulation_mask()
			if mask == 0:
				continue
			for bit in per_sim:
				if mask & bit:
					per_sim[bit][neighbour] = true

	for bit in _simulations:
		if per_sim[bit].is_empty():
			continue
		var sim: VoxelSimulation = _simulations[bit]
		var chunk_pos_list: Array[Vector3i] = []
		for cp in per_sim[bit]:
			chunk_pos_list.append(cp)
		var was_active := _active.has(sim)
		sim.add_candidates(chunk_pos_list)
		if not was_active:
			_active.append(sim)


func run(delta: float) -> void:
	if paused:
		return
	_accumulator += delta
	var tick_dt := 1.0 / SIM_HZ
	var ticks := 0
	while _accumulator >= tick_dt and ticks < MAX_CATCHUP:
		_accumulator -= tick_dt
		ticks += 1
		_tick()
		tick_advanced.emit()


# Debug single-step: advances exactly one simulation tick regardless of the accumulator,
# intended for use while paused (world.gd binds this to Right Arrow). Bypasses run()'s
# accumulator entirely so stepping doesn't get skipped or double-fired by leftover
# accumulated time from before the pause.
func step_once() -> void:
	_tick()
	tick_advanced.emit()


func _tick() -> void:
	# MPM path: MPM only. The CA sims below are deliberately not stepped — stepping both
	# would put two independent owners on the same voxels (see SolverMode's header comment).
	if is_mpm_mode():
		if mpm_simulation and mpm_simulation.is_bound():
			mpm_simulation.tick_all(mpm_substeps_per_tick)
		return

	var to_run := _active.duplicate()
	var still_active: Array[VoxelSimulation] = []
	for sim in to_run:
		sim.step()
		if sim.has_work():
			still_active.append(sim)
	for sim in _active:
		if not still_active.has(sim) and sim.has_work():
			still_active.append(sim)
	_active = still_active
