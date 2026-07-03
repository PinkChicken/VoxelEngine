class_name SimulationRuntime extends Node

const SIM_HZ := 60.0       # simulation ticks per second
const MAX_CATCHUP := 1     # max ticks per render frame to prevent spiral-of-death

var _chunk_manager: ChunkManager
var _material_db: MaterialDB
var _material_simulation: PackedInt32Array   # material_id → SimulationClass
var _simulations: Dictionary                 # SIM_BITS[class] → VoxelSimulation
var _active: Array[VoxelSimulation]          # simulations currently with work
var _accumulator: float = 0.0
var paused: bool = false


func _ready() -> void:
	_chunk_manager = get_node("/root/world/ChunkManager")
	_material_db   = get_node("/root/world/MaterialDB")
	var orch := get_node("/root/world/WorldOrchestrator")
	orch.chunks_committed.connect(_on_chunks_committed)
	orch.chunk_loaded.connect(func(cp: Vector3i): _on_chunks_committed([cp]))
	call_deferred("_build_material_cache")


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


# Debug single-step: advances exactly one simulation tick regardless of the accumulator,
# intended for use while paused (world.gd binds this to Right Arrow). Bypasses run()'s
# accumulator entirely so stepping doesn't get skipped or double-fired by leftover
# accumulated time from before the pause.
func step_once() -> void:
	_tick()


func _tick() -> void:
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
