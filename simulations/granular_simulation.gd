class_name GranularSimulation extends VoxelSimulation

# GPU dispatcher — step() compiles to one compute dispatch per active chunk.
# Active chunks are tracked at chunk granularity (not voxel): add_candidates()
# converts world positions to chunk positions. Chunks exit the active set when
# they produce no changelog entries (all granular voxels stable). They re-enter
# when chunks_committed fires for a neighbour and discovery calls add_candidates.

var _compute                          # ComputeHelper
var _pipeline: Dictionary = {}
var _zero_pipeline: Dictionary = {}   # zero_slots.comp — clears active chunk slots in voxels_b
var _global_params_buf: RID           # 32-byte global params, uploaded once per step
var _chunk_list_buf: RID              # N×3 ints: active chunk positions (granular pass)
var _chunk_list_capacity: int = 0
var _zero_chunk_list_buf: RID         # active + below-neighbors (zero pass)
var _zero_chunk_list_capacity: int = 0
var _dirty_chunks_buf: RID           # bitmask: 1 bit per voxel-buffer slot, written by shader atomicOr
var _orchestrator: WorldOrchestrator
var _chunk_manager: ChunkManager
var _active_chunks: Dictionary = {}  # Vector3i → int (consecutive idle frames, 0 = active this frame)
var _frame: int = 0

# 64 bytes covers 512 slots (512 / 32 * 4) — exactly the full fixed world (8³ chunks).
# If buffer_grow ever pushes capacity past 512, _dirty_chunks_buf must also grow.
const DIRTY_BUF_BYTES := 64


func _ready() -> void:
	_orchestrator  = get_node("/root/world/WorldOrchestrator")
	_chunk_manager = get_node("/root/world/ChunkManager")
	_compute = preload("res://Utils/compute_helper.gd").new()
	if _compute.has_device():
		_pipeline = _compute.compile_shader(_load_shader(), 4)
		if _pipeline.is_empty():
			push_error("GranularSimulation: shader compile failed")
		_zero_pipeline = _compute.compile_shader(_load_zero_shader(), 4)
		if _zero_pipeline.is_empty():
			push_error("GranularSimulation: zero_slots shader compile failed")
		var p := PackedByteArray(); p.resize(32); p.fill(0)
		_global_params_buf = _compute.buffer_upload(p)
		# Seed chunk lists; both grow as needed.
		_chunk_list_capacity = 64
		var cl := PackedByteArray(); cl.resize(_chunk_list_capacity * 12); cl.fill(0)
		_chunk_list_buf = _compute.buffer_upload(cl)
		_zero_chunk_list_capacity = 128  # active + below-neighbors
		var zl := PackedByteArray(); zl.resize(_zero_chunk_list_capacity * 12); zl.fill(0)
		_zero_chunk_list_buf = _compute.buffer_upload(zl)
		var db := PackedByteArray(); db.resize(DIRTY_BUF_BYTES); db.fill(0)
		_dirty_chunks_buf = _compute.buffer_upload(db)
	var runtime := get_parent()
	if runtime.has_method("register_simulation"):
		runtime.register_simulation(self)

	Performance.add_custom_monitor("Voxel/active_chunks", func(): return _active_chunks.size())


func _exit_tree() -> void:
	Performance.remove_custom_monitor("Voxel/active_chunks")
	if _global_params_buf.is_valid():
		_compute.free_rid(_global_params_buf)
	if _chunk_list_buf.is_valid():
		_compute.free_rid(_chunk_list_buf)
	if _zero_chunk_list_buf.is_valid():
		_compute.free_rid(_zero_chunk_list_buf)
	if _dirty_chunks_buf.is_valid():
		_compute.free_rid(_dirty_chunks_buf)


func get_simulation_class() -> int:
	return MaterialDef.SimulationClass.GRANULAR


func add_candidates(chunk_positions: Array[Vector3i]) -> void:
	for cp in chunk_positions:
		_active_chunks[cp] = 0  # reset idle counter whenever a chunk re-enters


func has_work() -> bool:
	return not _active_chunks.is_empty()


func get_active_chunk_positions() -> Array[Vector3i]:
	var out: Array[Vector3i] = []
	for cp in _active_chunks:
		out.append(cp)
	return out


func step() -> void:
	_frame += 1
	if _active_chunks.is_empty() or _pipeline.is_empty():
		return
	if not _orchestrator.is_grid_valid():
		return
	# Reset dirty bitmask and changelog before dispatch.
	var zero_dirty := PackedByteArray(); zero_dirty.resize(DIRTY_BUF_BYTES); zero_dirty.fill(0)
	_compute.buffer_update(_dirty_chunks_buf, 0, zero_dirty)
	_orchestrator.reset_changelog()

	var grid_min  := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()
	_upload_global_params(grid_min, grid_span, _frame)

	var active_list := _active_chunks.keys()
	var active_count := active_list.size()
	_upload_chunk_list(active_list)

	# Zero active chunks AND their below-neighbors so grains can fall cross-chunk
	# via CAS without hitting stale B data in non-active chunks.
	var zero_list := _build_zero_list(active_list)
	_upload_zero_chunk_list(zero_list)
	var zero_uniforms := _build_zero_uniforms()
	_compute.run(_zero_pipeline, zero_uniforms, zero_list.size() * 4, 4, 4)
	_compute.submit()
	_compute.sync()

	# Granular simulation — each grain atomically claims its destination via CAS.
	var uniforms := _build_uniforms()
	_compute.run(_pipeline, uniforms, active_count * 4, 4, 4)
	_compute.submit()
	_compute.sync()

	# Download dirty bitmask immediately post-sync — just a memcopy, no GPU stall.
	# The bitmask replaces _extract_changed_chunks (was O(131k) GDScript loop).
	var bitmask: PackedByteArray = _compute.download_range(_dirty_chunks_buf, 0, DIRTY_BUF_BYTES)

	# Patch _voxel_buf_a on the GPU from the changelog (no CPU extraction).
	_orchestrator.apply_changelog()

	# Decode bitmask → dirty chunk positions.
	var changed: Array[Vector3i] = _decode_dirty_bitmask(bitmask)
	if not changed.is_empty():
		# Stamp the simulation mask on every chunk sand moved into or out of.
		# apply_changelog only patches voxels_a (GPU) and never updates block_data or
		# _simulation_mask, so destination chunks would silently have mask=0 and be
		# skipped by the neighbor discovery path on future commits.
		for cp in changed:
			var chunk: Chunk = _chunk_manager.chunks.get(cp)
			if chunk:
				chunk.mark_simulation_active(MaterialDef.SimulationClass.GRANULAR)
		_orchestrator.chunks_committed.emit(changed)

	# Stability: a chunk must appear in zero dirty bitmask frames for FOUR consecutive
	# dispatch frames before it can sleep. Four frames covers both checkerboard parities
	# twice, giving boundary grains two full opportunities to find a free path.
	var changed_set: Dictionary = {}
	for cp in changed:
		changed_set[cp] = true
	var still_active: Dictionary = {}
	for cp in _active_chunks:
		if changed_set.has(cp):
			still_active[cp] = 0           # activity this frame: reset idle count
		else:
			var idle: int = _active_chunks[cp] + 1
			if idle < 4:
				still_active[cp] = idle    # keep alive until both parities checked twice
			# idle >= 4: confirmed stable — drop the chunk
	# Directly add every dirty chunk regardless of its CPU simulation mask.
	# apply_changelog() only patches the GPU buffer — it does not update block_data or
	# _simulation_mask. Chunks that received sand via the changelog would otherwise
	# never be discovered by SimulationRuntime (mask check returns 0). The dirty
	# bitmask IS the ground truth: any slot marked dirty had a granular grain move
	# into or out of it this tick, so it must be active next tick.
	for cp in changed:
		if not still_active.has(cp):
			still_active[cp] = 0
	_active_chunks = still_active


# --- Internal ---

func _build_uniforms() -> Array:
	var u := func(binding: int, rid: RID) -> RDUniform:
		var un := RDUniform.new()
		un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		un.binding = binding
		un.add_id(rid)
		return un
	return [
		u.call(0, _orchestrator.get_voxel_buf()),
		u.call(1, _orchestrator.get_voxel_buf_b()),
		u.call(2, _orchestrator.get_chunk_table()),
		u.call(3, _global_params_buf),
		u.call(4, _orchestrator.get_material_sim_buf()),
		u.call(5, _orchestrator.get_changelog_buf()),
		u.call(6, _chunk_list_buf),
		u.call(7, _dirty_chunks_buf),
		u.call(8, _orchestrator.get_sim_active_buf()),
	]


func _upload_global_params(grid_min: Vector3i, grid_span: Vector3i, frame: int) -> void:
	var cs := _chunk_manager.chunk_size
	var p := PackedByteArray(); p.resize(32)
	p.encode_s32(0,  cs)
	p.encode_s32(4,  grid_min.x)
	p.encode_s32(8,  grid_min.y)
	p.encode_s32(12, grid_min.z)
	p.encode_s32(16, grid_span.x)
	p.encode_s32(20, grid_span.y)
	p.encode_s32(24, grid_span.z)
	p.encode_s32(28, frame)
	_compute.buffer_update(_global_params_buf, 0, p)


func _upload_chunk_list(chunk_positions: Array) -> void:
	var count := chunk_positions.size()
	# Grow the GPU buffer if needed (doubles each time).
	if count > _chunk_list_capacity:
		if _chunk_list_buf.is_valid():
			_compute.free_rid(_chunk_list_buf)
		_chunk_list_capacity = max(count, _chunk_list_capacity * 2)
		var blank := PackedByteArray(); blank.resize(_chunk_list_capacity * 12); blank.fill(0)
		_chunk_list_buf = _compute.buffer_upload(blank)
	var data := PackedByteArray(); data.resize(count * 12)
	for i in count:
		var cp: Vector3i = chunk_positions[i]
		data.encode_s32(i * 12,     cp.x)
		data.encode_s32(i * 12 + 4, cp.y)
		data.encode_s32(i * 12 + 8, cp.z)
	_compute.buffer_update(_chunk_list_buf, 0, data)


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


func _build_zero_list(active_list: Array) -> Array:
	var seen := {}
	var out  := []
	# Same offsets as fluid: self + below + 4 diagonal-below + 4 horizontal.
	# Diagonal-below covers cross-chunk diagonal falls and water lateral displacement.
	# Horizontal covers water displacement at sand's own level.
	var offsets: Array[Vector3i] = [
		Vector3i( 0,  0,  0),
		Vector3i( 0, -1,  0),
		Vector3i( 1, -1,  0), Vector3i(-1, -1,  0),
		Vector3i( 0, -1,  1), Vector3i( 0, -1, -1),
		Vector3i( 1,  0,  0), Vector3i(-1,  0,  0),
		Vector3i( 0,  0,  1), Vector3i( 0,  0, -1),
	]
	for cp: Vector3i in active_list:
		for offset in offsets:
			var nb: Vector3i = cp + offset
			if not seen.has(nb) and _chunk_manager.chunks.has(nb):
				seen[nb] = true
				out.append(nb)
	return out


func _upload_zero_chunk_list(chunk_positions: Array) -> void:
	var count := chunk_positions.size()
	if count > _zero_chunk_list_capacity:
		if _zero_chunk_list_buf.is_valid():
			_compute.free_rid(_zero_chunk_list_buf)
		_zero_chunk_list_capacity = max(count, _zero_chunk_list_capacity * 2)
		var blank := PackedByteArray(); blank.resize(_zero_chunk_list_capacity * 12); blank.fill(0)
		_zero_chunk_list_buf = _compute.buffer_upload(blank)
	var data := PackedByteArray(); data.resize(count * 12)
	for i in count:
		var cp: Vector3i = chunk_positions[i]
		data.encode_s32(i * 12,     cp.x)
		data.encode_s32(i * 12 + 4, cp.y)
		data.encode_s32(i * 12 + 8, cp.z)
	_compute.buffer_update(_zero_chunk_list_buf, 0, data)


func _build_zero_uniforms() -> Array:
	var u := func(binding: int, rid: RID) -> RDUniform:
		var un := RDUniform.new()
		un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		un.binding = binding
		un.add_id(rid)
		return un
	return [
		u.call(1, _orchestrator.get_voxel_buf_b()),
		u.call(2, _orchestrator.get_chunk_table()),
		u.call(3, _global_params_buf),
		u.call(4, _orchestrator.get_sim_active_buf()),
		u.call(6, _zero_chunk_list_buf),
	]


func _load_shader() -> String:
	var f := FileAccess.open("res://shaders/granular_sim.comp", FileAccess.READ)
	if f: return f.get_as_text()
	push_error("GranularSimulation: failed to load granular_sim.comp")
	return ""


func _load_zero_shader() -> String:
	var f := FileAccess.open("res://shaders/zero_slots.comp", FileAccess.READ)
	if f: return f.get_as_text()
	push_error("GranularSimulation: failed to load zero_slots.comp")
	return ""
