class_name WorldOrchestrator extends Node

signal chunk_loaded(chunk_pos: Vector3i)
# MLS-MPM migration (2026-08-12) — individual placed-voxel positions/materials (not just
# chunk positions), emitted by EditDispatcher._dispatch() (declared here, emitted by the
# writer that has _orchestrator). MPMSimulation listens for this to promote newly-placed
# sand/water into particles.
signal voxels_placed(positions: Array[Vector3i], material_ids: Array[int])

var _chunk_manager: ChunkManager
var _material_db: MaterialDB
var _runtime: SimulationRuntime
var _material_simulation: PackedInt32Array  # material_id → SimulationClass (flat cache)

# --- GPU buffer state ---

const MAX_CHANGELOG_ENTRIES := 524288  # 4 MB at 8 bytes/entry
# Raised from 131072 (2026-07-02) — a falling sphere now legitimately activates far more
# voxels per tick than any scenario this was originally sized for (a single droplet's
# scale). Measured overflow: 259316 entries in one Transport dispatch, ~2x the old cap.
# Must match MAX_CHANGELOG in mpm/readout.comp (the only shader that writes this buffer) —
# a mismatch either wastes allocated capacity or lets a shader write past what's allocated.

var _compute           # ComputeHelper
var _edit_dispatcher   # EditDispatcher — untyped to avoid circular class_name dependency
var _patch_pipeline: Dictionary = {}           # patch_apply.comp — applies changelog to _voxel_buf_a on GPU
var _voxel_buf_a: RID  # canonical: renderer reads this, simulation reads this
var _chunk_table: RID
var _changelog_buf: RID      # layout: [u32 count | (flat_idx u32, new_value u32) × count]
var _grid_min: Vector3i = Vector3i.ZERO
var _grid_span: Vector3i = Vector3i.ZERO
var _chunk_to_slot: Dictionary = {}   # Vector3i → slot index
var _slot_to_chunk: Dictionary = {}   # slot index → Vector3i (reverse map for changelog)
var _voxel_buf_capacity: int = 0
var _voxel_buf_count: int = 0   # used slots (= loaded chunk count)
var _chunks_loaded: int = 0
var _rebuild_pending: bool = true
var _free_slots: Array[int] = []  # pool of available voxel buffer slot indices


# --- Simulation state ---

var _last_changelog_count: int = 0


func _init() -> void:
	# Pre-size before _ready() so background chunk-gen threads can safely
	# access _material_simulation before the deferred cache build completes.
	_material_simulation.resize(256)
	_material_simulation.fill(MaterialDef.SimulationClass.NONE)


func _ready() -> void:
	_chunk_manager = get_node("/root/world/ChunkManager")
	_material_db = get_node("/root/world/MaterialDB")
	_runtime = get_node_or_null("SimulationRuntime")
	call_deferred("_build_material_cache")

	_compute = preload("res://Utils/compute_helper.gd").new()
	if _compute.has_device():
		_patch_pipeline = _compute.compile_shader(_load_patch_shader(), 4)
		if _patch_pipeline.is_empty():
			push_error("WorldOrchestrator: patch_apply.comp compile failed")
		_init_grid()  # empty grid immediately; chunks stream in via _add_chunk_to_grid
	_chunk_manager.slab_invalidated.connect(_init_grid)  # R-key regenerate: reset grid immediately

	_edit_dispatcher = preload("res://simulations/edit_dispatcher.gd").new()
	add_child(_edit_dispatcher)

	Performance.add_custom_monitor("Voxel/changelog_per_tick", func(): return _last_changelog_count)
	Performance.add_custom_monitor("Voxel/slot_count", func(): return _voxel_buf_count)
	Performance.add_custom_monitor("Voxel/slot_capacity", func(): return _voxel_buf_capacity)
	# downloads_per_frame includes the cursor readback (32 bytes/frame) from RaymarchRenderer — floor is 1.
	Performance.add_custom_monitor("Voxel/downloads_per_frame", func(): return _compute.downloads_this_frame)


func _exit_tree() -> void:
	_free_voxel_buffers()
	Performance.remove_custom_monitor("Voxel/changelog_per_tick")
	Performance.remove_custom_monitor("Voxel/slot_count")
	Performance.remove_custom_monitor("Voxel/slot_capacity")
	Performance.remove_custom_monitor("Voxel/downloads_per_frame")


func _build_material_cache() -> void:
	_material_simulation.resize(256)
	_material_simulation.fill(MaterialDef.SimulationClass.NONE)
	for def in _material_db.materials:
		_material_simulation[def.id] = def.simulation


func _process(delta: float) -> void:
	_compute.downloads_this_frame = 0
	if _edit_dispatcher:
		_edit_dispatcher.process_queue()
	if _runtime:
		_runtime.run(delta)


# --- GPU buffer public API ---

func ensure_grid() -> void:
	if _rebuild_pending:
		_init_grid()

func get_voxel_buf() -> RID: return _voxel_buf_a
func get_chunk_table() -> RID: return _chunk_table
func get_voxel_buf_capacity() -> int: return _voxel_buf_capacity
func is_grid_valid() -> bool:
	return _voxel_buf_a.is_valid() and _chunk_table.is_valid()

func get_changelog_buf() -> RID: return _changelog_buf

func reset_changelog() -> void:
	var zero := PackedByteArray(); zero.resize(4); zero.fill(0)
	_compute.buffer_update(_changelog_buf, 0, zero)

# Downloads the changelog count, dispatches patch_apply.comp to sync _voxel_buf_a.
func apply_changelog() -> void:
	if not _changelog_buf.is_valid():
		return
	var count_buf: PackedByteArray = _compute.download_range(_changelog_buf, 0, 4)
	var count: int = count_buf.decode_u32(0)
	if count == 0:
		return
	if count > MAX_CHANGELOG_ENTRIES:
		push_warning("WorldOrchestrator: changelog overflow (%d entries); clamping to %d" % [count, MAX_CHANGELOG_ENTRIES])
		count = MAX_CHANGELOG_ENTRIES
	_last_changelog_count = count

	if not _patch_pipeline.is_empty() and _voxel_buf_a.is_valid():
		var workgroups := int(ceil(float(count) / 64.0))
		var u0 := RDUniform.new()
		u0.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u0.binding = 0; u0.add_id(_voxel_buf_a)
		var u1 := RDUniform.new()
		u1.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u1.binding = 1; u1.add_id(_changelog_buf)
		_compute.run(_patch_pipeline, [u0, u1], workgroups, 1, 1)
		_compute.submit()
		_compute.sync()


# Lightweight changelog drain — no block_data writes, O(count) slot lookups.
# Updates simulation masks from new_value so the runtime keeps scheduling correctly.
# Marks every affected chunk in _gpu_dirty for save-time sync.
func _extract_changed_chunks(buf: PackedByteArray, count: int) -> Array[Vector3i]:
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var seen: Dictionary = {}
	var masked_bits: Dictionary = {}
	var slot_to_chunk := _slot_to_chunk
	var chunk_manager := _chunk_manager
	var material_sim := _material_simulation
	for i in range(count):
		var base := 4 + i * 8
		var flat_idx  := int(buf.decode_u32(base))
		var new_value := buf.decode_u32(base + 4)
		var slot: int = flat_idx / vpc
		var chunk_pos: Variant = slot_to_chunk.get(slot, null)
		if chunk_pos == null:
			continue
		seen[chunk_pos] = true
		if not masked_bits.has(chunk_pos):
			masked_bits[chunk_pos] = 0
		var sim := material_sim[int(new_value & 0xFF)]
		var bit := MaterialDef.SIM_BITS[sim] if sim > 0 else 0
		if bit != 0 and (masked_bits[chunk_pos] & bit) == 0:
			masked_bits[chunk_pos] |= bit
			var chunk: Chunk = chunk_manager.chunks.get(chunk_pos)
			if chunk:
				chunk.mark_simulation_active(sim)
	var out: Array[Vector3i] = []
	for cp in seen:
		out.append(cp)
	return out


# Public alias used by EditDispatcher (avoids circular class_name dependency).
func extract_changed_chunks(buf: PackedByteArray, count: int) -> Array[Vector3i]:
	return _extract_changed_chunks(buf, count)


# Syncs block_data for one chunk from GPU. Called at save time for every dirty chunk.
func sync_chunk_to_cpu(chunk_pos: Vector3i) -> void:
	var slot = _chunk_to_slot.get(chunk_pos, -1)
	if slot < 0:
		return
	var chunk: Chunk = _chunk_manager.chunks.get(chunk_pos)
	if not chunk:
		return
	var cs := _chunk_manager.chunk_size
	var byte_offset = slot * cs * cs * cs
	var byte_count  := cs * cs * cs
	var gpu_data: PackedByteArray = _compute.download_range(_voxel_buf_a, byte_offset, byte_count)
	# 1 byte per voxel — the GPU buffer holds the same bytes (packed 4 to a u32), so this is
	# a direct copy back into block_data.
	chunk.block_data = gpu_data
	chunk.rebuild_simulation_mask(_material_simulation)


func _load_patch_shader() -> String:
	var f := FileAccess.open("res://shaders/patch_apply.comp", FileAccess.READ)
	if not f:
		push_error("WorldOrchestrator: cannot open patch_apply.comp")
		return ""
	return f.get_as_text()


# --- Edit convenience wrappers (used by ChunkManager — avoids circular class_name dep) ---

func edit_add_voxel(pos: Vector3i, mat: int) -> void:
	if not _edit_dispatcher: return
	_edit_dispatcher.apply_immediate(
		EditDispatcher.EditRequest.new(pos, EditDispatcher.EditShape.BOX,
			Vector4(0, 0, 0, 0), EditDispatcher.EditOp.ADD, mat))
	_write_voxel_block_data(pos, mat)

func edit_remove_voxel(pos: Vector3i) -> void:
	if not _edit_dispatcher: return
	_edit_dispatcher.apply_immediate(
		EditDispatcher.EditRequest.new(pos, EditDispatcher.EditShape.BOX,
			Vector4(0, 0, 0, 0), EditDispatcher.EditOp.REMOVE))
	_write_voxel_block_data(pos, 0)

func edit_add_sphere(center: Vector3i, radius: float, mat: int) -> void:
	if not _edit_dispatcher: return
	_edit_dispatcher.queue_edit(
		EditDispatcher.EditRequest.new(center, EditDispatcher.EditShape.SPHERE,
			Vector4(radius, 0, 0, 0), EditDispatcher.EditOp.ADD, mat))

func edit_remove_sphere(center: Vector3i, radius: float) -> void:
	if not _edit_dispatcher: return
	_edit_dispatcher.queue_edit(
		EditDispatcher.EditRequest.new(center, EditDispatcher.EditShape.SPHERE,
			Vector4(radius, 0, 0, 0), EditDispatcher.EditOp.REMOVE))

# General box edit (2026-07-04, U-tube test-scene builder) — same immediate-dispatch BOX
# path as edit_add_voxel/edit_remove_voxel above, generalized to an arbitrary half-extent
# so a script can carve/fill exact rectangular regions instead of looping per-voxel.
# Deliberately skips _write_voxel_block_data (same as the existing sphere-edit wrappers
# above) -- block_data is only a CPU cache refreshed at chunk generation/save time, not a
# runtime source of truth (see CLAUDE.md), so bulk edits not syncing it is established
# precedent, not a gap introduced here.
func edit_box(center: Vector3i, half_extent: Vector3i, op: int, mat: int = 0) -> void:
	if not _edit_dispatcher: return
	_edit_dispatcher.apply_immediate(
		EditDispatcher.EditRequest.new(center, EditDispatcher.EditShape.BOX,
			Vector4(half_extent.x, half_extent.y, half_extent.z, 0), op, mat))


func get_grid_min() -> Vector3i: return _grid_min
func get_grid_span() -> Vector3i: return _grid_span
func get_chunks_loaded() -> int: return _chunks_loaded
func is_rebuild_pending() -> bool: return _rebuild_pending


# Reads the material_id of a voxel from block_data — zero GPU cost.
# block_data is accurate for initial terrain and single-voxel player edits
# (kept in sync by _write_voxel_block_data). Simulation-moved sand may be stale
# but is acceptable: projectiles pass through falling grains, not solid terrain.
func query_voxel_material(world_pos: Vector3i) -> int:
	var cs := _chunk_manager.chunk_size
	var cp := Vector3i(
		floori(float(world_pos.x) / cs),
		floori(float(world_pos.y) / cs),
		floori(float(world_pos.z) / cs))
	var chunk: Chunk = _chunk_manager.chunks.get(cp)
	if not chunk:
		return 0
	var lp := world_pos - cp * cs
	return chunk.get_material_id((lp.z * cs * cs + lp.y * cs + lp.x) * Chunk.VOXEL_STRIDE)


# Write a material change directly into block_data for single-voxel player edits,
# keeping the CPU mirror current so query_voxel_material sees the change immediately.
func _write_voxel_block_data(world_pos: Vector3i, mat_id: int) -> void:
	var cs := _chunk_manager.chunk_size
	var cp := Vector3i(
		floori(float(world_pos.x) / cs),
		floori(float(world_pos.y) / cs),
		floori(float(world_pos.z) / cs))
	var chunk: Chunk = _chunk_manager.chunks.get(cp)
	if not chunk:
		return
	var lp := world_pos - cp * cs
	var byte_idx := (lp.z * cs * cs + lp.y * cs + lp.x) * Chunk.VOXEL_STRIDE
	if byte_idx >= 0 and byte_idx < chunk.block_data.size():
		chunk.block_data[byte_idx] = mat_id


# --- GPU buffer management ---

func _free_voxel_buffers() -> void:
	if _voxel_buf_a.is_valid():
		_compute.free_rid(_voxel_buf_a)
		_voxel_buf_a = RID()
	if _chunk_table.is_valid():
		_compute.free_rid(_chunk_table)
		_chunk_table = RID()
	if _changelog_buf.is_valid():
		_compute.free_rid(_changelog_buf)
		_changelog_buf = RID()
	_voxel_buf_capacity = 0
	_voxel_buf_count = 0
	_free_slots.clear()


func _init_grid() -> void:
	_free_voxel_buffers()
	var cm := _chunk_manager
	var cs := cm.chunk_size
	var nc := cm.world_chunks  # Vector3i — total world in chunks

	# World is centered at origin: chunks run from -nc/2 to nc/2-1 in each axis.
	_grid_min  = Vector3i(-nc.x / 2, -nc.y / 2, -nc.z / 2)
	_grid_span = nc

	var total_table_slots := _grid_span.x * _grid_span.y * _grid_span.z
	var voxels_per_chunk  := cs * cs * cs

	# Pre-allocate enough slots for the full view_distance ring × Y chunks so buffer_grow
	# never fires during normal streaming. Grows only if the player somehow loads more.
	var expected := (2 * cm.view_distance + 1) * (2 * cm.view_distance + 1) * nc.y
	var capacity := 64
	while capacity < expected:
		capacity *= 2
	var voxel_bytes := capacity * voxels_per_chunk
	var voxel_data  := PackedByteArray(); voxel_data.resize(voxel_bytes); voxel_data.fill(0)
	_voxel_buf_a    = _compute.buffer_upload(voxel_data)
	_voxel_buf_capacity = capacity

	# Seed free slot pool (pop_back gives lowest index first for cache locality).
	for i in range(capacity - 1, -1, -1):
		_free_slots.append(i)

	# Chunk table: all -1 (no chunks loaded yet). Shaders skip slot < 0.
	var table_data := PackedByteArray()
	table_data.resize(total_table_slots * 4)
	table_data.fill(0xFF)  # -1 as s32
	_chunk_table = _compute.buffer_upload(table_data)

	var changelog_data := PackedByteArray()
	changelog_data.resize(4 + MAX_CHANGELOG_ENTRIES * 8)
	changelog_data.fill(0)
	_changelog_buf = _compute.buffer_upload(changelog_data)

	_chunk_to_slot.clear()
	_slot_to_chunk.clear()
	_voxel_buf_count = 0
	_chunks_loaded   = 0
	_rebuild_pending = false
	# Chunks stream in individually via _add_chunk_to_grid, called by ChunkManager.


func _chunk_flat_idx(chunk_pos: Vector3i) -> int:
	var lx := chunk_pos.x - _grid_min.x
	var ly := chunk_pos.y - _grid_min.y
	var lz := chunk_pos.z - _grid_min.z
	return lx + ly * _grid_span.x + lz * _grid_span.x * _grid_span.y


func _add_chunk_to_grid(chunk_pos: Vector3i) -> void:
	var chunk = _chunk_manager.chunks.get(chunk_pos)
	if not chunk:
		return
	if _chunk_to_slot.has(chunk_pos):
		_update_chunk_in_grid(chunk_pos)
		return

	# Grow voxel buffer if the free pool is exhausted.
	var cs := _chunk_manager.chunk_size
	if _free_slots.is_empty():
		var old_bytes := _voxel_buf_capacity * cs * cs * cs
		var new_capacity := _voxel_buf_capacity * 2
		var new_bytes := new_capacity * cs * cs * cs
		_voxel_buf_a = _compute.buffer_grow(_voxel_buf_a, old_bytes, new_bytes)
		for i in range(new_capacity - 1, _voxel_buf_capacity - 1, -1):
			_free_slots.append(i)
		_voxel_buf_capacity = new_capacity

	var slot := _free_slots.pop_back() as int
	_chunk_to_slot[chunk_pos] = slot
	_slot_to_chunk[slot] = chunk_pos
	_voxel_buf_count += 1
	_chunks_loaded += 1

	var lx := chunk_pos.x - _grid_min.x
	var ly := chunk_pos.y - _grid_min.y
	var lz := chunk_pos.z - _grid_min.z
	if lx < 0 or ly < 0 or lz < 0 or lx >= _grid_span.x or ly >= _grid_span.y or lz >= _grid_span.z:
		_rebuild_pending = true
		return
	var flat_idx := lx + ly * _grid_span.x + lz * _grid_span.x * _grid_span.y
	var slot_bytes := PackedByteArray(); slot_bytes.resize(4)
	slot_bytes.encode_s32(0, slot)
	_compute.buffer_update(_chunk_table, flat_idx * 4, slot_bytes)
	_compute.buffer_update(_voxel_buf_a, slot * cs * cs * cs, chunk.block_data)
	chunk_loaded.emit(chunk_pos)


func _update_chunk_in_grid(chunk_pos: Vector3i) -> void:
	if not _voxel_buf_a.is_valid():
		_rebuild_pending = true
		return
	var slot = _chunk_to_slot.get(chunk_pos, -1)
	if slot < 0:
		_rebuild_pending = true
		return
	var chunk = _chunk_manager.chunks.get(chunk_pos)
	if not chunk:
		return
	var cs := _chunk_manager.chunk_size
	_compute.buffer_update(_voxel_buf_a, slot * cs * cs * cs, chunk.block_data)


func _remove_chunk_from_grid(chunk_pos: Vector3i) -> void:
	var slot = _chunk_to_slot.get(chunk_pos, -1)
	if slot < 0:
		return

	# Reset chunk table entry to -1 sentinel.
	var lx := chunk_pos.x - _grid_min.x
	var ly := chunk_pos.y - _grid_min.y
	var lz := chunk_pos.z - _grid_min.z
	if lx >= 0 and ly >= 0 and lz >= 0 and lx < _grid_span.x and ly < _grid_span.y and lz < _grid_span.z:
		var flat_idx := lx + ly * _grid_span.x + lz * _grid_span.x * _grid_span.y
		var sentinel := PackedByteArray(); sentinel.resize(4); sentinel.fill(0xFF)
		_compute.buffer_update(_chunk_table, flat_idx * 4, sentinel)

	# Zero and reclaim the voxel buffer slot.
	var cs := _chunk_manager.chunk_size
	var zero_data := PackedByteArray(); zero_data.resize(cs * cs * cs); zero_data.fill(0)
	_compute.buffer_update(_voxel_buf_a, slot * cs * cs * cs, zero_data)

	_free_slots.append(slot)
	_chunk_to_slot.erase(chunk_pos)
	_slot_to_chunk.erase(slot)
	_voxel_buf_count -= 1
	_chunks_loaded -= 1
