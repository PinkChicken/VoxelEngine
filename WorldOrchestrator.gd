class_name WorldOrchestrator extends Node

signal chunks_committed(chunk_positions: Array[Vector3i])
signal chunk_loaded(chunk_pos: Vector3i)

var _chunk_manager: ChunkManager
var _material_db: MaterialDB
var _runtime: SimulationRuntime
var _material_simulation: PackedInt32Array  # material_id → SimulationClass (flat cache)

# --- Inner classes ---

class VoxelMutation:
	var world_pos: Vector3i
	var chunk_pos: Vector3i
	var local_idx: int
	var material_id: int
	var fill: int
	var temp: int
	var flags: int
	var priority: int
	# Linked move pair: place mutation stores source pos; clear mutation stores dest pos.
	# Zero means "not part of a move pair" (standalone player edit, etc.).
	# Commit drops a clear if its paired place mutation did not win the destination.
	var move_partner: Vector3i = Vector3i.ZERO


# --- GPU buffer state ---

const MAX_CHANGELOG_ENTRIES := 524288  # 4 MB at 8 bytes/entry
# Raised from 131072 (2026-07-02) — a falling sphere now legitimately activates far more
# voxels per tick than any scenario this was originally sized for (a single droplet's
# scale). Measured overflow: 259316 entries in one Transport dispatch, ~2x the old cap.
# Must match MAX_CHANGELOG in every shader that writes this buffer (fluid_flux.comp,
# granular_sim.comp, edit_apply.comp) — a mismatch either wastes allocated capacity or lets
# a shader write past what's actually allocated here.

var _compute           # ComputeHelper
var _edit_dispatcher   # EditDispatcher — untyped to avoid circular class_name dependency
var _patch_pipeline: Dictionary = {}           # patch_apply.comp — applies changelog to _voxel_buf_a on GPU
var _surface_extract_pipeline: Dictionary = {} # surface_extract.comp — emits exposed voxels for collision
var _surface_out_buf: RID                      # persistent output buffer for surface extraction
var _surface_params_buf: RID                   # persistent params buffer for surface extraction

const SURFACE_OUT_MAX_PER_CHUNK  := 16384  # max surface entries per chunk (generous ceiling)
const SURFACE_EXTRACT_MAX_BATCH  := 8      # max chunks per batch dispatch
var _surface_slot_list_buf: RID            # int[SURFACE_EXTRACT_MAX_BATCH] — slot per chunk
var _voxel_buf_a: RID  # canonical: renderer reads this, simulation reads this
var _voxel_buf_b: RID  # scratch:   simulation writes here; swapped after each sim frame
var _sim_active_buf: RID  # 1 u32 per voxel slot; zeroed before sim, set on grain move
var _chunk_table: RID
var _changelog_buf: RID      # layout: [u32 count | (flat_idx u32, new_value u32) × count]
var _material_sim_buf: RID   # 256 u32s: material_id → SimulationClass (flat cache on GPU)
var _material_flow_buf: RID  # 256 floats: material_id → MaterialDef.flow (fluid_flux.comp viscosity)
var _material_density_buf: RID  # 256 floats: material_id → MaterialDef.density (density-ordered
								 # liquid stacking, 2026-07-12) — 0.0 default = not density-sorted
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

var _pending_mutations: Array[VoxelMutation] = []
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
		_init_surface_extract()
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
	if _material_sim_buf.is_valid():
		_compute.free_rid(_material_sim_buf)
	if _material_flow_buf.is_valid():
		_compute.free_rid(_material_flow_buf)
	if _material_density_buf.is_valid():
		_compute.free_rid(_material_density_buf)
	if _surface_out_buf.is_valid():
		_compute.free_rid(_surface_out_buf)
	if _surface_params_buf.is_valid():
		_compute.free_rid(_surface_params_buf)
	if _surface_slot_list_buf.is_valid():
		_compute.free_rid(_surface_slot_list_buf)
	Performance.remove_custom_monitor("Voxel/changelog_per_tick")
	Performance.remove_custom_monitor("Voxel/slot_count")
	Performance.remove_custom_monitor("Voxel/slot_capacity")
	Performance.remove_custom_monitor("Voxel/downloads_per_frame")


func _build_material_cache() -> void:
	_material_simulation.resize(256)
	_material_simulation.fill(MaterialDef.SimulationClass.NONE)
	for def in _material_db.materials:
		_material_simulation[def.id] = def.simulation
	# Upload flat cache to GPU for simulation shaders.
	if _material_sim_buf.is_valid():
		_compute.free_rid(_material_sim_buf)
	var buf := PackedByteArray(); buf.resize(256 * 4)
	for i in range(256):
		buf.encode_s32(i * 4, _material_simulation[i])
	_material_sim_buf = _compute.buffer_upload(buf)

	# Per-material flow (viscosity) cache — fluid_flux.comp binding 13. Defaults to 1.0
	# (thin/water) for every id, INCLUDING unregistered ones, so a material that somehow
	# isn't in _material_db.materials never silently freezes (a 0.0 default would zero
	# g_eff/cap_scale in the shader's viscosity formula and stop that liquid from flowing).
	if _material_flow_buf.is_valid():
		_compute.free_rid(_material_flow_buf)
	var flow_buf := PackedByteArray(); flow_buf.resize(256 * 4)
	for i in range(256):
		flow_buf.encode_float(i * 4, 1.0)
	for def in _material_db.materials:
		flow_buf.encode_float(def.id * 4, def.flow)
	_material_flow_buf = _compute.buffer_upload(flow_buf)

	# Per-material density cache (2026-07-12) — fluid_flux.comp binding 21. Defaults to
	# 0.0 (NOT density-sorted) for every id, including unregistered ones — unlike flow's
	# "default to 1.0 so nothing silently freezes," the safe default here is EXCLUSION:
	# a material nobody configured a density for should never accidentally participate
	# in density-ordered stacking just because it happens to be SIM_LIQUID.
	if _material_density_buf.is_valid():
		_compute.free_rid(_material_density_buf)
	var density_buf := PackedByteArray(); density_buf.resize(256 * 4)
	for def in _material_db.materials:
		density_buf.encode_float(def.id * 4, def.density)
	_material_density_buf = _compute.buffer_upload(density_buf)


func _process(delta: float) -> void:
	_compute.downloads_this_frame = 0
	if _edit_dispatcher:
		_edit_dispatcher.process_queue()
	if _runtime:
		_runtime.run(delta)
	commit()


# --- GPU buffer public API ---

func ensure_grid() -> void:
	if _rebuild_pending:
		_init_grid()

func get_voxel_buf() -> RID: return _voxel_buf_a
func get_voxel_buf_b() -> RID: return _voxel_buf_b
func get_sim_active_buf() -> RID: return _sim_active_buf
func get_chunk_table() -> RID: return _chunk_table
# Read-only capacity accessor — lets FluidSimulation size its privately-owned
# flux buffer without WorldOrchestrator knowing flux exists (Phase 7e Step 1).
func get_voxel_buf_capacity() -> int: return _voxel_buf_capacity
func is_grid_valid() -> bool:
	return _voxel_buf_a.is_valid() and _chunk_table.is_valid() and _material_sim_buf.is_valid()

func swap_voxel_bufs() -> void:
	var tmp := _voxel_buf_a
	_voxel_buf_a = _voxel_buf_b
	_voxel_buf_b = tmp

func get_changelog_buf() -> RID: return _changelog_buf
func get_material_sim_buf() -> RID: return _material_sim_buf
func get_material_flow_buf() -> RID: return _material_flow_buf
func get_material_density_buf() -> RID: return _material_density_buf

func reset_changelog() -> void:
	var zero := PackedByteArray(); zero.resize(4); zero.fill(0)
	_compute.buffer_update(_changelog_buf, 0, zero)


# Debug tool (world.gd binds F9): downloads the full voxel buffer once and prints total
# fill + liquid-voxel count per chunk, plus a grand total, to the console. On-demand only
# — never call this per-frame, it's a full GPU→CPU readback of every loaded chunk.
# Intended for tracking down conservation bugs in the fluid pipeline: compare the total
# before/after a single step_once() tick to see exactly how much fill a tick gained or lost.
func debug_dump_fluid_stats() -> void:
	if not _voxel_buf_a.is_valid():
		print("[FluidDebug] voxel buffer not ready")
		return
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var total_bytes := _voxel_buf_capacity * vpc * 4
	var data: PackedByteArray = _compute.download_range(_voxel_buf_a, 0, total_bytes)

	var grand_total_fill := 0
	var grand_total_voxels := 0
	var per_chunk: Array = []  # [{pos, fill, count}]

	for slot in range(_voxel_buf_capacity):
		var chunk_pos: Variant = _slot_to_chunk.get(slot, null)
		if chunk_pos == null:
			continue
		var base := slot * vpc * 4
		var chunk_fill := 0
		var chunk_count := 0
		for i in range(vpc):
			var off := base + i * 4
			var mat_id := data[off]
			if mat_id == 0:
				continue
			if _material_simulation[mat_id] != MaterialDef.SimulationClass.LIQUID:
				continue
			chunk_fill += data[off + 1]
			chunk_count += 1
		if chunk_count > 0:
			per_chunk.append({"pos": chunk_pos, "fill": chunk_fill, "count": chunk_count})
			grand_total_fill += chunk_fill
			grand_total_voxels += chunk_count

	print("[FluidDebug] === snapshot: %d chunk(s) with liquid ===" % per_chunk.size())
	for entry in per_chunk:
		var avg: float = float(entry["fill"]) / entry["count"]
		print("[FluidDebug]   chunk %s: total_fill=%d  liquid_voxels=%d  avg_fill=%.1f" % [entry["pos"], entry["fill"], entry["count"], avg])
	var grand_avg: float = (float(grand_total_fill) / grand_total_voxels) if grand_total_voxels > 0 else 0.0
	print("[FluidDebug] TOTAL fill=%d across %d liquid voxels (avg %.1f)" % [grand_total_fill, grand_total_voxels, grand_avg])

# --- Debug rewind snapshot (2026-07-07) --- a CPU-side copy of the canonical voxel buffer
# (_voxel_buf_a), so a hand-built or scripted test scenario can be captured ONCE and replayed
# repeatedly -- run forward, watch one voxel, restore, watch a different voxel, without having
# to reproduce a fiddly multi-tick scenario from scratch each time. World.gd binds Down Arrow
# to save, Up Arrow to load. Deliberately only snapshots _voxel_buf_a (the single source of
# truth per CLAUDE.md) -- _voxel_buf_b is scratch and _sim_active_buf/_changelog_buf are
# zeroed at the start of every dispatch, so none of those need to be captured for a restore to
# reproduce the exact same simulation trajectory going forward.
var _debug_voxel_snapshot: PackedByteArray = PackedByteArray()

func debug_save_voxel_snapshot() -> void:
	if not _voxel_buf_a.is_valid():
		print("[DebugSnapshot] voxel buffer not ready")
		return
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var total_bytes := _voxel_buf_capacity * vpc * 4
	_debug_voxel_snapshot = _compute.download_range(_voxel_buf_a, 0, total_bytes)
	print("[DebugSnapshot] saved %d bytes (%d slots)" % [_debug_voxel_snapshot.size(), _voxel_buf_capacity])

func debug_load_voxel_snapshot() -> void:
	if _debug_voxel_snapshot.is_empty():
		print("[DebugSnapshot] no snapshot saved yet")
		return
	if not _voxel_buf_a.is_valid():
		print("[DebugSnapshot] voxel buffer not ready")
		return
	_compute.buffer_update(_voxel_buf_a, 0, _debug_voxel_snapshot)
	# Re-run the same chunk-committed discovery a normal edit would trigger, so simulations
	# pick the restored content back up -- without this, chunks whose _active_chunks entry
	# already idled out (see fluid_simulation.gd's idle-guard) would sit quiet forever even
	# though the buffer now has fresh, un-settled content again.
	var positions: Array[Vector3i] = []
	for pos in _slot_to_chunk.values():
		positions.append(pos)
	chunks_committed.emit(positions)
	print("[DebugSnapshot] restored %d bytes to %d chunk(s)" % [_debug_voxel_snapshot.size(), positions.size()])

# Downloads the changelog count, dispatches patch_apply.comp to sync _voxel_buf_a.
# Changed-chunk discovery is now done by GranularSimulation via the dirty-chunk bitmask —
# this function no longer calls _extract_changed_chunks (was O(131k) GDScript loop).
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
	var byte_offset = slot * cs * cs * cs * 4
	var byte_count  := cs * cs * cs * 4
	var gpu_data: PackedByteArray = _compute.download_range(_voxel_buf_a, byte_offset, byte_count)
	# Repack GPU u32 format into 4-byte block_data (same layout — direct copy).
	chunk.block_data = gpu_data
	chunk.rebuild_counts_and_mask(_material_simulation)


# Applies (flat_idx, new_value) pairs from buf to block_data (CPU mirror only).
# Save-time path only — not called during normal gameplay.
# Shared by apply_changelog() and EditDispatcher._drain_edit_changelog().
# buf layout: u32 count at byte 0, then (flat_idx u32, new_value u32) pairs.
func sync_block_data_from_buf(buf: PackedByteArray, count: int) -> Array[Vector3i]:
	var cs := _chunk_manager.chunk_size
	var voxels_per_chunk := cs * cs * cs
	var affected: Dictionary = {}

	for i in range(count):
		var off := 4 + i * 8
		var flat_idx  := int(buf.decode_u32(off))
		var new_value := buf.decode_u32(off + 4)

		var slot          := flat_idx / voxels_per_chunk
		var local_byte_idx := (flat_idx % voxels_per_chunk) * Chunk.VOXEL_STRIDE

		if not _slot_to_chunk.has(slot):
			continue
		var chunk_pos: Vector3i = _slot_to_chunk[slot]
		var chunk: Chunk = _chunk_manager.chunks.get(chunk_pos)
		if not chunk:
			continue

		var mat_id   := int(new_value & 0xFF)
		var fill     := int((new_value >> 8) & 0xFF)
		var raw_temp := int((new_value >> 16) & 0xFF)
		var temp     := raw_temp if raw_temp < 128 else raw_temp - 256
		var flags    := int((new_value >> 24) & 0xFF)

		var old_sim := _material_simulation[chunk.block_data[local_byte_idx]]
		var new_sim := _material_simulation[mat_id]
		if chunk.apply_mutation(local_byte_idx, mat_id, fill, temp, flags, old_sim, new_sim):
			affected[chunk_pos] = true

	var positions: Array[Vector3i] = []
	for pos in affected:
		positions.append(pos)
	return positions


func _load_patch_shader() -> String:
	var f := FileAccess.open("res://shaders/patch_apply.comp", FileAccess.READ)
	if not f:
		push_error("WorldOrchestrator: cannot open patch_apply.comp")
		return ""
	return f.get_as_text()


func _init_surface_extract() -> void:
	if not _compute.has_device():
		return
	var src := _load_surface_extract_shader()
	if src.is_empty():
		push_error("WorldOrchestrator: cannot open surface_extract.comp")
		return
	_surface_extract_pipeline = _compute.compile_shader(src, 4)
	if _surface_extract_pipeline.is_empty():
		push_error("WorldOrchestrator: surface_extract.comp compile failed")
		return
	# Output: SURFACE_EXTRACT_MAX_BATCH chunks, each occupying (1 + SURFACE_OUT_MAX_PER_CHUNK) u32 words.
	var stride   := 1 + SURFACE_OUT_MAX_PER_CHUNK
	var out_size := SURFACE_EXTRACT_MAX_BATCH * stride * 4
	var init_data := PackedByteArray(); init_data.resize(out_size); init_data.fill(0)
	_surface_out_buf = _compute.buffer_upload(init_data)
	var param_data := PackedByteArray(); param_data.resize(16); param_data.fill(0)
	_surface_params_buf = _compute.buffer_upload(param_data)
	var slot_data := PackedByteArray(); slot_data.resize(SURFACE_EXTRACT_MAX_BATCH * 4); slot_data.fill(0xFF)
	_surface_slot_list_buf = _compute.buffer_upload(slot_data)


func _load_surface_extract_shader() -> String:
	var f := FileAccess.open("res://shaders/surface_extract.comp", FileAccess.READ)
	if not f:
		return ""
	return f.get_as_text()


# Batch surface extraction: one GPU dispatch + sync covers all N chunks.
# Returns {chunk_pos: PackedByteArray} where each value is u32 entries
# packed as bits[15:0]=local_flat_idx, bits[23:16]=mat_id.
func extract_surface_voxels_batch(positions: Array[Vector3i]) -> Dictionary:
	var n := positions.size()
	if n == 0 or _surface_extract_pipeline.is_empty():
		return {}
	if not _voxel_buf_a.is_valid() or not _surface_out_buf.is_valid() \
			or not _material_sim_buf.is_valid() or not _sim_active_buf.is_valid() \
			or not _surface_slot_list_buf.is_valid():
		return {}

	n = min(n, SURFACE_EXTRACT_MAX_BATCH)
	var cs     := _chunk_manager.chunk_size
	var stride := 1 + SURFACE_OUT_MAX_PER_CHUNK  # u32 words per chunk in output buffer

	# Upload slot list and zero per-chunk count fields.
	var slot_data := PackedByteArray(); slot_data.resize(n * 4)
	var zero4 := PackedByteArray([0, 0, 0, 0])
	for i in n:
		var slot = _chunk_to_slot.get(positions[i], -1)
		slot_data.encode_s32(i * 4, slot)
		_compute.buffer_update(_surface_out_buf, i * stride * 4, zero4)
	_compute.buffer_update(_surface_slot_list_buf, 0, slot_data)

	# Upload params: chunk_size, max_per_chunk, chunk_count, pad.
	var param_data := PackedByteArray(); param_data.resize(16)
	param_data.encode_s32(0, cs)
	param_data.encode_s32(4, SURFACE_OUT_MAX_PER_CHUNK)
	param_data.encode_s32(8, n)
	param_data.encode_s32(12, 0)
	_compute.buffer_update(_surface_params_buf, 0, param_data)

	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u0.binding = 0; u0.add_id(_voxel_buf_a)
	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u1.binding = 1; u1.add_id(_surface_out_buf)
	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u2.binding = 2; u2.add_id(_surface_params_buf)
	var u3 := RDUniform.new()
	u3.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u3.binding = 3; u3.add_id(_material_sim_buf)
	var u4 := RDUniform.new()
	u4.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u4.binding = 4; u4.add_id(_sim_active_buf)
	var u5 := RDUniform.new()
	u5.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u5.binding = 5; u5.add_id(_surface_slot_list_buf)

	_compute.run(_surface_extract_pipeline, [u0, u1, u2, u3, u4, u5], n * 4, 4, 4)
	_compute.submit()
	_compute.sync()

	# One download covers the entire output for all N chunks.
	var raw = _compute.download_range(_surface_out_buf, 0, n * stride * 4)

	var result: Dictionary = {}
	for i in n:
		var base_bytes := i * stride * 4
		var count := int(raw.decode_u32(base_bytes))
		if count == 0:
			continue
		count = min(count, SURFACE_OUT_MAX_PER_CHUNK)
		result[positions[i]] = raw.slice(base_bytes + 4, base_bytes + 4 + count * 4)
	return result


# --- Edit convenience wrappers (used by ChunkManager — avoids circular class_name dep) ---

func edit_add_voxel(pos: Vector3i, mat: int) -> void:
	if not _edit_dispatcher: return
	_edit_dispatcher.apply_immediate(
		EditDispatcher.EditRequest.new(pos, EditDispatcher.EditShape.BOX,
			Vector4(0, 0, 0, 0), EditDispatcher.EditOp.ADD, mat))
	_write_voxel_block_data(pos, mat, 255)

func edit_remove_voxel(pos: Vector3i) -> void:
	if not _edit_dispatcher: return
	_edit_dispatcher.apply_immediate(
		EditDispatcher.EditRequest.new(pos, EditDispatcher.EditShape.BOX,
			Vector4(0, 0, 0, 0), EditDispatcher.EditOp.REMOVE))
	_write_voxel_block_data(pos, 0, 0)

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
func _write_voxel_block_data(world_pos: Vector3i, mat_id: int, fill: int) -> void:
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
	if byte_idx >= 0 and byte_idx + 1 < chunk.block_data.size():
		chunk.block_data[byte_idx]     = mat_id
		chunk.block_data[byte_idx + 1] = fill


# --- GPU buffer management ---

func _free_voxel_buffers() -> void:
	if _voxel_buf_a.is_valid():
		_compute.free_rid(_voxel_buf_a)
		_voxel_buf_a = RID()
	if _voxel_buf_b.is_valid():
		_compute.free_rid(_voxel_buf_b)
		_voxel_buf_b = RID()
	if _sim_active_buf.is_valid():
		_compute.free_rid(_sim_active_buf)
		_sim_active_buf = RID()
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
	var voxel_bytes := capacity * voxels_per_chunk * 4
	var voxel_data  := PackedByteArray(); voxel_data.resize(voxel_bytes); voxel_data.fill(0)
	_voxel_buf_a    = _compute.buffer_upload(voxel_data)
	_voxel_buf_b    = _compute.buffer_upload(voxel_data)
	_sim_active_buf = _compute.buffer_upload(voxel_data)
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
		var old_bytes := _voxel_buf_capacity * cs * cs * cs * 4
		var new_capacity := _voxel_buf_capacity * 2
		var new_bytes := new_capacity * cs * cs * cs * 4
		_voxel_buf_a = _compute.buffer_grow(_voxel_buf_a, old_bytes, new_bytes)
		_voxel_buf_b = _compute.buffer_grow(_voxel_buf_b, old_bytes, new_bytes)
		_sim_active_buf = _compute.buffer_grow(_sim_active_buf, old_bytes, new_bytes)
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
	_compute.buffer_update(_voxel_buf_a, slot * cs * cs * cs * 4, chunk.block_data)
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
	_compute.buffer_update(_voxel_buf_a, slot * cs * cs * cs * 4, chunk.block_data)


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
	var zero_data := PackedByteArray(); zero_data.resize(cs * cs * cs * 4); zero_data.fill(0)
	_compute.buffer_update(_voxel_buf_a, slot * cs * cs * cs * 4, zero_data)

	_free_slots.append(slot)
	_chunk_to_slot.erase(chunk_pos)
	_slot_to_chunk.erase(slot)
	_voxel_buf_count -= 1
	_chunks_loaded -= 1


# --- Public API ---

func submit_mutation_direct(chunk_pos: Vector3i, local_idx: int, world_pos: Vector3i,
		mat_id: int, fill: int = 255, temp: int = 0, flags: int = 0, priority: int = 0) -> void:
	var m := VoxelMutation.new()
	m.world_pos = world_pos
	m.chunk_pos = chunk_pos
	m.local_idx = local_idx
	m.material_id = mat_id
	m.fill = fill
	m.temp = temp
	m.flags = flags
	m.priority = priority
	_pending_mutations.append(m)


func submit_batch(mutations: Array) -> void:
	_pending_mutations.append_array(mutations)


func submit_mutation(world_pos: Vector3i, mat_id: int, fill: int = 255,
		temp: int = 0, flags: int = 0, priority: int = 0) -> void:
	var cm := _chunk_manager
	var chunk_pos := cm.voxel_to_chunk_position(world_pos)
	if not cm.chunks.has(chunk_pos):
		return
	var local_pos := cm.global_to_local(world_pos)
	var m := VoxelMutation.new()
	m.world_pos = world_pos
	m.chunk_pos = chunk_pos
	m.local_idx = (local_pos.z * cm.chunk_size * cm.chunk_size + local_pos.y * cm.chunk_size + local_pos.x) * Chunk.VOXEL_STRIDE
	m.material_id = mat_id
	m.fill = fill
	m.temp = temp
	m.flags = flags
	m.priority = priority
	_pending_mutations.append(m)


# --- Commit ---

func commit() -> void:
	if _pending_mutations.is_empty():
		return

	var best: Dictionary = {}  # Vector3i -> VoxelMutation
	for m in _pending_mutations:
		if not best.has(m.world_pos) or m.priority > best[m.world_pos].priority:
			best[m.world_pos] = m
	_pending_mutations.clear()

	# Drop conditional clears whose paired place mutation lost the destination.
	# This prevents the losing grain from being deleted when two grains compete
	# for the same empty cell in the same frame.
	var to_drop: Array[Vector3i] = []
	for pos in best:
		var m: VoxelMutation = best[pos]
		if m.material_id == 0 and m.move_partner != Vector3i.ZERO:
			var dest_winner: VoxelMutation = best.get(m.move_partner)
			if dest_winner == null or dest_winner.move_partner != pos:
				to_drop.append(pos)
	for pos in to_drop:
		best.erase(pos)

	var affected_chunks: Dictionary = {}

	for m in best.values():
		var chunk = _chunk_manager.chunks.get(m.chunk_pos)
		if not chunk:
			continue
		var old_sim := _material_simulation[chunk.block_data[m.local_idx]]
		var new_sim := _material_simulation[m.material_id]
		if chunk.apply_mutation(m.local_idx, m.material_id, m.fill, m.temp, m.flags, old_sim, new_sim):
			affected_chunks[m.chunk_pos] = true

	if affected_chunks.is_empty():
		return

	# One full-chunk upload per affected chunk — passes block_data by ref, no GDScript copy.
	for chunk_pos in affected_chunks:
		_update_chunk_in_grid(chunk_pos)

	var positions: Array[Vector3i] = []
	for pos in affected_chunks:
		positions.append(pos)
	chunks_committed.emit(positions)
