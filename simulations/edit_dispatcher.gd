class_name EditDispatcher extends Node

# ── Enums ──────────────────────────────────────────────────────────────────────

enum EditShape { SPHERE = 0, BOX = 1, CYLINDER = 2, CAPSULE = 3 }
enum EditOp    { ADD = 0, REMOVE = 1, REPLACE = 2 }

# ── EditRequest ────────────────────────────────────────────────────────────────

class EditRequest:
	var center:       Vector3i
	var shape_type:   int          # EditShape
	var shape_params: Vector4      # SPHERE:(r,0,0,0)  BOX:(hx,hy,hz,0)
	                               # CYLINDER:(r,h,0,0)  CAPSULE:(r,half_h,0,0)
	var operation:    int          # EditOp
	var material_id:  int
	var priority:     int = 0

	func _init(
		p_center:    Vector3i,
		p_shape:     int,
		p_params:    Vector4,
		p_op:        int,
		p_mat:       int  = 0,
		p_priority:  int  = 0
	) -> void:
		center      = p_center
		shape_type  = p_shape
		shape_params = p_params
		operation   = p_op
		material_id = p_mat
		priority    = p_priority

# ── Constants ──────────────────────────────────────────────────────────────────

# 16 MB at 8 bytes/entry. Large enough for a 256³ sphere (~8.6 M voxels max).
const MAX_EDIT_CHANGELOG := 131072

# Voxel-volume budget consumed by _process_queue per frame (32³).
const EDIT_BUDGET_VOXELS := 32768

# ep buffer layout (int-indexed): 23 ints × 4 bytes → allocate 96 bytes.
const EP_BYTES := 96

# ── State ─────────────────────────────────────────────────────────────────────

var _orchestrator: WorldOrchestrator
var _compute           # ComputeHelper (shared with orchestrator)
var _pipeline: Dictionary = {}
var _params_buf:    RID
var _changelog_buf: RID
var _edit_queue: Array = []  # Array[EditRequest]

# ── Lifecycle ─────────────────────────────────────────────────────────────────

func _ready() -> void:
	_orchestrator = get_parent() as WorldOrchestrator
	_compute = _orchestrator._compute
	if not _compute or not _compute.has_device():
		push_warning("EditDispatcher: no compute device — GPU edits disabled")
		return
	_load_shader()
	_alloc_buffers()


func _exit_tree() -> void:
	if _params_buf.is_valid():    _compute.free_rid(_params_buf)
	if _changelog_buf.is_valid(): _compute.free_rid(_changelog_buf)


# ── Public API ────────────────────────────────────────────────────────────────

# Dispatch the edit immediately (current frame). Used for all player tools.
func apply_immediate(req: EditRequest) -> void:
	if not _ready_to_dispatch(): return
	_dispatch(req)


# Enqueue for rate-limited dispatch (spells, explosions, procedural stamps).
func queue_edit(req: EditRequest) -> void:
	_edit_queue.append(req)


# Called by WorldOrchestrator._process() before simulation runs.
func process_queue() -> void:
	if _edit_queue.is_empty(): return
	var budget := EDIT_BUDGET_VOXELS
	while not _edit_queue.is_empty() and budget > 0:
		var req: EditRequest = _edit_queue[0]
		var vol := _aabb_volume(req)
		# If a single request exceeds the full budget, dispatch it anyway.
		if vol > budget and budget < EDIT_BUDGET_VOXELS:
			break
		_edit_queue.pop_front()
		if _ready_to_dispatch():
			_dispatch(req)
		budget -= vol

# ── Helpers ───────────────────────────────────────────────────────────────────

func _ready_to_dispatch() -> bool:
	return (
		not _pipeline.is_empty()
		and _params_buf.is_valid()
		and _changelog_buf.is_valid()
		and _orchestrator.is_grid_valid()
	)


func _aabb_volume(req: EditRequest) -> int:
	var s := req.shape_params
	match req.shape_type:
		EditShape.BOX:
			return (int(s.x) * 2 + 1) * (int(s.y) * 2 + 1) * (int(s.z) * 2 + 1)
		EditShape.CYLINDER:
			var d := int(s.x) * 2 + 1
			return d * d * (int(s.y) * 2 + 1)
		_:
			var d := int(s.x) * 2 + 1
			return d * d * d


func _aabb_half(req: EditRequest) -> Vector3i:
	var s := req.shape_params
	match req.shape_type:
		EditShape.BOX:
			return Vector3i(ceili(s.x), ceili(s.y), ceili(s.z))
		EditShape.CYLINDER:
			return Vector3i(ceili(s.x), ceili(s.y), ceili(s.x))
		EditShape.CAPSULE:
			return Vector3i(ceili(s.x), ceili(s.y + s.x), ceili(s.x))
		_:  # SPHERE
			var r := ceili(s.x)
			return Vector3i(r, r, r)


func _dispatch(req: EditRequest) -> void:
	var half      := _aabb_half(req)
	var aabb_min  := req.center - half
	var aabb_size := half * 2 + Vector3i.ONE

	_upload_params(req, aabb_min, aabb_size)

	# Reset changelog counter.
	var zero := PackedByteArray(); zero.resize(4); zero.fill(0)
	_compute.buffer_update(_changelog_buf, 0, zero)

	var u0 := RDUniform.new()
	u0.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u0.binding = 0; u0.add_id(_orchestrator.get_voxel_buf())

	var u1 := RDUniform.new()
	u1.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u1.binding = 1; u1.add_id(_changelog_buf)

	var u2 := RDUniform.new()
	u2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u2.binding = 2; u2.add_id(_orchestrator.get_chunk_table())

	var u3 := RDUniform.new()
	u3.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u3.binding = 3; u3.add_id(_params_buf)

	var wg_x := ceili(aabb_size.x / 8.0)
	var wg_y := ceili(aabb_size.y / 8.0)
	var wg_z := ceili(aabb_size.z / 8.0)
	_compute.run(_pipeline, [u0, u1, u2, u3], wg_x, wg_y, wg_z)
	_compute.submit()
	_compute.sync()

	var count_buf: PackedByteArray = _compute.download_range(_changelog_buf, 0, 4)
	var count: int = count_buf.decode_u32(0)
	if count == 0:
		return
	count = mini(count, MAX_EDIT_CHANGELOG)
	var raw: PackedByteArray = _compute.download_range(_changelog_buf, 0, 4 + count * 8)

	# Updates simulation masks (MPM chunk discovery). The returned chunk list's only consumer
	# was the collision surface cache, which no longer exists.
	_orchestrator.extract_changed_chunks(raw, count)

	# MLS-MPM migration (2026-08-12) — decode individual placed-voxel positions/materials
	# (not just chunk positions) so MPMSimulation can promote newly-placed sand/water into
	# particles. Reuses the SAME raw changelog download above — no extra GPU round-trip.
	var cs: int = _orchestrator._chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var slot_to_chunk := _orchestrator._slot_to_chunk
	var voxel_positions: Array[Vector3i] = []
	var voxel_materials: Array[int] = []
	for i in range(count):
		var base := 4 + i * 8
		var flat_idx := int(raw.decode_u32(base))
		var new_value := raw.decode_u32(base + 4)
		var slot: int = flat_idx / vpc
		var chunk_pos: Variant = slot_to_chunk.get(slot, null)
		if chunk_pos == null:
			continue
		var local := flat_idx % vpc
		var lx := local % cs
		var ly := (local / cs) % cs
		var lz := local / (cs * cs)
		voxel_positions.append(Vector3i(chunk_pos) * cs + Vector3i(lx, ly, lz))
		voxel_materials.append(int(new_value & 0xFF))
	if not voxel_positions.is_empty():
		_orchestrator.voxels_placed.emit(voxel_positions, voxel_materials)


func _upload_params(req: EditRequest, aabb_min: Vector3i, aabb_size: Vector3i) -> void:
	var buf := PackedByteArray(); buf.resize(EP_BYTES); buf.fill(0)
	var s := req.shape_params
	var gmin  := _orchestrator.get_grid_min()
	var gspan := _orchestrator.get_grid_span()
	var cs    := _orchestrator._chunk_manager.chunk_size

	buf.encode_s32(0,  req.center.x)
	buf.encode_s32(4,  req.center.y)
	buf.encode_s32(8,  req.center.z)
	buf.encode_s32(12, req.shape_type)
	buf.encode_float(16, s.x)
	buf.encode_float(20, s.y)
	buf.encode_float(24, s.z)
	buf.encode_float(28, s.w)
	buf.encode_s32(32, req.operation)
	buf.encode_s32(36, req.material_id)
	buf.encode_s32(40, aabb_min.x)
	buf.encode_s32(44, aabb_min.y)
	buf.encode_s32(48, aabb_min.z)
	buf.encode_s32(52, aabb_size.x)
	buf.encode_s32(56, aabb_size.y)
	buf.encode_s32(60, aabb_size.z)
	buf.encode_s32(64, gmin.x)
	buf.encode_s32(68, gmin.y)
	buf.encode_s32(72, gmin.z)
	buf.encode_s32(76, gspan.x)
	buf.encode_s32(80, gspan.y)
	buf.encode_s32(84, gspan.z)
	buf.encode_s32(88, cs)

	_compute.buffer_update(_params_buf, 0, buf)


func _load_shader() -> void:
	var f := FileAccess.open("res://shaders/edit_apply.comp", FileAccess.READ)
	if not f:
		push_error("EditDispatcher: cannot open edit_apply.comp"); return
	_pipeline = _compute.compile_shader(f.get_as_text(), 4)
	if _pipeline.is_empty():
		push_error("EditDispatcher: edit_apply.comp compile failed")


func _alloc_buffers() -> void:
	var params_data := PackedByteArray(); params_data.resize(EP_BYTES); params_data.fill(0)
	_params_buf = _compute.buffer_upload(params_data)

	var cl_size := 4 + MAX_EDIT_CHANGELOG * 8
	var cl_data := PackedByteArray(); cl_data.resize(cl_size); cl_data.fill(0)
	_changelog_buf = _compute.buffer_upload(cl_data)
