extends Node

const PROFILE_INTERVAL := 60

var _compute
var _pipeline: Dictionary = {}
var _output_res := Vector2i(320, 180)
var _chunk_size := 32
var _chunk_manager: ChunkManager
var _camera: Camera3D
var _voxel_scale: float

@export var debug_chunk_borders: bool = false
@export var debug_active_chunks: bool = false
# Density-trigger debug tint (2026-07-12, see FluidSimulation.debug_density_trigger_tint
# and docs/BUG_LOG.md "SOLVED IN BENCH (2026-07-11)") — independent toggle from the
# simulation-side one; both must be on to see anything (sim writes the buffer, renderer
# reads it). Split so the buffer can stay allocated/bound (see get_density_touch_buf())
# without forcing the tint to render, and vice versa.
@export var debug_density_touch_tint: bool = false
# Debug material hide (2026-07-16, world.gd KEY_X) — a BITMASK, one bit per material id
# (bit N set = material N is hidden), so multiple materials can be hidden at once (e.g.
# rock AND oil, to see straight through a test rig's walls to whatever's submerged inside).
# Matched materials are treated as air by the raymarcher, purely on the render side;
# voxels_a/the simulation are never touched, so cutting materials away to inspect what's
# stacked underneath a stuck-looking fluid interaction can't perturb the physics being
# inspected. 0 = nothing hidden.
@export var debug_hide_mask: int = 0

var _out_buf: RID
var _debug_buf: RID
# Always-valid placeholder for binding 11 (density_touch) before FluidSimulation has
# allocated its real buffer, or if it's ever unavailable — keeps the render uniform
# set's binding count fixed regardless of the other node's readiness (2026-07-12).
var _density_touch_fallback_buf: RID
var _cursor_buf: RID
var _active_chunks_buf: RID
var _collision_chunks_buf: RID
var _cursor_data: PackedByteArray
var _pixel_count: int

var _granular_sim      # lazily resolved
var _fluid_sim          # lazily resolved (2026-07-12, density-trigger debug tint)

# Pre-allocated per-frame uniform buffers — updated in place each frame instead of alloc/free
var _params_buf: RID
var _cam_buf: RID
var _vp_buf: RID

var _material_db: MaterialDB
var _orchestrator: WorldOrchestrator

# Cached uniform set — rebuilt only when a bound RID changes.
var _render_uniform_set: RID
var _cached_voxel_rid: RID
var _cached_chunk_table_rid: RID
# _density_touch_buf's own RID changes on every capacity grow (free+realloc, same as
# _flux_buf/_demand_accum_buf) — must be tracked alongside voxel_buf/chunk_table so the
# uniform set rebuilds whenever it does, not just on voxel-buffer growth.
var _cached_density_touch_rid: RID

var _quad: MeshInstance3D
var _color_tex: ImageTexture
var _depth_tex: ImageTexture

var _frame_count := 0
var _render_times: Array[float] = []
var _frame_times: Array[float] = []
var _last_frame_t0 := 0

var _debug_label: Label
var _perf_label: Label

func _ready() -> void:
	_compute = preload("res://Utils/compute_helper.gd").new()
	if not _compute.has_device():
		push_warning("RaymarchRenderer: no compute device")
		return

	_pipeline = _compute.compile_shader(_shader_source(), 4)
	if _pipeline.is_empty():
		push_warning("RaymarchRenderer: shader compile failed")
		return

	var viewport_size = get_viewport().size
	if viewport_size.x > 0 and viewport_size.y > 0:
		_resize_out(viewport_size)

	_chunk_manager = get_node("/root/world/ChunkManager")
	_material_db = get_node("/root/world/MaterialDB")
	_orchestrator = get_node("/root/world/WorldOrchestrator")
	_voxel_scale = _chunk_manager.voxel_scale
	_camera = get_viewport().get_camera_3d()

	_alloc_frame_buffers()
	_ensure_quad()
	_init_labels()
	Performance.add_custom_monitor("Voxel/render_ms", _perf_render_ms)
	Performance.add_custom_monitor("Voxel/chunks_loaded", _perf_chunks_loaded)


func _init_labels() -> void:
	var canvas = CanvasLayer.new()
	canvas.name = "DebugCanvas"
	add_child(canvas)

	_debug_label = Label.new()
	_debug_label.name = "VoxelInfo"
	_debug_label.position = Vector2(10, 10)
	_debug_label.add_theme_font_size_override("font_size", 14)
	_debug_label.add_theme_color_override("font_color", Color.WHITE)
	_debug_label.add_theme_color_override("font_shadow_color", Color.BLACK)
	_debug_label.add_theme_constant_override("shadow_offset_x", 1)
	_debug_label.add_theme_constant_override("shadow_offset_y", 1)
	canvas.add_child(_debug_label)

	_perf_label = Label.new()
	_perf_label.name = "PerfStats"
	_perf_label.position = Vector2(10, 120)
	_perf_label.add_theme_font_size_override("font_size", 13)
	_perf_label.add_theme_color_override("font_color", Color(0.8, 1.0, 0.8, 1.0))
	_perf_label.add_theme_color_override("font_shadow_color", Color.BLACK)
	_perf_label.add_theme_constant_override("shadow_offset_x", 1)
	_perf_label.add_theme_constant_override("shadow_offset_y", 1)
	canvas.add_child(_perf_label)


func _alloc_frame_buffers() -> void:
	var p := PackedByteArray(); p.resize(84); p.fill(0)
	_params_buf = _compute.buffer_upload(p)
	var c := PackedByteArray(); c.resize(48); c.fill(0)
	_cam_buf = _compute.buffer_upload(c)
	var v := PackedByteArray(); v.resize(64); v.fill(0)
	_vp_buf = _compute.buffer_upload(v)
	# 4 bytes count + 256 chunks × 3 ints × 4 bytes = 3076; round up to 3080
	var ac := PackedByteArray(); ac.resize(3080); ac.fill(0)
	_active_chunks_buf = _compute.buffer_upload(ac)
	var cc := PackedByteArray(); cc.resize(3080); cc.fill(0)
	_collision_chunks_buf = _compute.buffer_upload(cc)


func _ensure_quad() -> void:
	if _quad:
		return
	if not _camera:
		return

	_quad = MeshInstance3D.new()
	_quad.name = "FullscreenQuad"
	_quad.mesh = QuadMesh.new()

	var mat = ShaderMaterial.new()
	mat.shader = preload("res://raymarch_quad.gdshader")
	mat.render_priority = 127
	_quad.material_override = mat
	_quad.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_quad.gi_mode = GeometryInstance3D.GI_MODE_DISABLED

	_camera.far = 1000
	_update_quad_size()
	_camera.add_child(_quad)
	_quad.owner = _camera.owner

	_init_textures()


func _init_textures() -> void:
	if not _quad or _output_res.x <= 0 or _output_res.y <= 0:
		return
	var w = _output_res.x
	var h = _output_res.y

	var color_data := PackedByteArray(); color_data.resize(w * h * 4); color_data.fill(0)
	var depth_data := PackedByteArray(); depth_data.resize(w * h * 4)
	for i in range(0, w * h * 4, 4):
		depth_data.encode_float(i, 1.0)

	var img = Image.create_from_data(w, h, false, Image.FORMAT_RGBA8, color_data)
	_color_tex = ImageTexture.create_from_image(img)
	img = Image.create_from_data(w, h, false, Image.FORMAT_RF, depth_data)
	_depth_tex = ImageTexture.create_from_image(img)

	var mat = _quad.material_override as ShaderMaterial
	mat.set_shader_parameter("raymarch_color", _color_tex)
	mat.set_shader_parameter("raymarch_depth", _depth_tex)


func _update_quad_size() -> void:
	if not _quad or not _camera:
		return
	var distance = 1.0
	var aspect = float(_output_res.x) / float(_output_res.y)
	var half_h = tan(deg_to_rad(_camera.fov) * 0.5)
	var height = 2.0 * distance * half_h
	var quad_mesh = _quad.mesh as QuadMesh
	quad_mesh.size = Vector2(height * aspect, height)
	_quad.position = Vector3(0, 0, -distance)


func _resize_out(size: Vector2i) -> void:
	if size.x <= 0 or size.y <= 0:
		return
	_output_res = size
	_pixel_count = _output_res.x * _output_res.y
	if _render_uniform_set.is_valid():
		_compute.free_rid(_render_uniform_set)
		_render_uniform_set = RID()
	if _out_buf.is_valid():
		_compute.free_rid(_out_buf)
	var out := PackedByteArray(); out.resize(_pixel_count * 8); out.fill(0)
	_out_buf = _compute.buffer_upload(out)

	if _debug_buf.is_valid():
		_compute.free_rid(_debug_buf)
	var dbg := PackedByteArray(); dbg.resize(32); dbg.fill(0)
	_debug_buf = _compute.buffer_upload(dbg)

	if not _cursor_buf.is_valid():
		var cur := PackedByteArray(); cur.resize(32); cur.fill(0)
		_cursor_buf = _compute.buffer_upload(cur)

	if not _density_touch_fallback_buf.is_valid():
		var dt := PackedByteArray(); dt.resize(4); dt.fill(0)
		_density_touch_fallback_buf = _compute.buffer_upload(dt)

	_init_textures()


func _process(_delta: float) -> void:
	_last_frame_t0 = Time.get_ticks_usec()
	var vp = get_viewport()
	if not vp:
		return
	var viewport_size = vp.size
	if viewport_size != _output_res:
		_resize_out(viewport_size)
		_update_quad_size()

	if not _camera:
		_camera = vp.get_camera_3d()
		_ensure_quad()

	if not _camera or not _chunk_manager or not _quad or _chunk_manager.chunks.is_empty():
		return

	var t0 := Time.get_ticks_usec()
	_render()
	var dt := Time.get_ticks_usec() - t0

	_render_times.append(dt * 0.001)
	if _render_times.size() > PROFILE_INTERVAL:
		_render_times.pop_front()

	_frame_count += 1
	_frame_times.append((Time.get_ticks_usec() - _last_frame_t0) * 0.001)
	if _frame_times.size() > PROFILE_INTERVAL:
		_frame_times.pop_front()

	_update_perf_label()


func _update_perf_label() -> void:
	if _render_times.is_empty() or _frame_times.is_empty():
		return
	var r_avg := 0.0; var r_hi := 0.0
	for t in _render_times:
		r_avg += t
		if t > r_hi: r_hi = t
	r_avg /= _render_times.size()

	var f_avg := 0.0; var f_hi := 0.0
	for t in _frame_times:
		f_avg += t
		if t > f_hi: f_hi = t
	f_avg /= _frame_times.size()

	_perf_label.text = "Render: %.1fms (peak %.1fms)\nChunks: %d  Rebuild: %s" % [
		r_avg, r_hi,
		_orchestrator.get_chunks_loaded(),
		"YES" if _orchestrator.is_rebuild_pending() else "no"
	]


func _perf_render_ms() -> float:
	if _render_times.is_empty(): return 0.0
	var s := 0.0
	for t in _render_times: s += t
	return s / _render_times.size()

func _perf_chunks_loaded() -> float:
	return float(_orchestrator.get_chunks_loaded()) if _orchestrator else 0.0


func _render() -> void:
	_orchestrator.ensure_grid()

	var voxel_buf := _orchestrator.get_voxel_buf()
	var chunk_table := _orchestrator.get_chunk_table()
	if not voxel_buf.is_valid() or not chunk_table.is_valid():
		return

	if not _fluid_sim:
		_fluid_sim = get_node_or_null(
			"/root/world/WorldOrchestrator/SimulationRuntime/FluidSimulation")
	var density_touch_buf: RID = _fluid_sim.get_density_touch_buf() if _fluid_sim else RID()
	if not density_touch_buf.is_valid():
		# FluidSimulation hasn't allocated its buffer yet (e.g. this frame runs before
		# its _ready(), or no compute device) — fall back to the always-valid dummy so
		# the uniform set's binding count never depends on another node's readiness.
		density_touch_buf = _density_touch_fallback_buf

	if _orchestrator.get_chunks_loaded() == 0:
		return

	var chunk_size = _chunk_size
	var grid_min := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()
	var grid_max = grid_min + grid_span - Vector3i(1, 1, 1)
	var slab_min = Vector3(grid_min) * chunk_size * _voxel_scale
	var slab_max = Vector3(grid_max + Vector3i(1, 1, 1)) * chunk_size * _voxel_scale

	var mouse_pos = get_viewport().get_mouse_position()

	var params_data := PackedByteArray(); params_data.resize(84)
	params_data.encode_s32(0, chunk_size)
	params_data.encode_s32(4, _output_res.x)
	params_data.encode_s32(8, _output_res.y)
	params_data.encode_s32(12, grid_span.x)
	params_data.encode_s32(16, grid_min.x)
	params_data.encode_s32(20, grid_min.y)
	params_data.encode_s32(24, grid_min.z)
	params_data.encode_s32(28, _orchestrator.get_chunks_loaded())
	params_data.encode_s32(32, grid_span.y)
	params_data.encode_s32(36, grid_span.z)
	params_data.encode_float(40, _voxel_scale)
	params_data.encode_float(44, slab_min.x)
	params_data.encode_float(48, slab_min.y)
	params_data.encode_float(52, slab_min.z)
	params_data.encode_float(56, slab_max.x)
	params_data.encode_float(60, slab_max.y)
	params_data.encode_float(64, slab_max.z)
	params_data.encode_s32(68, int(mouse_pos.x))
	params_data.encode_s32(72, int(mouse_pos.y))
	var chunk_flags := (1 if debug_chunk_borders else 0) | (2 if debug_active_chunks else 0) \
		| (8 if debug_density_touch_tint else 0)
	params_data.encode_s32(76, chunk_flags)
	params_data.encode_s32(80, debug_hide_mask)
	_compute.buffer_update(_params_buf, 0, params_data)
	_upload_active_chunks()
	_upload_collision_chunks()
	_compute.buffer_update(_cam_buf, 0, _camera_data())
	_compute.buffer_update(_vp_buf, 0, _vp_data())

	# Rebuild the uniform set only when a buffer RID has changed.
	if not _render_uniform_set.is_valid() \
			or voxel_buf != _cached_voxel_rid \
			or chunk_table != _cached_chunk_table_rid \
			or density_touch_buf != _cached_density_touch_rid:
		# Only manually free the old set when the underlying buffers haven't changed
		# (e.g. window resize). When the voxel buffer was rebuilt by _init_grid,
		# the RD auto-invalidates the set when the old buffer is freed — trying to
		# free it again triggers "Attempted to free invalid ID". Same auto-invalidate
		# applies to density_touch_buf's own free+realloc on capacity growth
		# (_ensure_density_touch_capacity()).
		if _render_uniform_set.is_valid() \
				and voxel_buf == _cached_voxel_rid \
				and chunk_table == _cached_chunk_table_rid \
				and density_touch_buf == _cached_density_touch_rid:
			_compute.free_rid(_render_uniform_set)
		_render_uniform_set = RID()
		var u0 = RDUniform.new(); u0.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u0.binding = 0; u0.add_id(voxel_buf)
		var u1 = RDUniform.new(); u1.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u1.binding = 1; u1.add_id(_params_buf)
		var u2 = RDUniform.new(); u2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u2.binding = 2; u2.add_id(_out_buf)
		var u3 = RDUniform.new(); u3.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u3.binding = 3; u3.add_id(_cam_buf)
		var u4 = RDUniform.new(); u4.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u4.binding = 4; u4.add_id(chunk_table)
		var u5 = RDUniform.new(); u5.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u5.binding = 5; u5.add_id(_vp_buf)
		var u6 = RDUniform.new(); u6.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u6.binding = 6; u6.add_id(_material_db.buffer_rid)
		var u7 = RDUniform.new(); u7.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u7.binding = 7; u7.add_id(_debug_buf)
		var u8 = RDUniform.new(); u8.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u8.binding = 8; u8.add_id(_cursor_buf)
		var u9  = RDUniform.new(); u9.uniform_type  = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u9.binding  = 9;  u9.add_id(_active_chunks_buf)
		var u10 = RDUniform.new(); u10.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u10.binding = 10; u10.add_id(_collision_chunks_buf)
		var u11 = RDUniform.new(); u11.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER; u11.binding = 11; u11.add_id(density_touch_buf)
		_render_uniform_set = _compute.create_uniform_set([u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11], _pipeline.shader)
		_cached_voxel_rid          = voxel_buf
		_cached_chunk_table_rid    = chunk_table
		_cached_density_touch_rid  = density_touch_buf

	var workgroups_x = (_output_res.x + 7) / 8
	var workgroups_y = (_output_res.y + 7) / 8
	_compute.run_with_set(_pipeline, _render_uniform_set, workgroups_x, workgroups_y)
	_compute.submit()
	_compute.sync()

	var raw = _compute.download(_out_buf)
	var debug_raw = _compute.download(_debug_buf)
	if _cursor_buf.is_valid():
		_cursor_data = _compute.download_range(_cursor_buf, 0, 32)

	var w = _output_res.x
	var h = _output_res.y
	_color_tex.update(Image.create_from_data(w, h, false, Image.FORMAT_RGBA8, raw.slice(0, _pixel_count * 4)))
	_depth_tex.update(Image.create_from_data(w, h, false, Image.FORMAT_RF, raw.slice(_pixel_count * 4, _pixel_count * 8)))

	var mat = _quad.material_override as ShaderMaterial
	mat.set_shader_parameter("raymarch_color", _color_tex)
	mat.set_shader_parameter("raymarch_depth", _depth_tex)

	_update_debug_label(debug_raw)


func _update_debug_label(debug_raw: PackedByteArray) -> void:
	if debug_raw.size() < 32:
		_debug_label.text = ""
		return
	var packed_info = debug_raw.decode_s32(12)
	if packed_info == 0:
		_debug_label.text = ""
		return

	var mat_id = packed_info & 0xFF
	var fill = (packed_info >> 8) & 0xFF
	var temperature = (packed_info >> 16) & 0xFF
	var flags_byte = (packed_info >> 24) & 0xFF
	var mat_type = debug_raw.decode_s32(16)

	var def = _material_db.get_def(mat_id)
	var mat_name = def.name if def else "ID %d" % mat_id
	var type_names := ["SOLID", "FLUID", "EFFECT"]
	var type_str = type_names[mat_type] if mat_type < type_names.size() else str(mat_type)

	_debug_label.text = "Pos: (%d, %d, %d)\nMaterial: %s (id=%d)\nType: %s\nFill: %d  Temp: %d  Flags: 0x%02X" % [
		debug_raw.decode_s32(0), debug_raw.decode_s32(4), debug_raw.decode_s32(8),
		mat_name, mat_id, type_str, fill, temperature - 128, flags_byte
	]


func _upload_collision_chunks() -> void:
	var positions: Array[Vector3i] = []
	var count := mini(positions.size(), 256)
	var buf := PackedByteArray(); buf.resize(4 + count * 12); buf.fill(0)
	buf.encode_s32(0, count)
	for i in range(count):
		buf.encode_s32(4 + i * 12,     positions[i].x)
		buf.encode_s32(4 + i * 12 + 4, positions[i].y)
		buf.encode_s32(4 + i * 12 + 8, positions[i].z)
	_compute.buffer_update(_collision_chunks_buf, 0, buf)


func _upload_active_chunks() -> void:
	var positions: Array[Vector3i] = []
	if debug_active_chunks:
		if not _granular_sim:
			_granular_sim = get_node_or_null(
				"/root/world/WorldOrchestrator/SimulationRuntime/GranularSimulation")
		if _granular_sim:
			positions.append_array(_granular_sim.get_active_chunk_positions())
		# FluidSimulation's own active set (2026-07-16) — this overlay previously only
		# reflected GranularSimulation (sand), silently useless for answering "is the fluid
		# sim still dispatching to this chunk at all" while diagnosing a stuck submerged
		# water body. Merged in, deduplicated below, so one overlay covers both sims.
		if not _fluid_sim:
			_fluid_sim = get_node_or_null(
				"/root/world/WorldOrchestrator/SimulationRuntime/FluidSimulation")
		if _fluid_sim:
			positions.append_array(_fluid_sim.get_active_chunk_positions())
	var seen := {}
	var deduped: Array[Vector3i] = []
	for p in positions:
		if not seen.has(p):
			seen[p] = true
			deduped.append(p)
	positions = deduped
	var count := mini(positions.size(), 256)
	var buf := PackedByteArray(); buf.resize(4 + count * 12); buf.fill(0)
	buf.encode_s32(0, count)
	for i in range(count):
		buf.encode_s32(4 + i * 12,     positions[i].x)
		buf.encode_s32(4 + i * 12 + 4, positions[i].y)
		buf.encode_s32(4 + i * 12 + 8, positions[i].z)
	_compute.buffer_update(_active_chunks_buf, 0, buf)


func get_cursor_remove_pos() -> Variant:
	if _cursor_data.size() < 12:
		return null
	if _cursor_data.decode_s32(0) == 0x7FFFFFFF:
		return null
	return Vector3i(_cursor_data.decode_s32(0), _cursor_data.decode_s32(4), _cursor_data.decode_s32(8))


func get_cursor_add_pos() -> Variant:
	if _cursor_data.size() < 28:
		return null
	if _cursor_data.decode_s32(0) == 0x7FFFFFFF:
		return null
	return Vector3i(_cursor_data.decode_s32(16), _cursor_data.decode_s32(20), _cursor_data.decode_s32(24))


func _vp_data() -> PackedByteArray:
	var transform = _camera.global_transform
	var basis = transform.basis
	var origin = transform.origin
	var view_col_3 = Vector4(
		-(basis.x.x * origin.x + basis.x.y * origin.y + basis.x.z * origin.z),
		-(basis.y.x * origin.x + basis.y.y * origin.y + basis.y.z * origin.z),
		-(basis.z.x * origin.x + basis.z.y * origin.y + basis.z.z * origin.z),
		1.0)
	var p = _camera.get_camera_projection()
	var cols := [
		Vector4(basis.x.x, basis.y.x, basis.z.x, 0.0),
		Vector4(basis.x.y, basis.y.y, basis.z.y, 0.0),
		Vector4(basis.x.z, basis.y.z, basis.z.z, 0.0),
		view_col_3
	]
	var vp_cols := []
	for col in cols:
		vp_cols.append(p.x * col.x + p.y * col.y + p.z * col.z + p.w * col.w)
	var buf := PackedByteArray(); buf.resize(64)
	var off := 0
	for col in vp_cols:
		buf.encode_float(off, col.x); off += 4
		buf.encode_float(off, col.y); off += 4
		buf.encode_float(off, col.z); off += 4
		buf.encode_float(off, col.w); off += 4
	return buf


func _camera_data() -> PackedByteArray:
	var pos = _camera.global_position
	var aspect = float(_output_res.x) / float(_output_res.y)
	var fwd = -_camera.global_transform.basis.z
	var right = _camera.global_transform.basis.x
	var up = _camera.global_transform.basis.y
	var half_h = tan(deg_to_rad(_camera.fov) * 0.5)
	var half_w = half_h * aspect
	var lower_left = pos - half_w * right + half_h * up + fwd
	var horizontal = 2.0 * half_w * right
	var vertical = -2.0 * half_h * up
	var buf := PackedByteArray(); buf.resize(48)
	var off := 0
	for v in [pos, lower_left, horizontal, vertical]:
		buf.encode_float(off, v.x); off += 4
		buf.encode_float(off, v.y); off += 4
		buf.encode_float(off, v.z); off += 4
	return buf


func _shader_source() -> String:
	var f = FileAccess.open("res://shaders/raymarch.comp", FileAccess.READ)
	if f:
		return f.get_as_text()
	push_error("RaymarchRenderer: failed to load shader")
	return ""


func _exit_tree() -> void:
	if _render_uniform_set.is_valid():
		_compute.free_rid(_render_uniform_set)
	for buf in [_out_buf, _debug_buf, _cursor_buf, _active_chunks_buf, _collision_chunks_buf, _params_buf, _cam_buf, _vp_buf, _density_touch_fallback_buf]:
		if buf.is_valid():
			_compute.free_rid(buf)
