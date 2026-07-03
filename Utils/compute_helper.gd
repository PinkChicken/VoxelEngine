extends RefCounted

static var _shared_rd: RenderingDevice = null

# Frame-level download counter. Reset by whoever owns the frame boundary (WorldOrchestrator).
static var downloads_this_frame: int = 0

var rd: RenderingDevice
var _pipeline_cache: Dictionary = {}



func _init():
	if _shared_rd == null:
		_shared_rd = RenderingServer.create_local_rendering_device()
	rd = _shared_rd


func has_device() -> bool:
	return rd != null


func compile_shader_verbose(glsl_source: String, stage: int = 4) -> Dictionary:
	var src = RDShaderSource.new()
	src.source_compute = glsl_source
	var spirv_obj = rd.shader_compile_spirv_from_source(src, true)
	if spirv_obj == null:
		print("[shader] SPIRV compile returned null")
		return {}
	var compile_err = spirv_obj.get_stage_compile_error(stage)
	if compile_err != "":
		print("[shader] GLSL compile error:\n", compile_err)
		return {}
	var shader = rd.shader_create_from_spirv(spirv_obj)
	var pipeline = rd.compute_pipeline_create(shader)
	return {"shader": shader, "pipeline": pipeline}


func compile_shader(glsl_source: String, stage: int = 4) -> Dictionary:
	var key = glsl_source.hash()
	if _pipeline_cache.has(key):
		return _pipeline_cache[key]

	var src = RDShaderSource.new()
	src.source_compute = glsl_source

	var spirv_obj = rd.shader_compile_spirv_from_source(src, stage)
	if spirv_obj == null:
		push_error("SPIR-V compilation returned null")
		return {}

	var compile_err = spirv_obj.get_stage_compile_error(stage)
	if compile_err != "":
		push_error("GLSL compile error:\n" + compile_err)
		return {}

	var shader = rd.shader_create_from_spirv(spirv_obj)
	var pipeline = rd.compute_pipeline_create(shader)
	var result = {"shader": shader, "pipeline": pipeline}
	_pipeline_cache[key] = result
	return result


func buffer_upload(data: PackedByteArray) -> RID:
	return rd.storage_buffer_create(data.size(), data)


func run(pipeline_data: Dictionary, uniforms: Array,
		groups_x: int, groups_y: int = 1, groups_z: int = 1) -> void:
	if pipeline_data.is_empty():
		push_error("Invalid pipeline data")
		return
	var set = rd.uniform_set_create(uniforms, pipeline_data.shader, 0)
	var list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(list, pipeline_data.pipeline)
	rd.compute_list_bind_uniform_set(list, set, 0)
	rd.compute_list_dispatch(list, groups_x, groups_y, groups_z)
	rd.compute_list_end()
	rd.free_rid(set)


# Creates a uniform set owned by the caller. Free with free_rid() when invalidated.
func create_uniform_set(uniforms: Array, shader: RID) -> RID:
	return rd.uniform_set_create(uniforms, shader, 0)


# Runs using a pre-built uniform set — no allocation or free.
func run_with_set(pipeline_data: Dictionary, uniform_set: RID,
		groups_x: int, groups_y: int = 1, groups_z: int = 1) -> void:
	if pipeline_data.is_empty():
		push_error("Invalid pipeline data")
		return
	var list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(list, pipeline_data.pipeline)
	rd.compute_list_bind_uniform_set(list, uniform_set, 0)
	rd.compute_list_dispatch(list, groups_x, groups_y, groups_z)
	rd.compute_list_end()


func submit() -> void:
	rd.submit()


func sync() -> void:
	rd.sync()


func buffer_update(rid: RID, offset: int, data: PackedByteArray) -> void:
	rd.buffer_update(rid, offset, data.size(), data)


func download(buffer_rid: RID) -> PackedByteArray:
	return rd.buffer_get_data(buffer_rid)


func download_range(buffer_rid: RID, offset: int, size: int) -> PackedByteArray:
	downloads_this_frame += 1
	return rd.buffer_get_data(buffer_rid, offset, size)


func free_rid(rid: RID) -> void:
	rd.free_rid(rid)


func buffer_grow(old_rid: RID, old_size: int, new_size: int) -> RID:
	var padded := rd.buffer_get_data(old_rid, 0, old_size)
	padded.resize(new_size)  # resize() zero-fills the new tail — no per-byte copy needed.
	var new_rid := rd.storage_buffer_create(new_size, padded)
	rd.free_rid(old_rid)
	return new_rid
