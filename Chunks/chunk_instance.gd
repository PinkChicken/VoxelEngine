@tool
class_name Chunk extends StaticBody3D

enum Mat { AIR = 0, ROCK = 1, SAND = 2, WATER = 3, GLASS = 4, FIRE = 5, SMOKE = 6, OIL = 7 }
# 1 byte per voxel: just the material id (AIR == 0). The GPU buffer holds these same bytes,
# packed four to a u32 word — little-endian storage means uploading this array as-is lands
# pre-packed, and the shaders index it via (flat >> 2) / (flat & 3).
const VOXEL_STRIDE := 1


@export var terrain: Node
@export var voxel_scale: float = 0.5

var block_data: PackedByteArray
var chunk_size: int = 0
var _simulation_mask: int = 0


func active_simulation_mask() -> int:
	return _simulation_mask

# Lazy set, tolerates false positives by design — MPM only uses the mask as a cheap
# "might contain simulated material" pre-filter.
func mark_simulation_active(sim_class: int) -> void:
	_simulation_mask |= MaterialDef.SIM_BITS[sim_class]

# One-time init after block_data is populated, and debug drift-check.
func rebuild_simulation_mask(material_simulation: PackedInt32Array) -> void:
	_simulation_mask = 0
	var i := 0
	while i < block_data.size():
		var mat_id := block_data[i]
		if mat_id != 0:
			var sim := material_simulation[mat_id]
			if sim != MaterialDef.SimulationClass.NONE:
				_simulation_mask |= MaterialDef.SIM_BITS[sim]
		i += VOXEL_STRIDE


func _voxel_index(local_x: int, local_y: int, local_z: int) -> int:
	return (local_z * chunk_size * chunk_size + local_y * chunk_size + local_x) * VOXEL_STRIDE


func get_material_id(local_idx: int) -> int:
	return block_data[local_idx]

func set_voxel(local_idx: int, mat_id: int) -> void:
	block_data[local_idx] = mat_id


func ensure_data(cs: int) -> void:
	if chunk_size == cs and block_data.size() == cs * cs * cs * VOXEL_STRIDE:
		return
	chunk_size = cs
	block_data.resize(cs * cs * cs * VOXEL_STRIDE)
	block_data.fill(0)
	_simulation_mask = 0


func generate_chunk(cs: int, world_size: Vector3) -> void:
	ensure_data(cs)
	block_data.fill(0)

	var voxel_scale_val = voxel_scale
	var offset_x = floori(position.x / voxel_scale_val)
	var offset_y = floori(position.y / voxel_scale_val)
	var offset_z = floori(position.z / voxel_scale_val)
	var min_h = int(terrain.get_minimum('elevation'))

	var chunk_min_y = offset_y
	var chunk_max_y = offset_y + chunk_size - 1

	for z in range(chunk_size):
		for x in range(chunk_size):
			var world_x = offset_x + x
			var world_z = offset_z + z
			var height = int(terrain.get_stat_at('elevation', world_x, world_z, world_size))
			var fill_start = chunk_min_y
			var fill_end = min(height - 1, chunk_max_y)
			if fill_start > fill_end:
				continue
			for wy in range(fill_start, fill_end + 1):
				set_voxel(_voxel_index(x, wy - offset_y, z), Mat.ROCK)

	if Engine.is_editor_hint():
		build_editor_mesh()


# Face directions: normal, and the 4 CCW vertices of the quad (in local voxel units).
const _FACE_DATA := [
	[Vector3(0,1,0),  [Vector3(0,1,1),Vector3(1,1,1),Vector3(1,1,0),Vector3(0,1,0)]],  # +Y top
	[Vector3(0,-1,0), [Vector3(0,0,0),Vector3(1,0,0),Vector3(1,0,1),Vector3(0,0,1)]],  # -Y bottom
	[Vector3(1,0,0),  [Vector3(1,0,0),Vector3(1,1,0),Vector3(1,1,1),Vector3(1,0,1)]],  # +X
	[Vector3(-1,0,0), [Vector3(0,0,1),Vector3(0,1,1),Vector3(0,1,0),Vector3(0,0,0)]],  # -X
	[Vector3(0,0,1),  [Vector3(1,0,1),Vector3(1,1,1),Vector3(0,1,1),Vector3(0,0,1)]],  # +Z
	[Vector3(0,0,-1), [Vector3(0,0,0),Vector3(0,1,0),Vector3(1,1,0),Vector3(1,0,0)]],  # -Z
]

const _MAT_COLORS := {
	1: Color(0.45, 0.43, 0.42),  # ROCK  — dark grey
	2: Color(0.87, 0.78, 0.50),  # SAND  — warm yellow
	3: Color(0.25, 0.50, 0.85),  # WATER — blue
	4: Color(0.75, 0.92, 0.97),  # GLASS — pale cyan
	5: Color(1.00, 0.45, 0.10),  # FIRE  — orange
	6: Color(0.60, 0.60, 0.60),  # SMOKE — grey
	7: Color(0.15, 0.12, 0.08),  # OIL   — near-black
}


func build_editor_mesh() -> void:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var cs := chunk_size
	var vs := voxel_scale

	# Fixed sun for baked lighting — no dependency on scene lights or WorldEnvironment.
	var sun_dir := Vector3(0.6, 1.0, 0.4).normalized()
	var ambient := 0.35

	for vz in cs:
		for vy in cs:
			for vx in cs:
				var idx := _voxel_index(vx, vy, vz)
				var mat_id := block_data[idx]
				if mat_id == 0:
					continue
				var base_color: Color = _MAT_COLORS.get(mat_id, Color(0.5, 0.5, 0.5))
				var origin := Vector3(vx, vy, vz) * vs

				for face in _FACE_DATA:
					var normal: Vector3 = face[0]
					var nx := vx + int(normal.x)
					var ny := vy + int(normal.y)
					var nz := vz + int(normal.z)
					# Out-of-bounds treated as solid — avoids Z-fighting at chunk seams.
					var neighbor := 1
					if nx >= 0 and nx < cs and ny >= 0 and ny < cs and nz >= 0 and nz < cs:
						neighbor = block_data[_voxel_index(nx, ny, nz)]
					if neighbor != 0:
						continue  # face is occluded

					# Bake diffuse into vertex color — consistent look in any environment.
					var diffuse := maxf(0.0, normal.dot(sun_dir))
					var light := ambient + (1.0 - ambient) * diffuse
					var color := Color(base_color.r * light, base_color.g * light, base_color.b * light)

					var verts: Array = face[1]
					st.set_color(color)
					st.set_normal(normal)
					for tri in [[0,1,2],[0,2,3]]:
						for vi in tri:
							st.add_vertex(origin + verts[vi] * vs)

	var mesh := st.commit()
	if mesh.get_surface_count() == 0:
		return  # all-air chunk — nothing to show

	var mat := StandardMaterial3D.new()
	mat.vertex_color_use_as_albedo = true
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.cull_mode = BaseMaterial3D.CULL_BACK

	var mi := get_node_or_null("EditorMesh") as MeshInstance3D
	if not mi:
		mi = MeshInstance3D.new()
		mi.name = "EditorMesh"
		add_child(mi)
	mi.mesh = mesh
	mi.material_override = mat


func add_voxel(local_pos: Vector3, material: int = Mat.SAND) -> void:
	var local_x = int(local_pos.x)
	var local_y = int(local_pos.y)
	var local_z = int(local_pos.z)
	set_voxel(_voxel_index(local_x, local_y, local_z), material)


func remove_voxel(local_pos: Vector3) -> void:
	remove_voxels([local_pos])


func remove_voxels(local_positions: Array) -> void:
	for pos in local_positions:
		var idx = _voxel_index(int(pos.x), int(pos.y), int(pos.z))
		set_voxel(idx, 0)
