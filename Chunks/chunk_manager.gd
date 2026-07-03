@tool
class_name ChunkManager extends Node

## Total world extent in chunk units. World is centered at origin.
## Chunk table costs world_chunks.x * world_chunks.y * world_chunks.z * 4 bytes on the GPU.
## For 64×8×64: 128 KB — trivial. Only *loaded* chunks consume voxel buffer slots.
@export var world_chunks: Vector3i = Vector3i(64, 8, 64)
@export var chunk_size: int = 32
## XZ streaming radius in chunks. 3 = 7×7 = 49 columns loaded around the player.
@export var view_distance: int = 3
@export var terrain: Node
@export var voxel_scale: float = 0.5
## Voxels of WATER stacked directly on top of the terrain surface at generation time, in
## every column. 0 = disabled (old behavior, no water generated). Useful for settling
## tests — combine with TerrainGetter.primary_amplitude/detail_amplitude = 0 for a
## perfectly flat rock surface under a perfectly flat water layer.
@export var water_depth: int = 5

var chunk_class
var chunks: Dictionary[Vector3i, Chunk] = {}

## Fires only on full regenerate (R key). Normal streaming uses WorldOrchestrator.chunk_loaded.
signal slab_invalidated

var _material_db: MaterialDB
var _orchestrator: WorldOrchestrator

var _generated_columns: Dictionary = {}  # Vector2i(col_x, col_z) → true (queued or done)
var _column_queue: Array[Vector2i] = []
var _loading_thread: Thread
var _thread_busy: bool = false
var _last_player_col: Vector2i = Vector2i(0x7FFFFFFF, 0)  # sentinel — forces initial enqueue


func _init() -> void:
	chunk_class = load("res://Chunks/chunk.tscn")
	_loading_thread = Thread.new()


func _ready() -> void:
	if Engine.is_editor_hint():
		return
	_material_db = get_node("/root/world/MaterialDB")
	_orchestrator = get_node("/root/world/WorldOrchestrator")
	await terrain.populate()
	# Seed columns around world origin; updated to player position once player is resolved.
	_enqueue_columns_around(Vector2i.ZERO)


func _process(_delta: float) -> void:
	if Engine.is_editor_hint():
		return

	# Collect finished column — all Y chunks added in the same frame so they pop in together.
	if _thread_busy and not _loading_thread.is_alive():
		var result: Array = _loading_thread.wait_to_finish()
		_thread_busy = false
		for item in result:
			var chunk_pos  := item[0] as Vector3i
			var block_data := item[1] as PackedByteArray
			var new_chunk: Chunk = chunk_class.instantiate()
			new_chunk.position    = Vector3(chunk_pos) * chunk_size * voxel_scale
			new_chunk.terrain     = terrain
			new_chunk.voxel_scale = voxel_scale
			new_chunk.ensure_data(chunk_size)
			new_chunk.block_data  = block_data
			new_chunk.rebuild_counts_and_mask(_orchestrator._material_simulation)
			chunks[chunk_pos] = new_chunk
			add_child(new_chunk)
			_orchestrator._add_chunk_to_grid(chunk_pos)

	# Update streaming radius when player moves to a new column.
	var player := get_node_or_null("/root/world/Player") as CharacterBody3D
	if player:
		var vp  := Vector3i(player.global_position / voxel_scale)
		var pc  := voxel_to_chunk_position(vp)
		var col := Vector2i(pc.x, pc.z)
		if col != _last_player_col:
			_last_player_col = col
			_enqueue_columns_around(col)

	# Start next column on the background thread.
	if not _thread_busy and not _column_queue.is_empty():
		var next = _column_queue.pop_front()
		_loading_thread = Thread.new()
		_loading_thread.start(_generate_column.bind(next.x, next.y))
		_thread_busy = true


func _enqueue_columns_around(center: Vector2i) -> void:
	var half_x := world_chunks.x / 2
	var half_z := world_chunks.z / 2
	var candidates: Array[Vector2i] = []
	for dz in range(-view_distance, view_distance + 1):
		for dx in range(-view_distance, view_distance + 1):
			var col := Vector2i(center.x + dx, center.y + dz)
			if col.x < -half_x or col.x >= half_x or col.y < -half_z or col.y >= half_z:
				continue
			if _generated_columns.has(col):
				continue
			_generated_columns[col] = true
			candidates.append(col)
	# Closest columns first so the area immediately around the player fills in first.
	candidates.sort_custom(func(a, b): return a.distance_squared_to(center) < b.distance_squared_to(center))
	_column_queue.append_array(candidates)


# Generates block_data for all Y-level chunks in one XZ column. Background-thread safe:
# only reads terrain (stateless noise) — no Node instantiation or property access.
func _generate_column(col_x: int, col_z: int) -> Array:
	var results: Array = []
	var cs       := chunk_size
	var vs       := voxel_scale
	var y_start  := -world_chunks.y / 2
	var y_end    :=  y_start + world_chunks.y
	var world_sz := Vector3(world_chunks * cs)
	var vpc      := cs * cs * cs

	# Pre-sample terrain heights once for the whole column (avoids redundant noise calls).
	var heights := PackedInt32Array(); heights.resize(cs * cs)
	for z in range(cs):
		for x in range(cs):
			heights[z * cs + x] = int(terrain.get_stat_at(
				'elevation', col_x * cs + x, col_z * cs + z, world_sz))

	for y in range(y_start, y_end):
		var chunk_pos  := Vector3i(col_x, y, col_z)
		var offset_y   := y * cs
		var chunk_maxY := offset_y + cs - 1

		var block_data := PackedByteArray(); block_data.resize(vpc * Chunk.VOXEL_STRIDE); block_data.fill(0)

		for z in range(cs):
			for x in range(cs):
				var height     := heights[z * cs + x]
				var fill_start := offset_y
				var fill_end   = min(height - 1, chunk_maxY)
				if fill_start > fill_end:
					continue
				for wy in range(fill_start, fill_end + 1):
					var local_y := wy - offset_y
					var idx     := (z * cs * cs + local_y * cs + x) * Chunk.VOXEL_STRIDE
					block_data[idx]     = Chunk.Mat.ROCK
					block_data[idx + 1] = 255

				# Flat water layer directly on top of the rock surface at this column
				# (2026-07-03) — starts exactly at `height` (the first air voxel above
				# rock, since the rock fill above stops at height-1), so there's never
				# overlap with the rock fill regardless of water_depth.
				if water_depth > 0:
					var water_start = max(height, offset_y)
					var water_end    = min(height + water_depth - 1, chunk_maxY)
					if water_start <= water_end:
						for wy in range(water_start, water_end + 1):
							var local_y := wy - offset_y
							var idx     := (z * cs * cs + local_y * cs + x) * Chunk.VOXEL_STRIDE
							block_data[idx]     = Chunk.Mat.WATER
							block_data[idx + 1] = 255

		results.append([chunk_pos, block_data])
	return results


func regenerate_chunks() -> void:
	if _thread_busy:
		_loading_thread.wait_to_finish()
		_thread_busy = false
	_column_queue.clear()
	_generated_columns.clear()
	for chunk in chunks.values():
		chunk.queue_free()
	chunks.clear()
	slab_invalidated.emit()  # WorldOrchestrator re-inits empty grid
	await terrain.populate()
	var center := Vector2i.ZERO
	var player := get_node_or_null("/root/world/Player") as CharacterBody3D
	if player:
		var vp := Vector3i(player.global_position / voxel_scale)
		var pc := voxel_to_chunk_position(vp)
		center = Vector2i(pc.x, pc.z)
	_last_player_col = Vector2i(0x7FFFFFFF, 0)
	_enqueue_columns_around(center)


# --- Voxel coordinate helpers ---

func _floor_div(a: int, b: int) -> int:
	var q := int(a / b)
	if a < 0 and a % b != 0:
		q -= 1
	return q


func _pos_mod(a: int, b: int) -> int:
	var r := a % b
	if r < 0:
		r += b
	return r


func voxel_to_chunk_position(pos: Vector3i) -> Vector3i:
	return Vector3i(
		_floor_div(pos.x, chunk_size),
		_floor_div(pos.y, chunk_size),
		_floor_div(pos.z, chunk_size)
	)


func global_to_local(pos: Vector3i) -> Vector3i:
	return Vector3i(
		_pos_mod(pos.x, chunk_size),
		_pos_mod(pos.y, chunk_size),
		_pos_mod(pos.z, chunk_size)
	)


# --- Voxel edit API (delegates to WorldOrchestrator) ---

func add_voxel(pos: Vector3i, material: int = Chunk.Mat.SAND) -> void:
	_orchestrator.edit_add_voxel(pos, material)


func remove_voxel(pos: Vector3i) -> void:
	_orchestrator.edit_remove_voxel(pos)


func get_chunk(pos: Vector3i) -> Chunk:
	return chunks.get(voxel_to_chunk_position(pos))


func spawn_sand_sphere(center: Vector3i, radius: int, material: int = Chunk.Mat.SAND) -> void:
	_orchestrator.edit_add_sphere(center, float(radius), material)


func explode_at(center: Vector3i, radius: float) -> void:
	_orchestrator.edit_remove_sphere(center, radius / voxel_scale)


func remove_sphere(center: Vector3i, radius: int) -> void:
	_orchestrator.edit_remove_sphere(center, float(radius))


func _exit_tree() -> void:
	if _thread_busy and _loading_thread.is_started():
		_loading_thread.wait_to_finish()
