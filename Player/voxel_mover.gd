class_name VoxelMover extends Node

## Voxel-grid AABB player mover.
##
## Surface-extraction cache: on chunk entry or chunks_committed, downloads surface
## voxel bitmaps from the GPU (up to 8 chunks per frame). is_solid() is then a pure
## CPU bitmap lookup — zero GPU cost per movement check.

const HALF_W := 0.4    # lateral half-extent (world units)
const HEIGHT := 2.5    # total player height (world units)
const STEP_H := 0.5    # auto-step: walk over ledges up to 1 voxel without jumping
const SKIN   := 0.02   # epsilon for floor/ceiling detection

var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")

var _orchestrator: WorldOrchestrator
var _chunk_manager: ChunkManager

# chunk_pos → PackedByteArray (cs³ bytes, 1 = solid surface voxel, 0 = passable)
var _cache: Dictionary = {}
var _pending_refresh := true
var _last_center_chunk := Vector3i(999999, 999999, 999999)


func _ready() -> void:
	_orchestrator  = get_node("/root/world/WorldOrchestrator")
	_chunk_manager = get_node("/root/world/ChunkManager")
	_orchestrator.chunks_committed.connect(_on_chunks_committed)


func _on_chunks_committed(positions: Array[Vector3i]) -> void:
	for cp in positions:
		_cache.erase(cp)
	_pending_refresh = true


func _process(_delta: float) -> void:
	_refresh_if_needed()


func _refresh_if_needed() -> void:
	var body: Node3D  = get_parent()
	var vs            := _chunk_manager.voxel_scale
	var cs            := _chunk_manager.chunk_size
	var vp            := Vector3i(floori(body.global_position.x / vs),
								  floori(body.global_position.y / vs),
								  floori(body.global_position.z / vs))
	var center        := _chunk_manager.voxel_to_chunk_position(vp)

	if center != _last_center_chunk:
		_last_center_chunk = center
		_pending_refresh   = true

	if not _pending_refresh:
		return
	_pending_refresh = false

	# Collect up to 8 uncached chunks in the 3×3×3 neighborhood.
	var needed: Array[Vector3i] = []
	for dz: int in range(-1, 2):
		for dy: int in range(-1, 2):
			for dx: int in range(-1, 2):
				var cp := center + Vector3i(dx, dy, dz)
				if not _cache.has(cp) and _chunk_manager.chunks.has(cp):
					needed.append(cp)
					if needed.size() >= 8:
						break
			if needed.size() >= 8:
				break
		if needed.size() >= 8:
			break

	if not needed.is_empty():
		var batch  := _orchestrator.extract_surface_voxels_batch(needed)
		var stride := cs * cs * cs
		for cp in needed:
			var entries: PackedByteArray = batch.get(cp, PackedByteArray())
			var bitmap := PackedByteArray()
			bitmap.resize(stride)
			bitmap.fill(0)
			var count := entries.size() / 4
			for i: int in count:
				var packed := entries.decode_u32(i * 4)
				var flat   := int(packed & 0xFFFF)
				bitmap[flat] = 1
			_cache[cp] = bitmap

		# If we hit the batch cap, more chunks may still need caching next frame.
		if needed.size() >= 8:
			_pending_refresh = true

	# Evict chunks more than 2 steps away.
	var to_erase: Array = []
	for cp: Vector3i in _cache:
		var d := cp - center
		if abs(d.x) > 2 or abs(d.y) > 2 or abs(d.z) > 2:
			to_erase.append(cp)
	for cp in to_erase:
		_cache.erase(cp)


func is_solid(world_pos: Vector3) -> bool:
	var vs := _chunk_manager.voxel_scale
	var cs := _chunk_manager.chunk_size
	var vx := floori(world_pos.x / vs)
	var vy := floori(world_pos.y / vs)
	var vz := floori(world_pos.z / vs)
	var cp := _chunk_manager.voxel_to_chunk_position(Vector3i(vx, vy, vz))
	var bitmap: PackedByteArray = _cache.get(cp, PackedByteArray())
	if bitmap.is_empty():
		return false
	var lx := vx - cp.x * cs
	var ly := vy - cp.y * cs
	var lz := vz - cp.z * cs
	if lx < 0 or ly < 0 or lz < 0 or lx >= cs or ly >= cs or lz >= cs:
		return false
	return bitmap[lz * cs * cs + ly * cs + lx] != 0


## Move pos (feet) by vel*delta. Returns {pos, vel, on_floor}.
## Resolves collisions per-axis with 1-voxel step-up on horizontal blocks.
func move(pos: Vector3, vel: Vector3, delta: float) -> Dictionary:
	var on_floor := false
	var vs       := _chunk_manager.voxel_scale

	# Depenetration: if already inside solid (e.g. sand dropped on player),
	# push upward by voxel steps until the body column is clear.
	if _column_solid(pos.x, pos.y, pos.z) or _bottom_solid(pos.x, pos.y, pos.z):
		for i: int in 20:
			pos.y += vs
			if not _column_solid(pos.x, pos.y, pos.z) and not _top_solid(pos.x, pos.y, pos.z):
				vel.y = 0.0
				break

	# Y axis
	var ny := pos.y + vel.y * delta
	if vel.y <= 0.0 and _bottom_solid(pos.x, ny, pos.z):
		pos.y    = ceil(ny / vs) * vs                          # snap feet to top of voxel
		vel.y    = 0.0
		on_floor = true
	elif vel.y > 0.0 and _top_solid(pos.x, ny, pos.z):
		pos.y = floor((ny + HEIGHT) / vs) * vs - HEIGHT       # snap head to bottom of voxel
		vel.y = 0.0
	else:
		pos.y = ny

	if not on_floor:
		on_floor = _bottom_solid(pos.x, pos.y - SKIN, pos.z)

	# X axis with step-up
	var nx := pos.x + vel.x * delta
	if _column_solid(nx, pos.y, pos.z):
		var sy := pos.y + STEP_H
		if not _column_solid(nx, sy, pos.z) and not _top_solid(nx, sy, pos.z):
			pos.y = sy
			pos.x = nx
		else:
			vel.x = 0.0
	else:
		pos.x = nx

	# Z axis with step-up
	var nz := pos.z + vel.z * delta
	if _column_solid(pos.x, pos.y, nz):
		var sy := pos.y + STEP_H
		if not _column_solid(pos.x, sy, nz) and not _top_solid(pos.x, sy, nz):
			pos.y = sy
			pos.z = nz
		else:
			vel.z = 0.0
	else:
		pos.z = nz

	return {pos = pos, vel = vel, on_floor = on_floor}


func _bottom_solid(x: float, y: float, z: float) -> bool:
	return (is_solid(Vector3(x - HALF_W, y, z - HALF_W)) or
			is_solid(Vector3(x + HALF_W, y, z - HALF_W)) or
			is_solid(Vector3(x - HALF_W, y, z + HALF_W)) or
			is_solid(Vector3(x + HALF_W, y, z + HALF_W)))


func _top_solid(x: float, y: float, z: float) -> bool:
	var top := y + HEIGHT
	return (is_solid(Vector3(x - HALF_W, top, z - HALF_W)) or
			is_solid(Vector3(x + HALF_W, top, z - HALF_W)) or
			is_solid(Vector3(x - HALF_W, top, z + HALF_W)) or
			is_solid(Vector3(x + HALF_W, top, z + HALF_W)))


func _column_solid(x: float, y: float, z: float) -> bool:
	for h: float in [SKIN, HEIGHT * 0.5, HEIGHT - SKIN]:
		if (is_solid(Vector3(x - HALF_W, y + h, z - HALF_W)) or
				is_solid(Vector3(x + HALF_W, y + h, z - HALF_W)) or
				is_solid(Vector3(x - HALF_W, y + h, z + HALF_W)) or
				is_solid(Vector3(x + HALF_W, y + h, z + HALF_W))):
			return true
	return false
