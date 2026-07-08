extends Node

@export var mouse_sensitivity: float = 0.005
@export var walk_speed: float = 5.0
@export var sprint_modifier: float = 1.5
@export var jump_velocity: float = 4.5
@export var fly_speed: float = 10.0
@export var sphere_radius: int = 15
@export var show_sphere_indicator: bool = true
@export var placement_distance: float = 20.0   # world units along camera ray; auto-tracks raycast when not placing
@export var min_placement_distance: float = 2.0
@export var max_placement_distance: float = 200.0
@export var distance_step: float = 2.0

var _prev_pressing_right: bool = false
var _editing: bool = false   # true while any mouse button is held

const HOTBAR_SIZE := 4
const HOTBAR_KEYS := [KEY_1, KEY_2, KEY_3, KEY_4, KEY_5]

var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")
var fly_mode: bool = true

const VS := 0.5  # voxel scale - must match ChunkManager.voxel_scale

@onready var body: CharacterBody3D = get_parent()
@onready var pivot: Node3D = body.get_node("Pivot")
@onready var camera: Camera3D = body.get_node("Pivot/Camera3D")
@onready var _renderer = get_node("/root/world/RaymarchRenderer")
@onready var _chunk_manager: ChunkManager = get_node("/root/world/ChunkManager")
@onready var _mover: VoxelMover = body.get_node("VoxelMover")

var _on_floor := false

var _indicator: MeshInstance3D
var _cursor_indicator: MeshInstance3D


func _ready() -> void:
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	_setup_indicator()


func _physics_process(delta: float) -> void:
	if fly_mode:
		fly_movement(delta)
	else:
		ground_movement(delta)


func _ray_hit_distance() -> float:
	var rem_pos = _renderer.get_cursor_remove_pos()
	if rem_pos:
		var hit_world := (Vector3(rem_pos) + Vector3(0.5, 0.5, 0.5)) * VS
		return (hit_world - camera.global_position).length()
	return max_placement_distance


func _ray_placement_voxel() -> Vector3i:
	var world_pos := camera.global_position + (-camera.global_transform.basis.z) * placement_distance
	return Vector3i(floor(world_pos.x / VS), floor(world_pos.y / VS), floor(world_pos.z / VS))


# Placement target for ADD operations (2026-07-07). While the sphere-visual indicator
# (V toggle) is showing, placement stays distance-tracked (_ray_placement_voxel) so you
# can push a sphere out into open air to place mid-air structures on purpose. When the
# indicator is off, placement snaps to RaymarchRenderer's GPU-computed cursor_add_pos --
# the exact air voxel adjacent to the surface face under the crosshair -- and returns
# null (no placement at all) when the ray doesn't hit anything, instead of silently
# falling back to a mid-air distance guess.
func _add_placement_voxel() -> Variant:
	if show_sphere_indicator:
		return _ray_placement_voxel()
	return _renderer.get_cursor_add_pos()


func _process(_delta: float) -> void:
	var pressing_left  := Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT)
	var pressing_right := Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT)
	var just_pressed_right := pressing_right and not _prev_pressing_right

	_editing = pressing_left or pressing_right
	_prev_pressing_right = pressing_right

	if not _editing:
		# Continuously pull back to surface if sphere is inside geometry, but
		# never push it further out. User input is the only thing that increases distance.
		var hit_d := _ray_hit_distance()
		if placement_distance > hit_d:
			placement_distance = clamp(hit_d, min_placement_distance, max_placement_distance)
	elif just_pressed_right:
		# On right-click press: push center half a radius into the surface so
		# removal sculpts the front face rather than blasting a deep hole.
		var hit_d := _ray_hit_distance()
		placement_distance = clamp(hit_d + (sphere_radius * VS) * 0.5, min_placement_distance, max_placement_distance)

	var voxel_center := _ray_placement_voxel()
	_indicator.global_position = (Vector3(voxel_center) + Vector3(0.5, 0.5, 0.5)) * VS
	_indicator.visible = show_sphere_indicator and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED

	# Single-voxel cursor: show at remove pos (E) if there's a hit, else placement pos (Q).
	var cursor_voxel = _renderer.get_cursor_remove_pos()
	if not cursor_voxel:
		cursor_voxel = _ray_placement_voxel()
	_cursor_indicator.global_position = (Vector3(cursor_voxel) + Vector3(0.5, 0.5, 0.5)) * VS
	_cursor_indicator.visible = Input.mouse_mode == Input.MOUSE_MODE_CAPTURED

	if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		if pressing_left:
			var add_center = _add_placement_voxel()
			if add_center != null:
				body.add_sphere.emit(add_center, sphere_radius, body.selected_material)
		elif pressing_right:
			body.remove_sphere.emit(voxel_center, sphere_radius)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

	if event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE

	if event.is_action_pressed("toggle_fly"):
		fly_mode = not fly_mode
		body.velocity = Vector3.ZERO
		get_viewport().set_input_as_handled()

	if event is InputEventKey and event.pressed and not event.echo \
			and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		if event.keycode == KEY_Q:
			var add_pos = _add_placement_voxel()
			if add_pos != null:
				body.add_sphere.emit(add_pos, 0, body.selected_material)
			get_viewport().set_input_as_handled()
		elif event.keycode == KEY_E:
			var rem = _renderer.get_cursor_remove_pos()
			if rem:
				body.remove_sphere.emit(rem, 0)
			get_viewport().set_input_as_handled()

	if event is InputEventKey and event.keycode == KEY_R and event.pressed and not event.echo:
		_chunk_manager.regenerate_chunks()
		get_viewport().set_input_as_handled()

	if event is InputEventKey and event.keycode == KEY_V and event.pressed and not event.echo:
		show_sphere_indicator = not show_sphere_indicator
		get_viewport().set_input_as_handled()

	_handle_material_selection(event)


func _handle_material_selection(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		for i in range(HOTBAR_KEYS.size()):
			if event.keycode == HOTBAR_KEYS[i]:
				_select_material(i + 1)
				return

	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_EQUAL or event.keycode == KEY_KP_ADD:
			sphere_radius = clampi(sphere_radius + 1, 1, 64)
			_update_indicator_size()
			get_viewport().set_input_as_handled()
			return
		if event.keycode == KEY_MINUS or event.keycode == KEY_KP_SUBTRACT:
			sphere_radius = clampi(sphere_radius - 1, 1, 64)
			_update_indicator_size()
			get_viewport().set_input_as_handled()
			return

	if event is InputEventMouseButton and event.pressed:
		# Shift+scroll overrides placement distance; plain scroll switches material.
		if Input.is_key_pressed(KEY_SHIFT):
			if event.button_index == MOUSE_BUTTON_WHEEL_UP:
				# Push out. While not editing, stop at the surface. While editing,
				# user input wins — no surface clamp so streaming sand doesn't fight you.
				var max_d := _ray_hit_distance() if not _editing else max_placement_distance
				placement_distance = clamp(placement_distance + distance_step, min_placement_distance, max_d)
				get_viewport().set_input_as_handled()
			elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
				placement_distance = clamp(placement_distance - distance_step, min_placement_distance, max_placement_distance)
				get_viewport().set_input_as_handled()
		else:
			if event.button_index == MOUSE_BUTTON_WHEEL_UP:
				_select_material(body.selected_material + 1)
			elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
				_select_material(body.selected_material - 1)


func _select_material(index: int) -> void:
	if index < 1:
		index = HOTBAR_SIZE
	elif index > HOTBAR_SIZE:
		index = 1
	body.selected_material = index
	body.material_changed.emit(index)


func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		pivot.rotate_y(-event.relative.x * mouse_sensitivity)
		camera.rotate_x(-event.relative.y * mouse_sensitivity)
		camera.rotation.x = clamp(camera.rotation.x, deg_to_rad(-89), deg_to_rad(89))


func fly_movement(delta: float) -> void:
	var input_dir  := Input.get_vector("left", "right", "down", "up")
	var forward    := -pivot.global_transform.basis.z
	forward.y = 0
	forward = forward.normalized()
	var right      := pivot.global_transform.basis.x
	var direction  := (forward * input_dir.y + right * input_dir.x).normalized()
	direction.y = Input.get_axis("ctrl", "space")
	var speed := fly_speed * (sprint_modifier if Input.is_key_pressed(KEY_SHIFT) else 1.0)
	body.velocity = direction * speed
	body.global_position += body.velocity * delta


func ground_movement(delta: float) -> void:
	var vel := body.velocity

	if not _on_floor:
		vel.y -= gravity * delta

	if Input.is_action_just_pressed("jump") and _on_floor:
		vel.y   = jump_velocity
		_on_floor = false

	var input_dir := Input.get_vector("left", "right", "down", "up")
	var direction := (pivot.transform.basis * Vector3(input_dir.x, 0, -input_dir.y)).normalized()
	var speed     := walk_speed * (sprint_modifier if Input.is_key_pressed(KEY_SHIFT) else 1.0)

	if direction != Vector3.ZERO:
		vel.x = direction.x * speed
		vel.z = direction.z * speed
	else:
		vel.x = move_toward(vel.x, 0, walk_speed)
		vel.z = move_toward(vel.z, 0, walk_speed)

	var result := _mover.move(body.global_position, vel, delta)
	body.global_position = result.pos
	body.velocity        = result.vel
	_on_floor            = result.on_floor


func _update_indicator_size() -> void:
	var mesh := _indicator.mesh as SphereMesh
	if mesh:
		mesh.radius = sphere_radius * VS
		mesh.height = sphere_radius * VS * 2.0


func _setup_indicator() -> void:
	var mesh = SphereMesh.new()
	mesh.radius = sphere_radius * VS
	mesh.height = sphere_radius * VS * 2.0
	mesh.radial_segments = 12
	mesh.rings = 6
	var shader = Shader.new()
	shader.code = "shader_type spatial;
render_mode unshaded, cull_disabled, blend_mix, depth_draw_never, wireframe;
void fragment() {
	ALBEDO = vec3(1.0, 1.0, 1.0);
	ALPHA = 0.35;
}"
	var mat = ShaderMaterial.new()
	mat.shader = shader
	mesh.material = mat
	_indicator = MeshInstance3D.new()
	_indicator.mesh = mesh
	_indicator.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_indicator.visible = false
	add_child(_indicator)

	var cursor_shader = Shader.new()
	cursor_shader.code = "shader_type spatial;
render_mode unshaded, cull_disabled, blend_mix, depth_draw_never, wireframe;
void fragment() {
	ALBEDO = vec3(1.0, 1.0, 1.0);
	ALPHA = 0.9;
}"
	var cursor_mat = ShaderMaterial.new()
	cursor_mat.shader = cursor_shader
	var cursor_mesh = BoxMesh.new()
	cursor_mesh.size = Vector3.ONE * VS * 1.02
	cursor_mesh.material = cursor_mat
	_cursor_indicator = MeshInstance3D.new()
	_cursor_indicator.mesh = cursor_mesh
	_cursor_indicator.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_cursor_indicator.visible = false
	add_child(_cursor_indicator)
