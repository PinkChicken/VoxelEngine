extends Node

@export var mouse_sensitivity: float = 0.005
@export var sprint_modifier: float = 1.5
@export var fly_speed: float = 10.0
@export var sphere_radius: int = 15
@export var show_sphere_indicator: bool = true
@export var placement_distance: float = 20.0   # world units along camera ray; auto-tracks raycast when not placing
@export var min_placement_distance: float = 2.0
@export var max_placement_distance: float = 200.0
@export var distance_step: float = 2.0

var _prev_pressing_right: bool = false
var _editing: bool = false   # true while any mouse button is held

const HOTBAR_KEYS := [KEY_1, KEY_2, KEY_3, KEY_4, KEY_5]
# Hotbar slot count/id mapping is NOT duplicated here — read from HotBar.MATERIAL_IDS
# (HUD/hot_bar.gd) so there is exactly one place that defines "slot N = material id X".

const VS := 0.5  # voxel scale - must match ChunkManager.voxel_scale

@onready var body: Node3D = get_parent()
@onready var pivot: Node3D = body.get_node("Pivot")
@onready var camera: Camera3D = body.get_node("Pivot/Camera3D")
@onready var _renderer = get_node("/root/world/RaymarchRenderer")
@onready var _chunk_manager: ChunkManager = get_node("/root/world/ChunkManager")

var _indicator: MeshInstance3D
var _cursor_indicator: MeshInstance3D


func _ready() -> void:
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	_setup_indicator()


func _physics_process(delta: float) -> void:
	fly_movement(delta)


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
				_select_material(_current_hotbar_slot() + 1)
			elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
				_select_material(_current_hotbar_slot() - 1)


func _current_hotbar_slot() -> int:
	# Inverse of the index in HotBar.MATERIAL_IDS -> 1-based slot for the currently
	# selected material id, so scroll-wheel can step by SLOT (not by raw material id,
	# which is not contiguous once ids and slot order diverge, e.g. OIL = id 7).
	var i := HotBar.MATERIAL_IDS.find(body.selected_material)
	return (i + 1) if i >= 0 else 1


func _select_material(index: int) -> void:
	# index is a 1-based HOTBAR SLOT, not a material id — must be mapped through
	# HotBar.MATERIAL_IDS (see the comment near HOTBAR_KEYS above). Slot index only
	# happened to equal material id before OIL (id 7) was added at slot 5.
	var slot_count := HotBar.MATERIAL_IDS.size()
	if index < 1:
		index = slot_count
	elif index > slot_count:
		index = 1
	var material_id: int = HotBar.MATERIAL_IDS[index - 1]
	body.selected_material = material_id
	body.material_changed.emit(material_id)


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
	body.global_position += direction * speed * delta


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
