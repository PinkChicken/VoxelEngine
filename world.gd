extends Node3D

@onready var player = $CharacterBody3D
@onready var chunk_manager = $ChunkManager
@onready var world_orchestrator: WorldOrchestrator = $WorldOrchestrator
@onready var simulation_runtime: SimulationRuntime = $WorldOrchestrator/SimulationRuntime
@onready var fluid_simulation: FluidSimulation = $WorldOrchestrator/SimulationRuntime/FluidSimulation
@onready var raymarch_renderer = $RaymarchRenderer

# Watch-voxel indicator (2026-07-04) — a red wireframe box always rendered at
# FluidSimulation.watch_chunk/watch_local so it's visible while testing (e.g.
# disturbing water around the watched voxel) without having to infer the position
# from log coordinates. Same unshaded/depth_draw_never/cull_disabled trick as
# player_controller.gd's _cursor_indicator, so it stays visible even submerged under
# other voxels. Hidden whenever FluidSimulation.watch_enabled is false.
var _watch_indicator: MeshInstance3D

func _ready() -> void:
	Engine.max_fps = 60
	player.add_sphere.connect(func(pos, r, mat): chunk_manager.spawn_sand_sphere(pos, r, mat))
	player.remove_sphere.connect(func(pos: Vector3i, r: int): chunk_manager.remove_sphere(pos, r))

	var shader := Shader.new()
	shader.code = "shader_type spatial;
render_mode unshaded, cull_disabled, blend_mix, depth_draw_never, wireframe;
void fragment() {
	ALBEDO = vec3(1.0, 0.1, 0.1);
	ALPHA = 0.9;
}"
	var mat := ShaderMaterial.new()
	mat.shader = shader
	var box := BoxMesh.new()
	box.size = Vector3.ONE * chunk_manager.voxel_scale * 1.02
	box.material = mat
	_watch_indicator = MeshInstance3D.new()
	_watch_indicator.mesh = box
	_watch_indicator.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	_watch_indicator.visible = false
	add_child(_watch_indicator)


func _process(_delta: float) -> void:
	_watch_indicator.visible = fluid_simulation.watch_enabled
	if fluid_simulation.watch_enabled:
		var cs = chunk_manager.chunk_size
		var vs = chunk_manager.voxel_scale
		var world_voxel = fluid_simulation.watch_chunk * cs + fluid_simulation.watch_local
		_watch_indicator.global_position = (Vector3(world_voxel) + Vector3(0.5, 0.5, 0.5)) * vs

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_P:
			simulation_runtime.paused = !simulation_runtime.paused
			get_node("HotBar").set_paused(simulation_runtime.paused)
		elif event.keycode == KEY_RIGHT and simulation_runtime.paused:
			simulation_runtime.step_once()
			print("[SimStep] advanced 1 tick")
		elif event.keycode == KEY_F6:
			world_orchestrator.debug_dump_fluid_stats()
			fluid_simulation.debug_dump_flux_stats()
		elif event.keycode == KEY_F7:
			# Crosshair-driven fluid watch target (2026-07-02) — typing chunk/local
			# coordinates into the Inspector by hand was too unreliable for isolating
			# one misbehaving voxel found by eye. Reuses the same cursor raycast
			# player_controller.gd already uses for add/remove voxel, so "look at the
			# voxel, press V" targets exactly what's under the crosshair. Moved off F7
			# (2026-07-07) — F6-F9 are Godot's own editor debug-menu shortcuts (step,
			# breakpoint, etc.) and can steal window focus away from the running game
			# depending on editor settings, which is exactly the opposite of what you
			# want when you need to hit this key repeatedly without re-focusing.
			var target = raymarch_renderer.get_cursor_remove_pos()
			if target == null:
				print("[FluidWatch] no voxel under cursor")
			else:
				var cp = chunk_manager.voxel_to_chunk_position(target)
				var lp = chunk_manager.global_to_local(target)
				fluid_simulation.watch_chunk = cp
				fluid_simulation.watch_local = lp
				fluid_simulation.watch_enabled = true
				print("[FluidWatch] now watching chunk %s local %s (world %s)" % [cp, lp, target])
		elif event.keycode == KEY_U:
				build_utube_test_scene()
		elif event.keycode == KEY_DOWN:
			# Debug rewind tool (2026-07-07) -- save the canonical voxel buffer to a CPU-side
			# snapshot so a test scenario can be captured once and replayed: build the scene,
			# press Down to save, unpause/step forward and watch one voxel, press Up to
			# restore, step forward again and watch a different voxel, etc.
			world_orchestrator.debug_save_voxel_snapshot()
		elif event.keycode == KEY_UP:
			world_orchestrator.debug_load_voxel_snapshot()


# Scripted U-tube acceptance-test scene (2026-07-04, Checkpoint 4/5 tooling — see
# docs/PRESSURE_PROJECTION_PORT_PLAN.md). Hand-carving with the sphere/single-voxel edit
# tools produced inconsistent, hard-to-reproduce geometry (uneven connector width, unclear
# arm volumes) that polluted the bottom-gradient measurement -- this builds an exact,
# known-volume rig instead: a solid rock shell (guaranteed sealed regardless of
# surrounding terrain), two equal-height vertical shafts connected by a bottom passage,
# and a water block placed near the top of ONE shaft (not pre-filling the whole arm) so it
# falls and fills naturally once unpaused -- closer to an actual pour, and the poured
# volume is exactly known instead of hand-placed. Pauses the sim immediately after
# building so chunks/geometry can be inspected (F7) before manually unpausing (P).
# Sizes bumped 2026-07-07 (post pressure-projection revert) -- bigger rig, more water, so
# whatever's tried next against the flux system has more headroom to actually show a
# lateral gradient/imbalance instead of settling before it's visible.
const UTUBE_ARM_HALF_XZ := 1      # half-extent -> 3x3 voxel shaft cross-section
const UTUBE_ARM_HEIGHT  := 32     # voxels tall, both arms equal height
const UTUBE_GAP         := 12     # horizontal distance between arm centers
const UTUBE_WALL        := 2      # rock shell thickness around the carved shape
const UTUBE_WATER_BLOCK := 14     # voxels tall, placed near the top of the pour arm

func build_utube_test_scene() -> void:
	var origin = raymarch_renderer.get_cursor_remove_pos()
	if origin == null:
		print("[UTubeBuilder] no voxel under cursor -- aim at solid ground first")
		return

	var arm_a_x = origin.x
	var arm_b_x = origin.x + UTUBE_GAP
	var base_y  = origin.y          # floor of the connecting passage
	var top_y   = base_y + UTUBE_ARM_HEIGHT
	var z       = origin.z

	# Solid rock shell first, sized to fully contain both shafts + connector + wall
	# thickness -- guarantees a sealed rig regardless of what's actually around the
	# cursor target (open air, existing terrain, previously-carved space, etc.).
	var shell_center := Vector3i((arm_a_x + arm_b_x) / 2, (base_y + top_y) / 2, z)
	var shell_half := Vector3i(
		(arm_b_x - arm_a_x) / 2 + UTUBE_ARM_HALF_XZ + UTUBE_WALL,
		(top_y - base_y) / 2 + UTUBE_WALL,
		UTUBE_ARM_HALF_XZ + UTUBE_WALL)
	world_orchestrator.edit_box(shell_center, shell_half, EditDispatcher.EditOp.ADD, Chunk.Mat.GLASS)

	# Carve the two vertical shafts (air).
	var arm_a_center := Vector3i(arm_a_x, (base_y + top_y) / 2, z)
	var arm_b_center := Vector3i(arm_b_x, (base_y + top_y) / 2, z)
	var arm_half := Vector3i(UTUBE_ARM_HALF_XZ, (top_y - base_y) / 2, UTUBE_ARM_HALF_XZ)
	world_orchestrator.edit_box(arm_a_center, arm_half, EditDispatcher.EditOp.REMOVE)
	world_orchestrator.edit_box(arm_b_center, arm_half, EditDispatcher.EditOp.REMOVE)

	# Carve the bottom connecting passage (air), joining both shafts.
	var connector_center := Vector3i((arm_a_x + arm_b_x) / 2, base_y, z)
	var connector_half := Vector3i((arm_b_x - arm_a_x) / 2 + UTUBE_ARM_HALF_XZ, 0, UTUBE_ARM_HALF_XZ)
	world_orchestrator.edit_box(connector_center, connector_half, EditDispatcher.EditOp.REMOVE)

	# Pre-poured water block near the top of arm A (not the whole arm) -- an exact,
	# known volume that falls and fills the tube once unpaused, rather than a
	# hand-placed splat of uncertain size.
	var water_top_y = top_y - 2  # leave a couple voxels of headroom below the opening
	var water_center := Vector3i(arm_a_x, water_top_y - UTUBE_WATER_BLOCK / 2, z)
	var water_half := Vector3i(UTUBE_ARM_HALF_XZ, UTUBE_WATER_BLOCK / 2, UTUBE_ARM_HALF_XZ)
	world_orchestrator.edit_box(water_center, water_half, EditDispatcher.EditOp.ADD, Chunk.Mat.WATER)

	simulation_runtime.paused = true
	get_node("HotBar").set_paused(true)
	print("[UTubeBuilder] built U-tube at %s (arm height %d, gap %d) -- sim paused, press P to run" % [origin, UTUBE_ARM_HEIGHT, UTUBE_GAP])
