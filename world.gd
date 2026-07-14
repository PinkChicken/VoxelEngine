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
		elif event.keycode == KEY_K:
				build_density_test_tube_scene()
		elif event.keycode == KEY_DOWN:
			# Debug rewind tool (2026-07-07) -- save the canonical voxel buffer to a CPU-side
			# snapshot so a test scenario can be captured once and replayed: build the scene,
			# press Down to save, unpause/step forward and watch one voxel, press Up to
			# restore, step forward again and watch a different voxel, etc.
			world_orchestrator.debug_save_voxel_snapshot()
		elif event.keycode == KEY_UP:
			world_orchestrator.debug_load_voxel_snapshot()
		elif event.keycode == KEY_T:
			# Density-trigger debug tint (2026-07-12, docs/BUG_LOG.md "SOLVED IN BENCH
			# (2026-07-11)") — step 1 of the oil-water port: flips BOTH halves of the
			# tint (FluidSimulation writes the per-voxel buffer, RaymarchRenderer reads
			# it) together, since neither alone shows anything. Every OIL voxel directly
			# under a WATER voxel should flash bright magenta while this is on. Pour
			# water onto an oil pool (hotbar slot 5 = OIL, 3 = WATER) and watch: no tint
			# at all means the pass isn't reaching the GPU/dispatch; tint that only
			# appears where expected (and only there) confirms the plumbing works before
			# any real displacement mechanism gets written on top of it.
			var on := not fluid_simulation.debug_density_trigger_tint
			fluid_simulation.debug_density_trigger_tint = on
			raymarch_renderer.debug_density_touch_tint = on
			print("[DensityTint] %s" % ("ON — oil-under-water voxels will flash magenta" if on else "off"))


# Scripted U-tube acceptance-test scene (2026-07-04, Checkpoint 4/5 tooling — originally
# built for the pressure-projection effort, later reused as the acceptance rig for
# phi-flood, the design that actually shipped; see docs/PHI_FLOOD_PORT_PLAN.md and
# docs/BUG_LOG.md for current status). Hand-carving with the sphere/single-voxel edit
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


# Scripted sand-into-water acceptance-test scene (2026-07-10, KEY_K) -- a DIRECT,
# term-by-term port of the "cluster" scene from
# tools/fluid_prototypes/python/scenes.py::build_sand_pool(pool_half=8, pool_depth=4,
# cluster_half=2, cluster_h=3, drop_h=2), the exact rig the phi_demand mechanism
# (docs/PHI_FILL_SAND_PLAN.md) was validated against in the Python bench before any
# GLSL was touched. This is the in-engine acceptance rig for whenever that mechanism is
# ported -- same two-stage discipline as the U-tube/phi-flood rig above (bench-validate
# in Python first, then confirm the SAME scripted geometry in-editor). Until the port
# lands, this scene still shows the CURRENTLY SHIPPED try_displace_priority behaviour
# (the diagonal-fall gap fix from this session) -- useful right now as a "before"
# baseline, and ready to re-run unchanged as the "after" check once phi_demand ports.
#
# Deliberately mirrors scenes.py's Y arithmetic TERM FOR TERM (not re-derived from a
# different convention) -- including its one real quirk: origin.y itself stays solid
# (matches ROCK bowl's own base, one voxel below the first carved row), and the water
# pool's own centre/half math leaves it starting 1 voxel above that carved floor, not
# flush with it. That gap is not a bug here; it is exactly what was Python-validated,
# and "fixing" it would make this rig test different geometry than the bench proved.
# Two deliberate deviations, both purely cosmetic (never touch the water/sand geometry
# itself): GLASS instead of ROCK for the shell (same substitution build_utube_test_scene
# already makes, same reason -- watch it happen, don't just trust the numbers), and the
# shell is stretched taller than scenes.py's own bowl so the sand cluster's STARTING
# position is also inside the visible glass box instead of floating above an open roof.
const CLUSTER_POOL_HALF_XZ := 8   # scenes.py pool_half
const CLUSTER_POOL_DEPTH   := 4   # scenes.py pool_depth
const CLUSTER_SAND_HALF_XZ := 2   # scenes.py cluster_half -> 5x5 footprint
const CLUSTER_SAND_HEIGHT  := 3   # scenes.py cluster_h
const CLUSTER_DROP_HEIGHT  := 2   # scenes.py drop_h
const CLUSTER_WALL         := 2   # glass shell thickness (cosmetic only, see note above)
const CLUSTER_HEADROOM     := 3   # extra glass roof clearance above the sand's start (cosmetic only)

func build_sand_cluster_test_scene() -> void:
	var origin = raymarch_renderer.get_cursor_remove_pos()
	if origin == null:
		print("[SandClusterBuilder] no voxel under cursor -- aim at solid ground first")
		return

	# --- direct port of scenes.py::build_sand_pool's Y arithmetic (cy = origin.y) ---
	var depth = CLUSTER_POOL_DEPTH + 2                              # scenes.py: depth
	var bowl_center_y = origin.y + depth / 2                        # scenes.py: bowl_center[1]
	var inner_center_y = origin.y + depth / 2 + 1                   # scenes.py: inner_center[1]
	var inner_half_y = depth / 2                                    # scenes.py: inner_half[1]
	var floor_y = origin.y + 1                                      # scenes.py: floor_y
	var water_center_y = floor_y + CLUSTER_POOL_DEPTH / 2            # scenes.py: water_center[1]
	var water_half_y = (CLUSTER_POOL_DEPTH - 1) / 2                  # scenes.py: water_half[1]
	var surface_y = floor_y + CLUSTER_POOL_DEPTH - 1                 # scenes.py: surface_y
	var sand_bottom_y = surface_y + 1 + CLUSTER_DROP_HEIGHT          # scenes.py: sand_bottom
	var sand_center_y = sand_bottom_y + (CLUSTER_SAND_HEIGHT - 1) / 2  # scenes.py: sand_center[1]
	var sand_half_y = (CLUSTER_SAND_HEIGHT - 1) / 2                  # scenes.py: sand_half[1]
	var sand_top_y = sand_bottom_y + CLUSTER_SAND_HEIGHT - 1

	# --- shell: cosmetic only -- GLASS instead of ROCK, stretched to also contain the
	# sand cluster's starting position (scenes.py's own bowl only needs to contain the
	# water pool, since it doesn't care about visual containment of the fall). Centered
	# independently of inner_center_y/bowl_center_y above -- this is the one place this
	# function does NOT mirror scenes.py term-for-term, by design (see comment above). ---
	var shell_top_y = sand_top_y + CLUSTER_HEADROOM
	var shell_bottom_y = origin.y - CLUSTER_WALL
	var shell_center_y = (shell_bottom_y + shell_top_y) / 2
	var shell_half_y = (shell_top_y - shell_bottom_y) / 2
	var shell_center := Vector3i(origin.x, shell_center_y, origin.z)
	var shell_half := Vector3i(CLUSTER_POOL_HALF_XZ + CLUSTER_WALL, shell_half_y, CLUSTER_POOL_HALF_XZ + CLUSTER_WALL)
	world_orchestrator.edit_box(shell_center, shell_half, EditDispatcher.EditOp.ADD, Chunk.Mat.GLASS)

	# --- everything below this line is the direct scenes.py port, unchanged ---
	var inner_center := Vector3i(origin.x, inner_center_y, origin.z)
	var inner_half := Vector3i(CLUSTER_POOL_HALF_XZ, inner_half_y, CLUSTER_POOL_HALF_XZ)
	world_orchestrator.edit_box(inner_center, inner_half, EditDispatcher.EditOp.REMOVE)

	var water_center := Vector3i(origin.x, water_center_y, origin.z)
	var water_half := Vector3i(CLUSTER_POOL_HALF_XZ, water_half_y, CLUSTER_POOL_HALF_XZ)
	world_orchestrator.edit_box(water_center, water_half, EditDispatcher.EditOp.ADD, Chunk.Mat.WATER)

	var sand_center := Vector3i(origin.x, sand_center_y, origin.z)
	var sand_half := Vector3i(CLUSTER_SAND_HALF_XZ, sand_half_y, CLUSTER_SAND_HALF_XZ)
	world_orchestrator.edit_box(sand_center, sand_half, EditDispatcher.EditOp.ADD, Chunk.Mat.SAND)

	simulation_runtime.paused = true
	get_node("HotBar").set_paused(true)
	print("[SandClusterBuilder] built sand-cluster rig at %s (pool half %d depth %d, cluster %dx%d half %d, drop %d) -- sim paused, press P to run" \
		% [origin, CLUSTER_POOL_HALF_XZ, CLUSTER_POOL_DEPTH, CLUSTER_SAND_HALF_XZ * 2 + 1, CLUSTER_SAND_HALF_XZ * 2 + 1, CLUSTER_SAND_HALF_XZ, CLUSTER_DROP_HEIGHT])


# Scripted liquid-density-sorting acceptance-test scene (2026-07-12, KEY_K -- REPLACES the
# sand-cluster binding above; build_sand_cluster_test_scene() itself is untouched and
# still callable, just no longer bound to a key). A sealed glass tube with WATER on top
# and OIL on the bottom — the WRONG order (water is denser, so oil should rise to the
# top) — plus a few isolated OIL droplets sprinkled inside the water layer. Two separate
# cases under test in one rig, both relevant to the momentum/buoyancy port:
#   1. The bulk flat-interface case (the whole oil/water boundary) — a wide, flat layer
#      that must sort straight up without fragmenting.
#   2. The isolated-droplet case (each sprinkled oil voxel, surrounded by water on all
#      4 horizontal sides) — a lone riser, exercised separately from the bulk interface.
# Built with EXACT, known geometry specifically so a reported bug ("water floats on oil
# until I edit something, then it all drops at once") can be diagnosed against a known
# starting fill count instead of an ad-hoc hand-placed scene — compare the printed
# baseline below against F6's live dump (WorldOrchestrator.debug_dump_fluid_stats()).
const TESTTUBE_HALF_XZ      := 2   # 5x5 interior cross-section
const TESTTUBE_OIL_HEIGHT   := 10  # bottom layer — OIL, correct position
const TESTTUBE_WATER_HEIGHT := 10  # top layer — WATER, WRONG position (denser above lighter)
const TESTTUBE_HEADROOM     := 3   # empty air above the water, inside the sealed tube
const TESTTUBE_WALL         := 2   # glass shell thickness
# Local (x, y-offset-from-water-centre, z) positions for the sprinkled oil droplets —
# spread across different heights/offsets within the water layer, not just one spot.
const TESTTUBE_SPRINKLE: Array[Vector3i] = [
	Vector3i(0, -3, 0), Vector3i(1, 0, 1), Vector3i(-1, 2, -1),
]

func build_density_test_tube_scene() -> void:
	var origin = raymarch_renderer.get_cursor_remove_pos()
	if origin == null:
		print("[DensityTestTube] no voxel under cursor -- aim at solid ground first")
		return

	var oil_bottom_y = origin.y + 1
	var oil_top_y = oil_bottom_y + TESTTUBE_OIL_HEIGHT - 1
	var water_bottom_y = oil_top_y + 1
	var water_top_y = water_bottom_y + TESTTUBE_WATER_HEIGHT - 1
	var interior_top_y = water_top_y + TESTTUBE_HEADROOM

	# --- sealed glass shell (open nowhere -- a real test tube, not an open-top bowl) ---
	var shell_center_y = (origin.y + interior_top_y) / 2
	var shell_half_y = (interior_top_y - origin.y) / 2 + TESTTUBE_WALL
	var shell_center := Vector3i(origin.x, shell_center_y, origin.z)
	var shell_half := Vector3i(TESTTUBE_HALF_XZ + TESTTUBE_WALL, shell_half_y, TESTTUBE_HALF_XZ + TESTTUBE_WALL)
	world_orchestrator.edit_box(shell_center, shell_half, EditDispatcher.EditOp.ADD, Chunk.Mat.GLASS)

	var inner_half := Vector3i(TESTTUBE_HALF_XZ, shell_half_y - TESTTUBE_WALL, TESTTUBE_HALF_XZ)
	world_orchestrator.edit_box(shell_center, inner_half, EditDispatcher.EditOp.REMOVE)

	# --- oil layer (bottom, correct) ---
	var oil_center := Vector3i(origin.x, (oil_bottom_y + oil_top_y) / 2, origin.z)
	var oil_half := Vector3i(TESTTUBE_HALF_XZ, (oil_top_y - oil_bottom_y) / 2, TESTTUBE_HALF_XZ)
	world_orchestrator.edit_box(oil_center, oil_half, EditDispatcher.EditOp.ADD, Chunk.Mat.OIL)

	# --- water layer (top, WRONG -- this is the state that needs sorting) ---
	var water_center := Vector3i(origin.x, (water_bottom_y + water_top_y) / 2, origin.z)
	var water_half := Vector3i(TESTTUBE_HALF_XZ, (water_top_y - water_bottom_y) / 2, TESTTUBE_HALF_XZ)
	world_orchestrator.edit_box(water_center, water_half, EditDispatcher.EditOp.ADD, Chunk.Mat.WATER)

	# --- sprinkled isolated oil droplets inside the water layer ---
	for off in TESTTUBE_SPRINKLE:
		var p := Vector3i(origin.x + off.x, water_center.y + off.y, origin.z + off.z)
		world_orchestrator.edit_box(p, Vector3i.ZERO, EditDispatcher.EditOp.ADD, Chunk.Mat.OIL)

	# --- known baseline (computed from geometry, not read back from the GPU) ---
	var cross_section := (2 * TESTTUBE_HALF_XZ + 1) * (2 * TESTTUBE_HALF_XZ + 1)
	var oil_fill := cross_section * TESTTUBE_OIL_HEIGHT * 255
	var water_fill := cross_section * TESTTUBE_WATER_HEIGHT * 255
	# Sprinkled oil REPLACES water voxels it overwrites -- adjust both totals so the
	# printed baseline matches what's actually in the world, not just the two boxes'
	# nominal volumes.
	var sprinkle_count := TESTTUBE_SPRINKLE.size()
	water_fill -= sprinkle_count * 255
	oil_fill += sprinkle_count * 255

	simulation_runtime.paused = true
	get_node("HotBar").set_paused(true)
	print("[DensityTestTube] built at %s: oil height=%d (bottom, correct) water height=%d (top, WRONG) %d sprinkled oil droplet(s) in water" \
		% [origin, TESTTUBE_OIL_HEIGHT, TESTTUBE_WATER_HEIGHT, sprinkle_count])
	print("[DensityTestTube] baseline fill: oil=%d water=%d (compare against F6's live dump — WorldOrchestrator.debug_dump_fluid_stats() — after running)" \
		% [oil_fill, water_fill])
	print("[DensityTestTube] sim paused: P to run, T for the density debug tint, F6 for live fill stats")
