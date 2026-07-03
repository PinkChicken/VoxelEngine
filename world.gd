extends Node3D

@onready var player = $CharacterBody3D
@onready var chunk_manager = $ChunkManager
@onready var world_orchestrator: WorldOrchestrator = $WorldOrchestrator
@onready var simulation_runtime: SimulationRuntime = $WorldOrchestrator/SimulationRuntime
@onready var fluid_simulation: FluidSimulation = $WorldOrchestrator/SimulationRuntime/FluidSimulation
@onready var raymarch_renderer = $RaymarchRenderer

func _ready() -> void:
	Engine.max_fps = 60
	player.add_sphere.connect(func(pos, r, mat): chunk_manager.spawn_sand_sphere(pos, r, mat))
	player.remove_sphere.connect(func(pos: Vector3i, r: int): chunk_manager.remove_sphere(pos, r))

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
			pass
		elif event.keycode == KEY_F9:
			# Crosshair-driven fluid watch target (2026-07-02) — typing chunk/local
			# coordinates into the Inspector by hand was too unreliable for isolating
			# one misbehaving voxel found by eye. Reuses the same cursor raycast
			# player_controller.gd already uses for add/remove voxel, so "look at the
			# voxel, press F8" targets exactly what's under the crosshair.
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
