extends Node3D

@onready var player = $CharacterBody3D
@onready var chunk_manager = $ChunkManager
@onready var world_orchestrator: WorldOrchestrator = $WorldOrchestrator
@onready var simulation_runtime: SimulationRuntime = $WorldOrchestrator/SimulationRuntime
@onready var mpm_simulation: MPMSimulation = $WorldOrchestrator/SimulationRuntime/MPMSimulation
@onready var raymarch_renderer = $RaymarchRenderer
@onready var material_db = $MaterialDB

func _ready() -> void:
	Engine.max_fps = 60
	player.add_sphere.connect(func(pos, r, mat): chunk_manager.spawn_sand_sphere(pos, r, mat))
	player.remove_sphere.connect(func(pos: Vector3i, r: int): chunk_manager.remove_sphere(pos, r))

	# MLS-MPM migration (2026-08-12, brief §37) — setup() on EVERY normal boot, not just under a
	# debug flag. Revised same day: MPM used to bind ONE fixed window centered on the player's
	# spawn position — rejected by the user ("the simulation shouldn't be player-dependent; a
	# waterfall behind the player shouldn't stop just because we're not looking at it"). Now
	# uses the active-chunk model: setup()
	# itself scans all loaded chunks for existing MPM-eligible material and connects to
	# voxels_placed/chunk_loaded for ongoing, chunk-keyed activation — nothing here reads
	# player.global_position at all.
	await get_tree().create_timer(3.0).timeout  # chunks stream in over several frames, not
	# synchronously at boot — WorldOrchestrator.is_grid_valid() isn't meaningful before this.
	if not mpm_simulation.setup():
		print("[MPM] setup() failed — see error above")

	if OS.get_cmdline_user_args().has("mpm_stress"):
		_mpm_stress()


# STRESS HARNESS — one deliberately oversized water body, dropped above the player, so the cost
# curve can be read while the region grows under it. KEY_B in-game. How ms/tick tracks `nodes`
# (grid size) vs `particles` is what decides whether `promote_stride` (fewer, bigger particles)
# is worth spending surface fidelity on, or whether the grid itself is the cost and the answer
# is a smaller region.
func _mpm_stress() -> void:
	var vs: float = chunk_manager.voxel_scale
	var center := Vector3i(player.global_position / vs) + Vector3i(0, 20, 0)
	print("[MPMStress] spawning a radius-11 WATER body at voxel %s -> chunk %s (~5.6k voxels, ~5.6k particles)" % [
		center, chunk_manager.voxel_to_chunk_position(center)])
	world_orchestrator.edit_add_sphere(center, 11.0, 3)  # 3 = WATER
	if not simulation_runtime.log_mpm_perf:
		print("[MPMStress] perf logging is OFF — press G to start reporting 60-tick averages, Y for region state")
	else:
		print("[MPMStress] watch [MPM perf] lines; press Y for region state")


# KEY_G — toggle the 60-tick average. The window is reset on every toggle so the first printed
# average after enabling covers only ticks that were actually being measured.
func _mpm_toggle_perf() -> void:
	simulation_runtime.log_mpm_perf = not simulation_runtime.log_mpm_perf
	simulation_runtime.reset_perf_window()
	print("[MPMPerf] %s" % (
		"ON — a 60-tick average prints after each window" if simulation_runtime.log_mpm_perf else "off"))


# Material parameter sweep — MPM physics only, applied to WATER (id 3). Ordered so pressing O walks
# one axis of behaviour at a time: stiffness first, then viscosity, then density.
const MPM_PARAM_PRESETS := [
	{"name": "baseline",             "density": 1.0,  "stiffness": 8000.0,  "viscosity": 0.0},
	{"name": "very soft (E/16)",     "density": 1.0,  "stiffness": 500.0,   "viscosity": 0.0},
	{"name": "soft (E/4)",           "density": 1.0,  "stiffness": 2000.0,  "viscosity": 0.0},
	{"name": "stiff (Ex4)",          "density": 1.0,  "stiffness": 32000.0, "viscosity": 0.0},
	{"name": "very stiff (Ex16)",    "density": 1.0,  "stiffness": 128000.0,"viscosity": 0.0},
	{"name": "viscous",              "density": 1.0,  "stiffness": 8000.0,  "viscosity": 10.0},
	{"name": "heavy (density 4)",    "density": 4.0,  "stiffness": 8000.0,  "viscosity": 0.0},
	{"name": "light (density 0.25)", "density": 0.25, "stiffness": 8000.0,  "viscosity": 0.0},
]
var _mpm_preset_index: int = 0


func _mpm_cycle_material() -> void:
	_mpm_preset_index = (_mpm_preset_index + 1) % MPM_PARAM_PRESETS.size()
	var p: Dictionary = MPM_PARAM_PRESETS[_mpm_preset_index]
	# Drive the export setters rather than the material DB directly, so the Inspector sliders stay
	# in sync with the preset that O just selected.
	mpm_simulation.water_stiffness = p["stiffness"]
	mpm_simulation.water_viscosity = p["viscosity"]
	mpm_simulation.water_density = p["density"]
	print("[MPM params] %s — density %.2f, stiffness %.0f, viscosity %.1f (region rebuilt from voxels)" % [
		p["name"], p["density"], p["stiffness"], p["viscosity"]])


# KEY_Z — drop the MPM region without touching the world. Distinct from R (full terrain regen):
# this is "stop simulating, keep what's there", which is what you want before re-running a test
# from a clean simulation state. Material survives as voxels and re-activates on the next edit.
func _mpm_reset() -> void:
	mpm_simulation.reset()
	print("[MPM] region + active set dropped — voxels kept; press R for a full world regen")


# Decode the face-presence mask for the Y dump.
func _mpm_face_names(mask: int) -> String:
	if mask == 0:
		return "none"
	var names: Array[String] = []
	if mask & 1:  names.append("-X")
	if mask & 2:  names.append("+X")
	if mask & 4:  names.append("-Y")
	if mask & 8:  names.append("+Y")
	if mask & 16: names.append("-Z")
	if mask & 32: names.append("+Z")
	return " ".join(names)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_P:
			simulation_runtime.paused = !simulation_runtime.paused
			get_node("HotBar").set_paused(simulation_runtime.paused)
		elif event.keycode == KEY_RIGHT and simulation_runtime.paused:
			simulation_runtime.step_once()
			print("[SimStep] advanced 1 tick")
		elif event.keycode == KEY_T:
			# MPM time-scale probe. T cycles the simulated-seconds-per-tick ladder so you can see
			# whether the body is metastable because the substep is too coarse: watch whether it
			# still expands to fill its container as dt gets finer.
			print("[MPM] %s" % mpm_simulation.cycle_timescale(simulation_runtime.mpm_substeps_per_tick))
		elif event.keycode == KEY_O:
			# Material parameter sweep (2026-08-13). Cycles WATER's MPM physics through a ladder of
			# presets and rebuilds the region, so "is the boundary a solver/parameter problem or a
			# container problem" is one keypress per hypothesis instead of one build.
			_mpm_cycle_material()
		elif event.keycode == KEY_X:
			# Debug tool (2026-07-16) — TOGGLE the currently-selected hotbar material's bit
			# in raymarch_renderer.debug_hide_mask, hiding/unhiding it from the RENDERER ONLY
			# without touching voxels_a/the simulation at all. Widened from a single
			# material id to a bitmask (2026-07-16, same day) so multiple materials can be
			# hidden at once — e.g. hide ROCK to see through a test rig's walls, then also
			# hide OIL to see straight to a submerged water body, both at the same time. An
			# earlier version of this tool actually cleared matching voxels to real air,
			# which was rejected on the spot — that mutates the very physics being inspected
			# (e.g. removes the water whose rest state you're trying to observe). Pair with
			# pause (P) to freeze a moment and look through layers.
			var mat_to_hide = player.selected_material
			var hide_bit = 1 << mat_to_hide
			var now_hidden = (raymarch_renderer.debug_hide_mask & hide_bit) == 0
			if now_hidden:
				raymarch_renderer.debug_hide_mask |= hide_bit
			else:
				raymarch_renderer.debug_hide_mask &= ~hide_bit
			print("[HideMaterial] material %d %s (mask=0x%x)" % [mat_to_hide, ("now hidden" if now_hidden else "now visible"), raymarch_renderer.debug_hide_mask])
		elif event.keycode == KEY_C:
			# Debug tool (2026-07-16) — active-chunk overlay (RaymarchRenderer.debug_active_
			# chunks, bit 1 of debug_chunk_flags in raymarch.comp): tints every currently-
			# ACTIVE chunk. Added while diagnosing a stuck submerged water body, to answer a
			# question the flux-magnitude reading alone can't: is this voxel's chunk still
			# being dispatched to at all, or has it gone idle/asleep, in which case its flux
			# reading is just a stale leftover value, never recomputed and never acted on
			# (a scheduling question, not a rest-gate question) — see docs/BUG_LOG.md.
			var on_active = not raymarch_renderer.debug_active_chunks
			raymarch_renderer.debug_active_chunks = on_active
			print("[ActiveChunkTint] %s" % ("ON — active chunks tinted" if on_active else "off"))
		elif event.keycode == KEY_Y:
			_mpm_status_dump()
		elif event.keycode == KEY_B:
			# MPM stress test. CLI modes (`-- mpm_stress`) are unusable when launching from the
			# editor, which passes no user args, so the tests that matter are bound to keys.
			_mpm_stress()
		elif event.keycode == KEY_G:
			_mpm_toggle_perf()
		elif event.keycode == KEY_Z:
			_mpm_reset()


# Interactive debug tool (2026-08-12, brief §37, KEY_Y) — dumps the live MPM region: grid
# origin/dims/node count, how many chunks it covers, particle count against capacity, and
# idle_ticks (whether it is trending toward sleeping). No player-position logic anywhere in this
# file — activation is entirely chunk/material-driven (MPMSimulation.setup()/
# scan_loaded_chunks_for_existing_material()/_on_voxels_placed()).
func _mpm_status_dump() -> void:
	if not mpm_simulation.is_bound():
		if mpm_simulation.setup_error != "":
			print("[MPM] NOT RUNNING — setup() failed: %s" % mpm_simulation.setup_error)
		else:
			print("[MPM] not bound")
		return
	# Regions are shared across chunks now (2026-08-13), so there is one grid, not one per
	# chunk. Report the region, then the chunks it covers.
	var info := mpm_simulation.region_grid_info()
	var chunks := mpm_simulation.active_chunk_positions()
	if info.is_empty():
		print("[MPM] no live region — %d chunk(s) requested, none simulated (asleep or refused)" % chunks.size())
		return
	print("[MPM] region origin=%s dims=%s nodes=%d (active %d/%d blocks) | %d requested, %d occupied, %d covered | %d/%d particles, idle_ticks=%d" % [
		info["origin"], info["dims"], info["nodes"],
		int(info["num_blocks"]), int(info["total_blocks"]),
		int(info["requested"]),
		int(info["occupied"]), mpm_simulation.active_chunk_positions().size(),
		int(info["particles"]), int(info["capacity"]), int(info["idle_ticks"])])
	var requested := mpm_simulation.requested_chunk_positions()
	if not requested.is_empty():
		print("  requested (drives budget + growth): %s" % [requested])
	# Window extent and the material's own chunk extents, so the dump answers "where is the box
	# relative to my water" directly.
	print("  window chunks %s .. %s  (region_chunks %s)" % [info["window_min"], info["window_max"], info["region_chunks"]])
	print("  occupied chunks: %s" % [info["occupied_list"]])
	# On-demand timing — no averaging window needed, and no CLI flag.
	print("  last tick: %.2f ms  (G to log 60-tick averages, B to spawn a stress water body)" % (
		float(simulation_runtime.last_mpm_tick_usec) / 1000.0))
	print("  faces holding material: %s  (bit0=-X bit1=+X bit2=-Y bit3=+Y bit4=-Z bit5=+Z)" % (
		_mpm_face_names(int(info["faces"]))))
	print("  water physics: density %.2f  stiffness %.0f  viscosity %.1f   (O cycles presets; sliders in the remote Inspector)" % [
		mpm_simulation.water_density, mpm_simulation.water_stiffness, mpm_simulation.water_viscosity])
	print("  timescale: %.4f s/tick  (T cycles the ladder: 1/60 1/120 1/240 1/480 1/30)" % (
		mpm_simulation.sim_seconds_per_tick))
	print("  %s" % mpm_simulation.s0_summary())
	# Cost warning: grid buffers are ~60 bytes/node and are REALLOCATED on every window move, so
	# an oversized window shows up as multi-frame hitches, not just a higher average.
	var nodes := int(info["nodes"])
	if nodes > 600000:
		print("  WARNING: %d nodes (~%d MB of grid buffers) — every window move reallocates all of it, which is where the 90ms hitches come from. Prefer 2x2x2 = 262,144." % [
			nodes, int(float(nodes) * 60.0 / 1048576.0)])
