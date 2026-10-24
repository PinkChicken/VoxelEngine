class_name SimulationRuntime extends Node

# Fired once per real simulation tick, whether from run()'s accumulator (unpaused) or
# step_once() (paused, manual single-step) — both funnel through _tick().
signal tick_advanced

const SIM_HZ := 60.0       # simulation ticks per second
const MAX_CATCHUP := 1     # max ticks per render frame to prevent spiral-of-death

var _accumulator: float = 0.0
var paused: bool = false

# MPMSimulation is the only solver. It is a CHILD of this node, so it is already ready by the
# time this runs — Godot readies children before their own parent.
var mpm_simulation: MPMSimulation
# Substeps of MPM physics per real 60Hz tick. 24 mirrors the retired evaluation's own tuned
# default — a starting point, not a value re-derived for this game's specific performance
# profile; tune directly if pacing looks wrong in play.
@export var mpm_substeps_per_tick: int = 24
# Per-tick wall-clock timing for the MPM path — the measurement ARCHITECTURE.md §7.0 asked for.
# Enable with this Inspector flag or the G key in-game; it prints an average every 60 ticks
# including the live node/particle counts, so the cost curve can be attributed to grid size vs
# particle count rather than guessed at.
@export var log_mpm_perf: bool = false
var _perf_usec: int = 0
var _perf_ticks: int = 0
# Wall-clock cost of the most recent MPM tick, always recorded (two cheap clock reads per tick)
# so world.gd's Y dump can report it on demand without needing the 60-tick averaging window.
var last_mpm_tick_usec: int = 0


func reset_perf_window() -> void:
	_perf_usec = 0
	_perf_ticks = 0


func _ready() -> void:
	mpm_simulation = get_node_or_null("MPMSimulation")


func run(delta: float) -> void:
	if paused:
		return
	_accumulator += delta
	var tick_dt := 1.0 / SIM_HZ
	var ticks := 0
	while _accumulator >= tick_dt and ticks < MAX_CATCHUP:
		_accumulator -= tick_dt
		ticks += 1
		_tick()
		tick_advanced.emit()


# Debug single-step: advances exactly one simulation tick regardless of the accumulator,
# intended for use while paused (world.gd binds this to Right Arrow). Bypasses run()'s
# accumulator entirely so stepping doesn't get skipped or double-fired by leftover
# accumulated time from before the pause.
func step_once() -> void:
	_tick()
	tick_advanced.emit()


func _tick() -> void:
	if mpm_simulation and mpm_simulation.is_bound():
		var simulated := mpm_simulation.is_simulating()
		var t0 := Time.get_ticks_usec()
		mpm_simulation.tick_all(mpm_substeps_per_tick)
		last_mpm_tick_usec = Time.get_ticks_usec() - t0
		# Only ticks with a live region count toward the average — otherwise an idle engine
		# dilutes them to a meaningless 0.00 ms.
		if log_mpm_perf and simulated:
			_perf_usec += last_mpm_tick_usec
			_perf_ticks += 1
			if _perf_ticks >= 60:
				var info := mpm_simulation.region_grid_info()
				print("[MPM perf] %.2f ms/tick avg over %d ticks | nodes=%d particles=%d substeps=%d" % [
					float(_perf_usec) / 1000.0 / float(_perf_ticks), _perf_ticks,
					int(info.get("nodes", 0)), int(info.get("particles", 0)),
					mpm_substeps_per_tick])
				_perf_usec = 0
				_perf_ticks = 0
