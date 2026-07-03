class_name VoxelSimulation extends Node

func get_simulation_class() -> int:
	return MaterialDef.SimulationClass.NONE

# Runtime calls this when it discovers relevant voxels in changed chunks.
# Simulation merges them into its internal state however it sees fit.
func add_candidates(_positions: Array[Vector3i]) -> void:
	pass

# Called each frame while has_work() is true.
func step() -> void:
	pass

# false = runtime removes this simulation from the active set.
func has_work() -> bool:
	return false
