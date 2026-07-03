@tool
extends Node

## Y voxel level the surface mostly sits at.
@export var base_height: float = 40.0

## Primary noise — controls the large rolling shape of the surface.
## Set frequency low (e.g. 0.006) for gentle, wide undulations.
@export var primary_noise: FastNoiseLite

## Peak height deviation from base_height in voxels (primary layer).
@export var primary_amplitude: float = 3.0

## Detail noise — small rocky surface texture on top of the primary shape.
@export var detail_noise: FastNoiseLite

## Peak height deviation in voxels (detail layer). Keep at 1 or below.
@export var detail_amplitude: float = 1.0


func populate() -> void:
	if not primary_noise:
		primary_noise = FastNoiseLite.new()
		primary_noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
		primary_noise.frequency = 1.0 / 160.0

	if not detail_noise:
		detail_noise = FastNoiseLite.new()
		detail_noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
		detail_noise.frequency = 1.0 / 32.0


# Returns the surface height (in voxels) at world column (x, z).
# Noise returns -1..1, scaled by amplitude.
func get_stat_at(_stat: String, x: int, z: int, _world_size: Vector3) -> float:
	var fx := float(x)
	var fz := float(z)
	var h := base_height
	if primary_noise:
		h += primary_noise.get_noise_2d(fx, fz) * primary_amplitude
	if detail_noise:
		h += detail_noise.get_noise_2d(fx, fz) * detail_amplitude
	return h


func get_minimum(_endpoint: String) -> float:
	return 0.0
