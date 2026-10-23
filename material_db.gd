class_name MaterialDB extends Node

@export var materials: Array[MaterialDef] = []
@export var max_materials: int = 256

var buffer_rid: RID
var mpm_buffer_rid: RID
var _compute

var _id_to_def: Dictionary = {}


func _ready() -> void:
	_compute = preload("res://Utils/compute_helper.gd").new()
	if materials.is_empty():
		materials = _default_materials()
	_index_materials()
	build_buffer()
	build_mpm_buffer()


func _index_materials() -> void:
	_id_to_def.clear()
	for def in materials:
		_id_to_def[def.id] = def


func get_def(material_id: int) -> MaterialDef:
	return _id_to_def.get(material_id, null)


func is_collidable(material_id: int) -> bool:
	if material_id == 0:
		return false
	var def = get_def(material_id)
	if not def:
		return false
	# NONE materials are non-simulating solids (rock, wood) — collidable.
	# GRANULAR materials (sand) are also collidable. Everything else is not.
	return def.simulation == MaterialDef.SimulationClass.NONE \
		or def.simulation == MaterialDef.SimulationClass.GRANULAR


func register_material(def: MaterialDef) -> void:
	_id_to_def[def.id] = def
	var found := false
	for i in range(materials.size()):
		if materials[i].id == def.id:
			materials[i] = def
			found = true
			break
	if not found:
		materials.append(def)
	build_buffer()
	build_mpm_buffer()


func _default_materials() -> Array[MaterialDef]:
	var air := MaterialDef.new()
	air.id = 0; air.name = "AIR"
	air.phase = MaterialDef.Phase.GAS
	air.simulation = MaterialDef.SimulationClass.NONE
	air.gravity = MaterialDef.GravityMode.NONE

	var rock := MaterialDef.new()
	rock.id = 1; rock.name = "ROCK"
	rock.phase = MaterialDef.Phase.SOLID
	rock.simulation = MaterialDef.SimulationClass.NONE
	rock.gravity = MaterialDef.GravityMode.NONE
	rock.color = Color(0.604, 0.604, 0.604, 1)
	rock.mpm_model = MaterialDef.MPMModel.STATIC
	rock.mpm_density = 2.6  # granite-ish bulk density, water=1.0 baseline

	var sand := MaterialDef.new()
	sand.id = 2; sand.name = "SAND"
	sand.phase = MaterialDef.Phase.SOLID
	sand.simulation = MaterialDef.SimulationClass.GRANULAR
	sand.gravity = MaterialDef.GravityMode.DOWN
	sand.color = Color(0.867, 0.8, 0.533, 1)
	sand.mpm_model = MaterialDef.MPMModel.GRANULAR
	sand.mpm_density = 1.6  # dry bulk density (brief §35.5)
	sand.mpm_stiffness = 30000.0  # Young's modulus E, mirrors the retired mpm2d_sand.py/mpm_simulation.gd default (mu/lambda derived from this + a fixed Poisson ratio in build_mpm_buffer())
	sand.mpm_friction_deg = 40.0  # mirrors the retired mpm2d_sand.py / mpm_simulation.gd default

	var water := MaterialDef.new()
	water.id = 3; water.name = "WATER"
	water.phase = MaterialDef.Phase.LIQUID
	water.simulation = MaterialDef.SimulationClass.LIQUID
	water.gravity = MaterialDef.GravityMode.DOWN
	water.color = Color(0.2, 0.4, 0.9, 0.7)
	water.flow = 1.0  # thin — the viscosity baseline every other liquid is scaled against
	water.density = 1.0  # baseline every other liquid's density is set relative to
	water.mpm_model = MaterialDef.MPMModel.FLUID
	water.mpm_density = 1.0  # baseline every other MPM material's mpm_density is set relative to
	water.mpm_stiffness = 8000.0  # swept and re-defaulted in the retired prototype (brief §34.3) — tight J, stable to 300K
	water.mpm_viscosity = 0.0  # pressure-only baseline; oil/lava are set relative to this

	var glass := MaterialDef.new()
	glass.id = 4; glass.name = "GLASS"
	glass.phase = MaterialDef.Phase.SOLID
	glass.simulation = MaterialDef.SimulationClass.NONE
	glass.gravity = MaterialDef.GravityMode.NONE
	glass.color = Color(0.75, 0.92, 0.97, 1)
	glass.transparent = true
	glass.mpm_model = MaterialDef.MPMModel.STATIC
	glass.mpm_density = 2.5  # soda-lime glass-ish

	var fire := MaterialDef.new()
	fire.id = 5; fire.name = "FIRE"
	fire.phase = MaterialDef.Phase.EFFECT
	fire.simulation = MaterialDef.SimulationClass.EFFECT
	fire.gravity = MaterialDef.GravityMode.UP
	fire.color = Color(1.0, 0.6, 0.1, 0.9)
	# mpm_model left NOT_SIMULATED: fire needs mass-decay (fuel depletion) + a temperature
	# field for ignition (brief §37) that don't exist yet — a real material, deliberately
	# not defaulted here until that lands, not an oversight.

	var smoke := MaterialDef.new()
	smoke.id = 6; smoke.name = "SMOKE"
	smoke.phase = MaterialDef.Phase.GAS
	smoke.simulation = MaterialDef.SimulationClass.GAS
	smoke.gravity = MaterialDef.GravityMode.UP
	smoke.color = Color(0.3, 0.3, 0.3, 0.2)
	# mpm_model left NOT_SIMULATED: gas profile (bounded/transient vs. ambient vs. two-tier
	# hand-off) is still an open decision — brief §36.5/M5 — not defaulted here until settled.

	var oil := MaterialDef.new()
	oil.id = 7; oil.name = "OIL"
	oil.phase = MaterialDef.Phase.LIQUID
	oil.simulation = MaterialDef.SimulationClass.LIQUID
	oil.gravity = MaterialDef.GravityMode.DOWN
	oil.color = Color(0.2, 0.15, 0.05, 0.85)
	oil.flow = 0.69  # thicker than water — runs through the same viscosity formula (fluid_flux.comp)
	oil.density = 0.85  # lighter than water (1.0) — floats on top when density-stacking is on
	oil.mpm_model = MaterialDef.MPMModel.FLUID
	oil.mpm_density = 0.85  # lighter than water — floats via emergent MPM sorting, no bespoke code
	oil.mpm_stiffness = 6000.0  # slightly softer than water; untuned, a starting point
	oil.mpm_viscosity = 3.0  # thicker than water (0.0) — this field is what makes oil actually distinct from water in MPM (brief §35.1a)

	return [air, rock, sand, water, glass, fire, smoke, oil]


func build_buffer() -> void:
	if buffer_rid.is_valid():
		_compute.free_rid(buffer_rid)

	# GPU layout per material (36 bytes):
	#   offset  0: packed RGBA8 color (u32)
	#   offset  4: simulation class (u32)
	#   offset  8: phase (u32)
	#   offset 12: gravity mode (u32)
	#   offset 16-31: reserved
	#   offset 32: transparent flag (u32, 0/1) — raymarch.comp sees through non-edge
	#              pixels of these voxels instead of terminating the ray (2026-07-04,
	#              glass test-tube walls for visually verifying fluid U-tube behavior)
	var data := PackedByteArray()
	data.resize(max_materials * 36)
	data.fill(0)

	for def in materials:
		var base = def.id * 36
		if base < 0 or base + 36 > data.size():
			continue
		var col = def.color
		var packed = (int(col.a8) << 24) | (int(col.b8) << 16) | (int(col.g8) << 8) | int(col.r8)
		data.encode_u32(base + 0, packed)
		data.encode_u32(base + 4, def.simulation)
		data.encode_u32(base + 8, def.phase)
		data.encode_u32(base + 12, def.gravity)
		data.encode_u32(base + 32, 1 if def.transparent else 0)

	buffer_rid = _compute.buffer_upload(data)


func build_mpm_buffer() -> void:
	if mpm_buffer_rid.is_valid():
		_compute.free_rid(mpm_buffer_rid)

	# GPU layout per material (32 bytes) — read by shaders/mpm/p2g.comp, indexed directly by
	# material id (mirrors build_buffer()'s pattern, a separate mirror because these are
	# MPM-only fields with no equivalent in the flux/render table above).
	#   offset  0: mpm_model (u32) — MaterialDef.MPMModel
	#   offset  4: mpm_density (f32)
	#   offset  8: mpm_stiffness (f32) — raw E, used directly by FLUID's pressure formula
	#   offset 12: mpm_viscosity (f32) — FLUID only
	#   offset 16: alpha_sand (f32) — Drucker-Prager constant, precomputed from mpm_friction_deg
	#   offset 20: mu_sand (f32)     — Lamé mu, precomputed from mpm_stiffness assuming a fixed
	#   offset 24: lambda_sand (f32) —  Poisson ratio (0.3, matching the retired prototype's own
	#              sand default) — GRANULAR/ELASTIC only. All three sand constants are derived
	#              here, not per-substep in the shader, so the conversion lives in one place.
	#   offset 28: reserved
	const STRIDE := 32
	const SAND_POISSON := 0.3
	var data := PackedByteArray()
	data.resize(max_materials * STRIDE)
	data.fill(0)

	for def in materials:
		var base = def.id * STRIDE
		if base < 0 or base + STRIDE > data.size():
			continue
		data.encode_u32(base + 0, def.mpm_model)
		data.encode_float(base + 4, def.mpm_density)
		data.encode_float(base + 8, def.mpm_stiffness)
		data.encode_float(base + 12, def.mpm_viscosity)
		var s := sin(deg_to_rad(def.mpm_friction_deg))
		var alpha_sand := sqrt(2.0 / 3.0) * (2.0 * s) / (3.0 - s) if def.mpm_friction_deg > 0.0 else 0.0
		data.encode_float(base + 16, alpha_sand)
		var nu := SAND_POISSON
		var mu_sand := def.mpm_stiffness / (2.0 * (1.0 + nu))
		var lambda_sand := def.mpm_stiffness * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
		data.encode_float(base + 20, mu_sand)
		data.encode_float(base + 24, lambda_sand)

	mpm_buffer_rid = _compute.buffer_upload(data)
