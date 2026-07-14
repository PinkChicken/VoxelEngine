class_name MaterialDB extends Node

@export var materials: Array[MaterialDef] = []
@export var max_materials: int = 256

var buffer_rid: RID
var _compute

var _id_to_def: Dictionary = {}


func _ready() -> void:
	_compute = preload("res://Utils/compute_helper.gd").new()
	if materials.is_empty():
		materials = _default_materials()
	_index_materials()
	build_buffer()


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

	var sand := MaterialDef.new()
	sand.id = 2; sand.name = "SAND"
	sand.phase = MaterialDef.Phase.SOLID
	sand.simulation = MaterialDef.SimulationClass.GRANULAR
	sand.gravity = MaterialDef.GravityMode.DOWN
	sand.color = Color(0.867, 0.8, 0.533, 1)

	var water := MaterialDef.new()
	water.id = 3; water.name = "WATER"
	water.phase = MaterialDef.Phase.LIQUID
	water.simulation = MaterialDef.SimulationClass.LIQUID
	water.gravity = MaterialDef.GravityMode.DOWN
	water.color = Color(0.2, 0.4, 0.9, 0.7)
	water.flow = 1.0  # thin — the viscosity baseline every other liquid is scaled against
	water.density = 1.0  # baseline every other liquid's density is set relative to

	var glass := MaterialDef.new()
	glass.id = 4; glass.name = "GLASS"
	glass.phase = MaterialDef.Phase.SOLID
	glass.simulation = MaterialDef.SimulationClass.NONE
	glass.gravity = MaterialDef.GravityMode.NONE
	glass.color = Color(0.75, 0.92, 0.97, 1)
	glass.transparent = true

	var fire := MaterialDef.new()
	fire.id = 5; fire.name = "FIRE"
	fire.phase = MaterialDef.Phase.EFFECT
	fire.simulation = MaterialDef.SimulationClass.EFFECT
	fire.gravity = MaterialDef.GravityMode.UP
	fire.color = Color(1.0, 0.6, 0.1, 0.9)

	var smoke := MaterialDef.new()
	smoke.id = 6; smoke.name = "SMOKE"
	smoke.phase = MaterialDef.Phase.GAS
	smoke.simulation = MaterialDef.SimulationClass.GAS
	smoke.gravity = MaterialDef.GravityMode.UP
	smoke.color = Color(0.3, 0.3, 0.3, 0.2)

	var oil := MaterialDef.new()
	oil.id = 7; oil.name = "OIL"
	oil.phase = MaterialDef.Phase.LIQUID
	oil.simulation = MaterialDef.SimulationClass.LIQUID
	oil.gravity = MaterialDef.GravityMode.DOWN
	oil.color = Color(0.2, 0.15, 0.05, 0.85)
	oil.flow = 0.69  # thicker than water — runs through the same viscosity formula (fluid_flux.comp)
	oil.density = 0.85  # lighter than water (1.0) — floats on top when density-stacking is on

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
