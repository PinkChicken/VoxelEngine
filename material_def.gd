class_name MaterialDef extends Resource

enum Phase {
	SOLID  = 0,
	LIQUID = 1,
	GAS    = 2,
	EFFECT = 3,
}

enum SimulationClass {
	NONE     = 0,  # rock, wood, air — no simulation
	GRANULAR = 1,  # sand, gravel
	LIQUID   = 2,  # water, oil
	GAS      = 3,  # smoke, steam
	EFFECT   = 4,  # fire, sparks
}

const SIMULATION_COUNT := 5  # size of simulation_counts array in Chunk

# Precomputed bitmask for each SimulationClass, indexed by value.
# SIM_BITS[NONE]=0, SIM_BITS[GRANULAR]=1, SIM_BITS[LIQUID]=2, SIM_BITS[GAS]=4, SIM_BITS[EFFECT]=8
const SIM_BITS: Array[int] = [0, 1, 2, 4, 8]

enum GravityMode {
	NONE    = 0,  # static / unaffected
	DOWN    = 1,  # sand, water
	UP      = 2,  # smoke, steam
	NEUTRAL = 3,  # spreads laterally (gas clouds)
}

@export var id: int = 0
@export var name: String = ""
@export var phase: int = Phase.SOLID
@export var simulation: int = SimulationClass.NONE
@export var gravity: int = GravityMode.NONE
@export var color: Color = Color.WHITE
