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

# MLS-MPM constitutive model. NOT_SIMULATED (default) means "no MPM particle for this
# material" — the id never gets a particle, same as AIR/FIRE/SMOKE today (see
# tools/fluid_prototypes/MPM_EVALUATION_BRIEF.md §35.4/§36 for why gas/fire are pending
# a separate design decision rather than defaulted here).
enum MPMModel {
	NOT_SIMULATED = 0,
	STATIC        = 1,  # zero-mass wall particle — no-slip flag on its grid neighbourhood, never moves (rock, glass)
	FLUID         = 2,  # weakly-compressible J-model + viscosity shear term (water, oil, lava)
	GRANULAR      = 3,  # deformation-gradient F + Drucker-Prager return mapping (sand)
	ELASTIC       = 4,  # fixed corotated — reuses GRANULAR's SVD, no return mapping (unused today)
}

@export var id: int = 0
@export var name: String = ""
@export var phase: int = Phase.SOLID
@export var simulation: int = SimulationClass.NONE
@export var gravity: int = GravityMode.NONE
@export var color: Color = Color.WHITE
@export var transparent: bool = false
# Per-material viscosity scalar for LIQUID simulation (fluid_flux.comp's MaterialFlow
# buffer, WorldOrchestrator._build_material_cache()). 1.0 = thin/water (bit-identical to
# the pre-per-material-flow shader); smaller = thicker (eases toward sluggish per the
# viscosity formula — see VISCOSITY_NOTES.md). Ignored by non-liquid materials.
@export_range(0.05, 1.0, 0.01) var flow: float = 1.0
# Per-material density scalar for density-ordered liquid stacking (fluid_flux.comp's
# MaterialDensity buffer, WorldOrchestrator._build_material_cache()) — 2026-07-12,
# generalizing what was previously hardcoded to WATER/OIL in the trigger pass. 0.0
# (default) means "not a density-sorted material": the trigger only considers a pair
# when BOTH sides have density > 0, so solids/gases/unregistered ids never accidentally
# participate just by touching a real liquid. Relative scale only matters, not absolute
# units — water=1.0 is the baseline every other liquid is set relative to; any liquid
# with a SMALLER positive density than a neighbouring liquid floats above it.
@export_range(0.0, 5.0, 0.01) var density: float = 0.0

# --- MLS-MPM properties (tools/fluid_prototypes/MPM_EVALUATION_BRIEF.md §35.6 M1) ---
# Deliberately separate from `density` above: that field's 0.0-sentinel ("not a
# density-sorted material") is read live by the still-running flux system's trigger
# logic. Giving a solid a real MPM density in the SAME field would silently activate
# flux's density-sort trigger for materials it was built to exclude. Split until CA
# is deleted, then `density`/`mpm_density` can merge.
@export var mpm_model: int = MPMModel.NOT_SIMULATED
@export_range(0.0, 5.0, 0.01) var mpm_density: float = 0.0
# FLUID: weakly-compressible bulk stiffness E (mpm2d_hybrid.py/p2g.comp's `E`). Higher =
# closer to incompressible ("more water-like"); water's shipped default was 400, later
# swept and re-defaulted to 8000 for a tighter J (brief §34.3). GRANULAR/ELASTIC/STATIC ignore.
@export var mpm_stiffness: float = 0.0
# FLUID only: shear viscosity term (brief §35.1a / §36's MPM-can-do-gas-too follow-up —
# this is the field that was missing entirely from the retired prototype and is why water
# and oil were physically identical apart from mass). 0.0 = pressure-only, byte-for-byte
# the old behaviour. Mirrors MaterialDef.flow's role for the flux system but is NOT the
# same scale/units — flow is a flux-viscosity multiplier, this is a real MPM stress coefficient.
@export var mpm_viscosity: float = 0.0
# GRANULAR only: Drucker-Prager friction angle in degrees, converted to the alpha_sand
# constant the same way mpm_simulation.gd's retired set_sand_friction_deg() did.
@export_range(0.0, 60.0, 0.1) var mpm_friction_deg: float = 0.0
