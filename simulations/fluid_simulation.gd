class_name FluidSimulation extends VoxelSimulation

# GPU dispatcher for liquid simulation (water, oil) — Phase 7e persistent-flux pipeline.
# Core loop per sub-step, all in fluid_flux.comp: Accelerate -> Transport. (Was four
# passes -- Accelerate -> Scale-Outflow -> Scale-Inflow -> Transport -- until Phase 7e
# Step 9, 2026-07-06: Transport now recomputes each cell's conservation-scale factor
# inline as a pure function of already-synced flux_buf/voxels_a data instead of reading it
# back from a buffer a prior pass wrote, so the two Scale passes and their rewritten-lane
# race-avoidance proof are gone entirely. See fluid_flux.comp's header comment.)
# No CAS, no voxels_b scratch buffer — Transport writes the changelog directly, same
# single-writer discipline as every other simulation.
#
# Phi-flood (2026-07-08, tools/fluid_prototypes/python/phi_flood.py) — three more passes
# (PhiSeed -> PhiFloodStep x(K-1) -> PhiFloodStepFinal) now run BEFORE Accelerate each
# sub-step, replacing head() with a piezometric pressure field as Accelerate's dh input.
# This is the connected-vessel/U-tube fix: head()'s strictly-local column scan has zero
# visibility across a saturated horizontal connector (see docs/CONNECTED_VESSEL_HANDOFF.md
# §3); the flood gives real non-local pressure awareness while reducing to head() exactly
# for any settled, isolated body. See step()'s dispatch loop and _build_phi_chunk_list()
# for the mechanism and the (separate) sleep-scheduler fix this required.
#
# Phase 7e Step 7 (2026-07-01): the legacy vertical-CAS shader (fluid_sim.comp) has been
# deleted. It was dispatched every sub-step BEFORE Pass A/Scale/Transport and its
# try_compress still contained a destructive "snap remainder below 32 to zero" hack from
# before this redesign existed (a patch for a problem Scale's conservation guarantee now
# solves properly). With both systems running, vertical CAS was silently eating fill every
# time it compressed a low-fill cell created by Transport — confirmed via step_once() +
# debug_dump_fluid_stats(): a 255-fill droplet decayed 255->255->220->220->165->102->51->1
# over 7 ticks, losses correlating exactly with the growing count of small-fill voxels
# Transport was creating. Y (vertical) flow is now handled entirely by fluid_flux.comp's
# Y-axis flux (head-driven, same mechanism as X/Z) — see ARCHITECTURE.md Step 7.

var _compute
var _flux_pipeline: Dictionary = {}   # fluid_flux.comp — Accelerate / Scale / Transport
var _flux_params_buf: RID             # 64-byte params for fluid_flux.comp (incl. pass, damp, g_dt, gravity_bias, settle_threshold, phi_read_buf, flow, tint_mode)
var _flux_chunk_list_buf: RID         # active + 6 face neighbours — flux dispatch range
var _flux_chunk_list_capacity: int = 0
var _dirty_chunks_buf: RID
# Phase 7e Step 1 — privately owned by FluidSimulation (was WorldOrchestrator._flux_buf).
# Nothing outside fluid ever reads flux, so it never belongs on the Orchestrator's
# shared buffer table.
#
# Phase 7e Step 2 — sparse paging. _flux_buf is no longer sized to the full voxel-slot
# capacity; it holds one "page" (vpc entries) per chunk that has ever been a fluid
# candidate, allocated from a freelist and indexed indirectly through _flux_chunk_table.
# A chunk with no page is a zero-flux boundary, not an error — most loaded chunks never
# touch fluid and should cost zero flux storage.
var _flux_buf: RID
var _flux_page_capacity: int = 0        # total pages currently backing _flux_buf
var _flux_freelist: Array = []          # free page indices (int), pop_back() to allocate
var _flux_chunk_table: Dictionary = {}  # chunk_pos → page index (CPU bookkeeping)
# Idle-guard for pages allocated purely as flux-dispatch neighbours (chunk-boundary fix,
# 2026-07-02) that never actually became active — chunk_pos → idle ticks. Separate from
# _active_chunks's own idle tracking, which only covers chunks that were simulating.
var _flux_page_idle: Dictionary = {}
# GPU-side mirror of _flux_chunk_table — a shader can't read a GDScript Dictionary.
# Dense int array, indexed exactly like WorldOrchestrator._chunk_table (grid-flat index),
# value = page index or -1 (no page / zero-flux boundary). Sized to grid_span, which is
# fixed after WorldOrchestrator._init_grid() — unlike _flux_buf, this never needs to grow.
var _flux_page_table_buf: RID
const FLUX_PAGE_INITIAL_CAPACITY := 64

# Phi-flood (2026-07-08 port of tools/fluid_prototypes/python/phi_flood.py) — piezometric
# pressure field replacing head() as Pass A's dh input, giving the flux system real
# non-local awareness across a connected vessel (the U-tube's structurally-blind saturated
# horizontal connector — see docs/CONNECTED_VESSEL_HANDOFF.md §3). Two buffers because the
# flood is a Jacobi relaxation: each iteration must read last iteration's FULLY-settled
# field while writing this one's, so a single shared buffer would race. Paged 1:1 with
# _flux_buf via the SAME _flux_chunk_table/_flux_page_table_buf (stride 1 int per voxel,
# not 3) — no new page table or allocator; a "page" now reserves flux + phi storage
# together. See _ensure_flux_buf_capacity/_allocate_flux_page/_free_flux_page for the
# lockstep sizing/zeroing, and _build_phi_chunk_list for why the phi DISPATCH RANGE is
# wider than the flux one (the second structural bug found in the Python port: the sleep
# scheduler is itself connector-blind).
var _phi_buf_a: RID
var _phi_buf_b: RID
# K, the number of Jacobi flood iterations per sub-step. IMPORTANT (found 2026-07-08,
# in-editor KEY_U testing): phi is re-seeded and re-flooded FROM SCRATCH every tick, with
# no field persisted across ticks (matches the validated Python port exactly — see
# phi_flood.py's module docstring). That makes K a hard PER-TICK REACH BUDGET, not a
# "converges eventually given enough ticks" iteration count: pressure information can
# travel at most K voxel-hops from a free surface before this tick's flood simply stops,
# and since nothing carries over, two seeds farther apart than K can NEVER meet, no matter
# how many ticks pass. It must therefore cover the worst-case path a real rig needs —
# roughly (down one arm) + (across the connector) + (up the other arm) in voxels.
#
# The Python bench's own measured convergence (avg 7, max ~30 hops) was against its small
# test rigs and does NOT transfer to the shipped KEY_U scene, whose arms are much taller
# (world.gd's UTUBE_ARM_HEIGHT = 32). Confirmed empirically in-editor: K=16 left the U-tube
# visibly uneven (~2x mass imbalance between arms, stable — not still converging); K=32
# fully equalized it. Default raised to 40 for headroom above the demonstrated-working 32;
# if a future scene uses much taller arms or a much longer connector, this needs raising
# again (or the whole approach needs cross-tick warm-starting — not implemented — see
# docs/PHI_FLOOD_PORT_PLAN.md). Under-convergence is still safe either way: phi degrades
# toward head()'s value and never overshoots, so a too-small K produces a stable but
# uneven rest, never an explosion or leak.
@export_range(1, 96, 1) var phi_flood_iterations: int = 40

# Automatic conservation check (2026-07-02) — one signed int, atomicAdd'd by Transport
# with every real (new_fill - old_fill) it applies. Zero after a tick means mass
# conserved exactly; nonzero is a direct, quantified leak/gain, no manual F6 diffing
# needed. See step()'s Transport block for the read/reset side.
var _conservation_buf: RID
# Location tracker (2026-07-02) — the stale-page fix didn't resolve the leak, so this
# pinpoints WHERE the single largest gain each tick happens instead of guessing further.
# 8 bytes: [flat_idx: s32, amount: s32]. Plain (non-atomic) write on the shader side.
var _gain_tracker_buf: RID
# Density-trigger debug tint scratch (2026-07-12, see debug_density_trigger_tint) — one
# uint per FULL FLAT VOXEL INDEX (slot * vpc + local, same convention as voxels_a/
# changelog), NOT per chunk slot. Sized via get_voxel_buf_capacity() * chunk_size^3 —
# get_voxel_buf_capacity() returns chunk-slot count; multiplying is mandatory here (see
# docs/BUG_LOG.md "_ensure_demand_accum_capacity() sized its scratch buffer to the
# chunk-slot count..." — the exact bug this sizing must not repeat).
var _density_touch_buf: RID
var _density_touch_capacity: int = 0
# Shared CAS claim scratch (2026-07-14) — one uint per FULL FLAT VOXEL INDEX, same sizing
# rule as _density_touch_buf above. Pass 6 zeroes it; used by any claim-based resolution
# pass (currently BuoyancyExchange, pass 8) so parallel exchanges/moves can't collide.
var _momentum_claim_buf: RID
var _momentum_claim_capacity: int = 0
# Per-voxel fill history (2026-07-16, see stuck_rest_ticks) — one uint per FULL FLAT VOXEL
# INDEX, same sizing rule as _density_touch_buf above. Low 16 bits = last-seen
# (material|fill) u16, high 16 = consecutive unchanged tick count (saturating). Written
# only by fluid_flux.comp pass 10 (FillHistoryUpdate), read by buoy_at_rest(). Persistent —
# never zeroed between ticks (the whole point is remembering across ticks); a reused chunk
# slot's stale entry self-corrects on the next pass-10 dispatch (content mismatch resets).
var _fill_history_buf: RID
var _fill_history_capacity: int = 0

# --- Momentum punch demand queue (Phase 5 port step 2b/3, 2026-07-15, see use_punch) ---
# Written by fluid_flux.comp pass 9 (MomentumPunch); drained by momentum_resolve.comp's
# own separate pipeline. Same interleaved shape as the old (deleted) density-stacking
# demand queue: {count, entries[]}, 4 ints/entry (origin_x,y,z, material|amount<<8).
var _punch_demand_queue_buf: RID
var _punch_demand_accum_buf: RID       # per FULL FLAT VOXEL INDEX — atomicAdd accumulator
var _punch_demand_accum_capacity: int = 0
var _punch_touched_cells_buf: RID      # {count, entries[]} — 4 ints/entry (x,y,z,material)
var _punch_demand_requeue_buf: RID     # {count, entries[]} — carried into next substep's queue
const MAX_PUNCH_DEMAND_ENTRIES := 262144  # must match momentum_resolve.comp / fluid_flux.comp's own literal
var _momentum_resolve_pipeline: Dictionary = {}
var _momentum_resolve_params_buf: RID  # 40 bytes: chunk_size, grid_min xyz, grid_span xyz, pass, settle_threshold, rescue
# Standing punch-demand backlog (2026-07-17) — the queue count left over after each
# resolve cycle's requeue carry-forward (one 4-byte readback per cycle). Nonzero while the
# active set is empty = the frozen-forever state the F6 punch-queue dump exposed (269,404
# fill stranded in one orphaned entry after a submerged-sphere plunge): nothing dirty for
# 4+ ticks emptied the active set, so the resolver never ran again and the queued mass
# stayed missing from the world indefinitely. has_work() reports true while this is
# nonzero so SimulationRuntime keeps calling step(), and step()'s idle path then runs
# _rescue_punch_backlog() — see both below.
var _punch_backlog: int = 0

var _orchestrator: WorldOrchestrator
var _chunk_manager: ChunkManager
var _active_chunks: Dictionary = {}
var _frame: int = 0

# Voxel watch (2026-07-02) — GainTracker only ever shows the CURRENT tick's single
# worst offender, which can jump between correlated cells and never reveals one cell's
# actual trajectory over time. This watches one fixed (chunk, local) voxel every
# sub-step and prints its fill + all 3 flux lanes, to see oscillation shape directly
# instead of inferring it from aggregate conservation numbers. Exported (not a function
# call) so it can be toggled/edited live from the remote Inspector while playing —
# no debugger console access needed. Defaults to the recurring offender found in the
# resting-hole-water leak (chunk (0,-3,-1) local (1,28,30)); zero extra GPU work when
# watch_enabled is false.
@export var watch_enabled: bool = false
@export var watch_chunk: Vector3i = Vector3i(0, -3, -1)
@export var watch_local: Vector3i = Vector3i(1, 28, 30)

# Automatic conservation checker toggle (default false, added 2026-07-03) — gates a
# GPU readback that forces a sync every sub-step; see the in-step() comment for why this
# was left on by default during the leak investigation and needs to be off otherwise.
@export var debug_conservation_check: bool = false

# Buoyancy trigger tint (Phase 1 of the momentum/buoyancy port, 2026-07-14) — an
# unmissable in-editor marker on exactly the voxels buoyancy.py's trigger would fire on,
# written BEFORE the mutating exchange resolution (Phase 2), per the debug-tint-before-
# mutation discipline the 2026-07-10 "bench-valid but no visible in-editor effect"
# mystery taught (three bench-validated GLSL changes each reported as "looks exactly the
# same" — a new pass's silence must never again be mistaken for "not dispatching"). This
# pass (fluid_flux.comp pass 5) does ONE thing: every dispatched voxel overwrites its OWN
# flat slot in _density_touch_buf with 1 if it's a LIGHTER liquid, at rest, with a DENSER
# different liquid it could rise into (straight-up always, diagonal-up if isolated —
# buoyancy.py's trigger exactly), else 0 (self-write only — no neighbour writes, no CAS,
# no race; and self-clearing — a slot that stops qualifying, including because a DIFFERENT
# material now occupies that flat index, goes back to 0 next dispatch, never showing stale
# tint on whatever moved in). No voxel data is mutated, no changelog entry is written, no
# mass moves. Purely additive and fully inert when false (pass 5 is skipped entirely; the
# buffer stays allocated at minimum size so the renderer's uniform set never changes shape).
@export var debug_density_trigger_tint: bool = false

# Which trigger the tint (pass 5) visualizes: 0 = buoyancy (lighter liquid at rest that
# could rise), 2 = punch (liquid arriving with real downward flux over bulk lighter
# liquid). Mode 1 (the retired push-aside momentum trigger) is no longer set by any
# toggle. Set by world.gd's KEY_T (buoyancy) / KEY_H (punch) toggles. Only meaningful
# while debug_density_trigger_tint is true.
@export var tint_mode: int = 0

# REMOVED (2026-07-15): the single-hop push-aside MomentumPush (use_momentum_push,
# passes 6/7, KEY_M/KEY_G) — superseded by the momentum-graded punch + min-Y demand
# resolver validated in tools/fluid_prototypes/python/momentum_demand.py. See
# ARCHITECTURE.md's momentum/buoyancy to-do item for the full history.

# Buoyancy (Phase 4 of the momentum/buoyancy port, 2026-07-14) — the liquid-interaction
# "aha": a LIGHTER liquid at rest with a DENSER different liquid above it trades places
# (lighter rises, denser sinks), so water sinks into oil, oil rises, a mixed pool sorts.
# Direct 2-cell exchange (pass 8) — exact conservation, no air needed. Rest-gated, so it
# acts on calm liquid and defers anything still falling. Port of buoyancy.py's validated
# exchange resolution. Runs after Transport in its own changelog cycle. Toggle in-editor
# with KEY_N (see world.gd). Defaulted to true 2026-07-16 — this is the shipped mechanism,
# not an experimental toggle; defaulting it off meant every fresh test session silently
# needed KEY_N pressed first or was testing base-flux-only behaviour without realizing it
# (cost real diagnostic time chasing symptoms that were actually just "buoyancy is off").
@export var use_buoyancy: bool = true

# Buoyancy rest tolerance (raw flux units) — max |flux| on a cell's own+behind lanes for
# it to still count as "at rest" for the buoyancy exchange AND the pass-5 buoyancy tint.
# The bench (momentum_demand.py, "SETTLE SPEED" knob) mapped this: 0 waits for dead-still
# (slow, ~920-tick settle); ~40-50 sorts nearly-still cells (~4.6x faster) while STILL
# holding climb=0; CEILING ~100 — above it buoyancy fires on MOVING stream cells, oil rises
# into the stream, and the chimney returns. Default 40 = the validated fast-and-clean sweet
# spot; do not exceed ~80.
@export_range(0, 64, 1) var buoy_rest_flux: int = 40

# Fill-history rest override (2026-07-16 port of momentum_demand.py's stuck_rest_ticks —
# the "water film permanently stuck on oil" fix, LIQUID_INTERACTION_HANDOFF.md §3.2). A
# liquid cell whose (material|fill) bytes have been byte-identical for this many
# consecutive ticks counts as at rest for the buoyancy exchange no matter what the flux
# field claims: phi-flood re-saturates a partial-fill surface cell's own upward lane every
# tick whether the cell is moving or not, so flux magnitude can't distinguish "still
# splashing" from "structurally stuck" — actual content change can. Bench-validated at 12:
# waterfall suite holds climb_events=0 AND the test_stuck_film.py FILM repro (the first
# bench reproduction of the in-editor stick) sorts in ~N+1 ticks. -1 disables (pre-fix
# behaviour). Backed by _fill_history_buf + fluid_flux.comp pass 10 (FillHistoryUpdate).
@export var stuck_rest_ticks: int = 12

# Momentum punch knobs (Phase 5 port of momentum_demand.py, 2026-07-15) — the CRATER
# STRENGTH group. A liquid cell whose downward flux reaches punch_threshold craters
# straight down through `reach = 1 + (F - punch_threshold)/punch_scale` (capped at
# max_reach) bulk-lighter cells. LOWER threshold/scale = stronger crater; both hold climb=0
# (displaced oil only ever routes down/laterally). Validated defaults 120/90/3 (see the
# bench KNOBS table). Only read by the punch tint (pass 5 mode 2) and the punch pass.
@export var punch_threshold: int = 120
@export var punch_scale: int = 90
@export_range(1, 12, 1) var max_reach: int = 3

# Punch demand-resolve walk cap (2026-07-17) — how many cells the min-Y BFS searches for a
# dischargeable surface per demand, live-tunable from the Inspector. Was a hard-coded 64 in
# momentum_resolve.comp; the in-editor submerged-sphere test stranded a demand whose deep,
# buried origin couldn't reach a distant rising surface within 64 (see docs/BUG_LOG.md's
# buried-origin entry). Clamped in-shader to WALK_CAP_MAX=512 (the visited[] array is a
# fixed compile-time size; only the loop bound is dynamic). Raise this to test whether reach
# alone lets deep displaced oil route out; the bench uses 512.
@export_range(8, 128, 1) var punch_walk_cap: int = 64

# Momentum punch (Phase 5 port step 2b/3, 2026-07-15) — the actual mutation: a fast-falling
# liquid craters straight down through bulk lighter liquid (pass 9), queuing each displaced
# cell as a demand drained by momentum_resolve.comp's min-Y walk (pass 0 walk + pass 1
# apply). Port of tools/fluid_prototypes/python/momentum_demand.py's validated
# MomentumDemandFluidSim — denser-into-lighter, different-material, bulk-only (see
# punch_candidate()'s own comment for why material-agnostic was rejected), no bubble
# landing, fallback rise only into same-material headroom (never bare air) — the exact
# combination the bench proved gives climb=0 AND full sorting AND sealed-basin
# conservation. Toggle in-editor with KEY_J. Defaulted to true 2026-07-16, same reasoning
# as use_buoyancy above — this is the shipped mechanism, not an experimental toggle.
@export var use_punch: bool = true

# Full per-voxel diagnostic dump (fill, own 3 owned lanes, 3 incoming neighbour-owned
# lanes) — factored out (2026-07-03) so both the fixed watch target AND the
# conservation checker's per-tick "biggest gain" location can share it. Watching a
# fixed cell only helps if the leak happens to recur there; the gain-tracker's flagged
# location moves every tick (confirmed 2026-07-03: manually retargeting watch_chunk/
# watch_local after the fact kept missing the actual leaking cell, since the location
# reported one tick was rarely still the worst offender when the user could react to
# it), so _debug_print_conservation() below calls this directly on the CURRENT tick's
# flagged voxel instead of relying on the user to catch up to a moving target.
func _debug_dump_voxel(label: String, chunk_pos: Vector3i, local: Vector3i) -> void:
	var slot: int = _orchestrator._chunk_to_slot.get(chunk_pos, -1)
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var lx := local.x
	var ly := local.y
	var lz := local.z
	var fill_str := "n/a"
	var mat_str := "n/a"
	if slot >= 0:
		var byte_idx := (slot * vpc + lz * cs * cs + ly * cs + lx) * Chunk.VOXEL_STRIDE
		var vdata: PackedByteArray = _compute.download_range(_orchestrator.get_voxel_buf(), byte_idx, 4)
		mat_str = str(vdata.decode_u8(0))
		fill_str = str(vdata.decode_u8(1))
	var page: int = _flux_chunk_table.get(chunk_pos, -1)
	var flux_str := "n/a (no page)"
	if page >= 0:
		var local_idx := lz * cs * cs + ly * cs + lx
		var lane_base := (page * vpc * 3 + local_idx * 3) * 4
		var fdata: PackedByteArray = _compute.download_range(_flux_buf, lane_base, 12)
		flux_str = "(fx=%d, fy=%d, fz=%d)" % [fdata.decode_s32(0), fdata.decode_s32(4), fdata.decode_s32(8)]

	# Incoming neighbour-owned lanes (2026-07-03) — the voxel's OWN 3 owned lanes above
	# only tell half the story: Transport's delta also reads the -axis neighbour's owned
	# face pointing INTO this voxel (delta += eff_flux(flux_buf[nlane])). A voxel whose
	# own 3 lanes read all-zero can still have its fill change every tick if a neighbour's
	# owned face is nonzero — exactly the pattern that motivated this addition (found
	# 2026-07-03: a watched voxel's fill oscillated 236<->240 forever while fx/fy/fz all
	# printed 0, meaning the real driver was invisible until now).
	var world_pos := chunk_pos * cs + local
	var axis_dir := [Vector3i(1, 0, 0), Vector3i(0, 1, 0), Vector3i(0, 0, 1)]
	var axis_name := ["X", "Y", "Z"]
	var incoming_str := ""
	for axis in range(3):
		var nb_world: Vector3i = world_pos - axis_dir[axis]
		var nb_chunk := _chunk_manager.voxel_to_chunk_position(nb_world)
		var nb_local := _chunk_manager.global_to_local(nb_world)
		var nb_page: int = _flux_chunk_table.get(nb_chunk, -1)
		var val_str := "n/a"
		if nb_page >= 0:
			var nb_local_idx := nb_local.z * cs * cs + nb_local.y * cs + nb_local.x
			var nb_lane_base := (nb_page * vpc * 3 + nb_local_idx * 3 + axis) * 4
			var ndata: PackedByteArray = _compute.download_range(_flux_buf, nb_lane_base, 4)
			val_str = str(ndata.decode_s32(0))
		incoming_str += " in_%s(from %s)=%s" % [axis_name[axis], nb_chunk, val_str]

	print("[%s] chunk %s local (%d,%d,%d): mat=%s fill=%s flux=%s%s" % [label, chunk_pos, lx, ly, lz, mat_str, fill_str, flux_str, incoming_str])


func _debug_print_watch() -> void:
	if not watch_enabled:
		return
	_debug_dump_voxel("FluidWatch", watch_chunk, watch_local)

# Sub-steps per simulation tick — higher = faster fluid, more GPU cost.
# Set to 1 for slow lava-like materials; 2-3 for water.
@export var steps_per_tick: int = 1

# Phase 7e Step 4 tuning — live-editable on the running instance via the Godot remote
# Inspector (select the FluidSimulation node under Remote in the Scene dock while playing).
# damp: fraction of last tick's flux retained before this tick's acceleration is added.
# Lower = more viscous (settles fast, little slosh). Higher = more wave/ripple, longer-lived.
@export_range(0.0, 1.0, 0.001) var damp: float = 0.97
# g_dt: acceleration applied per unit of hydrostatic head difference. Head ranges 0–255
# for a single cell (up to ~6120 for a deep column), and FILL_CAP clamps a single tick's
# flux to 255 — so g_dt needs to be small: a value of 6.0 made a head diff of just ~43
# saturate flux to the maximum possible transfer, meaning a full droplet could empty
# itself in one tick instead of building pressure gradually. Keep this well under 1.0.
#
# Range capped at 0.5 (was 2.0, found 2026-07-02): gravity_bias multiplies this on the Y
# axis by up to its own max — at the old bounds, g_dt=2.0 * gravity_bias=8.0 gives an
# EFFECTIVE Y multiplier of 16, saturating flux for any dh >= 16 (very common, even a
# shallow puddle). Saturating isn't itself the bug (FILL_CAP always clamps to +-255) — the
# bug is that once MULTIPLE competing directions all hit that SAME ceiling, they lose the
# relative distinction gravity_bias exists to create, and Scale divides the budget roughly
# evenly among them again — the exact symmetric-spreading problem gravity_bias fixed,
# reintroduced by oversaturation. Combined with damp carrying a saturated value forward
# each tick, this can oscillate without net progress, which reads as water freezing solid
# mid-air. Reported symptom: raising g_dt froze standing full-fill (255) columns in place;
# resetting g_dt to 0 let stored flux cleanly decay via *damp alone, so ramping back up
# started clean and worked again — until enough ticks/new water pushed it back into the
# saturating regime. This range keeps g_dt*gravity_bias inside the region that stays
# proportional instead of collapsing to the clamp ceiling for typical head differences.
@export_range(0.0, 0.5, 0.01) var g_dt: float = 0.15

# gravity_bias: multiplier on the (dh - elevation) term, Y-axis only (see
# fluid_flux.comp's elevation-head fix, 2026-07-04). Without this, Pass A's formula is
# fully symmetric across axes — a cell's push strength toward an exposed neighbour is
# identical whether that neighbour is above, below, or to the side, producing
# octahedral/spherical expansion instead of coherent falling. 1.0 = no bias (original
# symmetric behavior, useful for regression testing). Range capped at 5.0 (was 8.0) for
# the same oversaturation reason as g_dt above — see its comment.
#
# gravity_force (a separate flat downward constant this variable used to be paired
# with) was removed 2026-07-04 — it broke hydrostatic equilibrium (see docs/BUG_LOG.md
# "ROOT CAUSE FOUND (2026-07-03)"): head() is pressure head only, so a settled column
# always has a nonzero head difference across every vertical face (255, not 0), and a
# flat constant only cancels that by coincidence of tuning. Fixed by using elevation
# head instead (Y-axis dh is now `dh - FILL_CAP`, computed directly in the shader), which
# makes rest a genuine fixed point and gives a floating mass its baseline pull for free.
@export_range(1.0, 5.0, 0.1) var gravity_bias: float = 3.0

# settle_threshold: minimum |applied fill change| (see fluid_flux.comp Transport) for a
# voxel's change to count toward keeping its chunk in _active_chunks next tick. Separate
# from and coarser than MIN_FLOW in the shader (which governs conservation correctness —
# untouched by this). A large settled body of water can slosh at 1-3 units/tick forever
# (flux_buf intentionally persists across ticks so waves don't instantly damp), which
# with the old "any nonzero change keeps the chunk active" rule meant a big lake never
# fully slept even once visually and physically settled. Raising this lets FluidSimulation
# treat sub-threshold sloshing as "settled enough to sleep" — after 4 idle ticks under
# this softer rule the chunk's flux page is freed AND zeroed (see _free_flux_page()), so
# sleep and clearing residual momentum happen together. 1 = old behavior (any nonzero
# change keeps the chunk awake indefinitely).
@export_range(1, 20, 1) var settle_threshold: int = 3

# flow: single-knob viscosity, 0 < flow <= 1 — 1.0 = thin/water (bit-identical to the
# pre-viscosity shader, special-cased), smaller = thicker. Bench-validated 2026-07-08
# (tools/fluid_prototypes/python/viscosity.py + VISCOSITY_NOTES.md, knob-isolation
# spread matrix + viscous U-tube + settle/sleep gates): inside Pass A it eases damp
# toward 0.6 (kills slosh AND lowers steady-state flow speed) and scales g_dt +
# axis_cap linearly (sluggish response + throttled pour/fall throughput). Rough feel:
# ~0.5 syrupy, ~0.25 thick, ~0.1 honey dollop that barely creeps yet still sleeps.
# CAVEAT (measured, not hidden): low flow coarsens U-tube equalization — accel
# magnitudes under 0.5 round to zero, a yield-stress-like dead zone (a thick fluid
# holds small imbalances; at flow=0.4 the bench U-tube froze at a ~3.5% residual gap).
# Keep >= ~0.85 if sub-voxel leveling precision matters. Currently GLOBAL (all
# liquids); per-material (MaterialDef + a 256-float buffer keyed by active_mat, form
# already bench-validated) is the planned follow-up once a second liquid (oil) exists.
@export_range(0.05, 1.0, 0.01) var flow: float = 1.0

const DIRTY_BUF_BYTES := 64


func _ready() -> void:
	_orchestrator  = get_node("/root/world/WorldOrchestrator")
	_chunk_manager = get_node("/root/world/ChunkManager")
	_compute = preload("res://Utils/compute_helper.gd").new()
	if _compute.has_device():
		_flux_pipeline = _compute.compile_shader(_load_flux_shader(), 4)
		if _flux_pipeline.is_empty():
			push_error("FluidSimulation: fluid_flux.comp compile failed")
		var fp := PackedByteArray(); fp.resize(84); fp.fill(0)  # ...flow, tint_mode, buoy_rest_flux, punch knobs, stuck_rest_ticks
		_flux_params_buf = _compute.buffer_upload(fp)
		_flux_chunk_list_capacity = 256
		var fl := PackedByteArray(); fl.resize(_flux_chunk_list_capacity * 12); fl.fill(0)
		_flux_chunk_list_buf = _compute.buffer_upload(fl)
		var db := PackedByteArray(); db.resize(DIRTY_BUF_BYTES); db.fill(0)
		_dirty_chunks_buf = _compute.buffer_upload(db)
		var cb := PackedByteArray(); cb.resize(4); cb.fill(0)
		_conservation_buf = _compute.buffer_upload(cb)
		var gb := PackedByteArray(); gb.resize(8)
		gb.encode_s32(0, -1); gb.encode_s32(4, 0)  # flat_idx=-1 (no candidate yet), amount=0
		_gain_tracker_buf = _compute.buffer_upload(gb)
		var dt := PackedByteArray(); dt.resize(4); dt.fill(0)
		_density_touch_capacity = 1
		_density_touch_buf = _compute.buffer_upload(dt)
		var mc := PackedByteArray(); mc.resize(4); mc.fill(0)
		_momentum_claim_capacity = 1
		_momentum_claim_buf = _compute.buffer_upload(mc)
		var fh := PackedByteArray(); fh.resize(4); fh.fill(0)
		_fill_history_capacity = 1
		_fill_history_buf = _compute.buffer_upload(fh)

		# --- Momentum punch demand-queue buffers (2026-07-15, see use_punch) ---
		var pdq := PackedByteArray(); pdq.resize(4 + MAX_PUNCH_DEMAND_ENTRIES * 16); pdq.fill(0)
		_punch_demand_queue_buf = _compute.buffer_upload(pdq)
		var ptc := PackedByteArray(); ptc.resize(4 + MAX_PUNCH_DEMAND_ENTRIES * 16); ptc.fill(0)
		_punch_touched_cells_buf = _compute.buffer_upload(ptc)
		var pdr := PackedByteArray(); pdr.resize(4 + MAX_PUNCH_DEMAND_ENTRIES * 16); pdr.fill(0)
		_punch_demand_requeue_buf = _compute.buffer_upload(pdr)
		_momentum_resolve_pipeline = _compute.compile_shader(_load_momentum_resolve_shader(), 4)
		if _momentum_resolve_pipeline.is_empty():
			push_error("FluidSimulation: momentum_resolve.comp compile failed")
		var mrp := PackedByteArray(); mrp.resize(44); mrp.fill(0)
		_momentum_resolve_params_buf = _compute.buffer_upload(mrp)
		# _punch_demand_accum_buf sized lazily in step() once the voxel buffer's real
		# capacity is known (mirrors _density_touch_buf's own growth pattern above).

	var runtime := get_parent()
	if runtime.has_method("register_simulation"):
		runtime.register_simulation(self)

	Performance.add_custom_monitor("Voxel/fluid_active_chunks", func(): return _active_chunks.size())


func _exit_tree() -> void:
	if Performance.has_custom_monitor("Voxel/fluid_active_chunks"):
		Performance.remove_custom_monitor("Voxel/fluid_active_chunks")
	for rid in [_flux_params_buf, _flux_chunk_list_buf, _dirty_chunks_buf, _flux_buf, _flux_page_table_buf, _conservation_buf, _gain_tracker_buf, _phi_buf_a, _phi_buf_b, _density_touch_buf, _momentum_claim_buf, _fill_history_buf,
			_punch_demand_queue_buf, _punch_demand_accum_buf, _punch_touched_cells_buf, _punch_demand_requeue_buf, _momentum_resolve_params_buf]:
		if rid.is_valid():
			_compute.free_rid(rid)


func get_simulation_class() -> int:
	return MaterialDef.SimulationClass.LIQUID


func add_candidates(chunk_positions: Array[Vector3i]) -> void:
	_ensure_flux_page_table()
	for cp in chunk_positions:
		_active_chunks[cp] = 0
		if _flux_page_table_buf.is_valid():
			_allocate_flux_page(cp)


func has_work() -> bool:
	# Backlog counts as work (2026-07-17): without it, SimulationRuntime drops this sim
	# from its active list the moment the chunk set empties, and a stranded demand backlog
	# (mass in flight, invisible to world-fill stats) freezes forever. See _punch_backlog.
	return not _active_chunks.is_empty() or (use_punch and _punch_backlog > 0)


# Zero the punch demand queue + backlog (2026-07-17). Call when a test scene is rebuilt:
# the queue is a persistent GPU buffer that survives scene edits, so successive KEY_L
# rebuilds otherwise ACCUMULATE stranded demands from prior builds — and because each build
# can use a different world origin, an old entry's coordinates point into empty space in the
# new scene (observed: an entry at z=-23, outside a pool at z in [-8,8], reading as air).
# That pollution makes the queue dump un-diagnosable; clearing on rebuild keeps each test clean.
func clear_punch_queue() -> void:
	if not _punch_demand_queue_buf.is_valid():
		return
	var zero4 := PackedByteArray(); zero4.resize(4); zero4.fill(0)
	_compute.buffer_update(_punch_demand_queue_buf, 0, zero4)
	_punch_backlog = 0


# --- Exact snapshot of ALL persistent fluid state (2026-07-17) ------------------------
# WorldOrchestrator's voxel snapshot (KEY_DOWN/KEY_UP) only captures voxels_a — but the
# flux model keeps PERSISTENT per-tick state the old comment predates: the flux field
# itself (never zeroed between ticks — that's the whole model), the punch demand queue
# (in-flight displaced mass), the fill-history counters, and the CPU-side paging/active/
# frame bookkeeping. Restoring voxels alone leaves all of that stale, so a mid-sim
# save→restore→replay diverges. These two methods capture and re-upload that state
# EXACTLY, so a restore reproduces the identical trajectory. world.gd calls them right
# after the orchestrator's own save/load (restore MUST run after the orchestrator's load,
# which emits chunks_committed and perturbs _active_chunks/paging — we overwrite it here).
# NOTE: untested in-editor at time of writing (no local Godot); verify a mid-plunge
# save→run→restore→run reproduces bit-identical F6/PunchQueue output.
var _debug_fluid_snapshot: Dictionary = {}

func debug_save_snapshot() -> void:
	if not _flux_buf.is_valid():
		return
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var span := _orchestrator.get_grid_span()
	var span_total: int = span.x * span.y * span.z
	_debug_fluid_snapshot = {
		"flux_buf": _compute.download_range(_flux_buf, 0, _flux_page_capacity * vpc * 3 * 4),
		"flux_page_capacity": _flux_page_capacity,
		"page_table": (_compute.download_range(_flux_page_table_buf, 0, span_total * 4)
			if (_flux_page_table_buf.is_valid() and span_total > 0) else PackedByteArray()),
		"queue": _compute.download_range(_punch_demand_queue_buf, 0, 4 + MAX_PUNCH_DEMAND_ENTRIES * 16),
		"fill_history": _compute.download_range(_fill_history_buf, 0, _fill_history_capacity * 4),
		"chunk_table": _flux_chunk_table.duplicate(true),
		"page_idle": _flux_page_idle.duplicate(true),
		"active_chunks": _active_chunks.duplicate(true),
		"punch_backlog": _punch_backlog,
		"frame": _frame,
	}
	print("[FluidSnapshot] saved: flux %d B, queue backlog=%d, active=%d, pages_used=%d" %
		[_debug_fluid_snapshot["flux_buf"].size(), _punch_backlog, _active_chunks.size(), _flux_chunk_table.size()])


func debug_restore_snapshot() -> void:
	if _debug_fluid_snapshot.is_empty():
		return
	var s := _debug_fluid_snapshot
	# Ensure the live buffers are at least as large as the saved payloads before uploading.
	_ensure_flux_buf_capacity(int(s["flux_page_capacity"]))
	_ensure_fill_history_capacity()
	_ensure_flux_page_table()
	var flux_bytes: PackedByteArray = s["flux_buf"]
	if _flux_buf.is_valid():
		_compute.buffer_update(_flux_buf, 0, flux_bytes)
	var pt: PackedByteArray = s["page_table"]
	if _flux_page_table_buf.is_valid() and not pt.is_empty():
		_compute.buffer_update(_flux_page_table_buf, 0, pt)
	_compute.buffer_update(_punch_demand_queue_buf, 0, s["queue"])
	var fh: PackedByteArray = s["fill_history"]
	if _fill_history_buf.is_valid() and fh.size() <= _fill_history_capacity * 4:
		_compute.buffer_update(_fill_history_buf, 0, fh)
	# Restore CPU bookkeeping. Rebuild the freelist from scratch (every page index not used
	# by the restored chunk_table) rather than restoring a saved copy — this stays correct
	# no matter what pages the orchestrator's chunks_committed discovery allocated between
	# the voxel restore and this call, so save/restore cycles can't leak or double-hand pages.
	_flux_chunk_table = s["chunk_table"].duplicate(true)
	_flux_page_idle = s["page_idle"].duplicate(true)
	_active_chunks = s["active_chunks"].duplicate(true)
	_punch_backlog = int(s["punch_backlog"])
	_frame = int(s["frame"])
	var used := {}
	for p in _flux_chunk_table.values():
		used[p] = true
	_flux_freelist = []
	for i in range(_flux_page_capacity - 1, -1, -1):
		if not used.has(i):
			_flux_freelist.append(i)
	print("[FluidSnapshot] restored: backlog=%d, active=%d, pages_used=%d" %
		[_punch_backlog, _active_chunks.size(), _flux_chunk_table.size()])


func get_active_chunk_positions() -> Array[Vector3i]:
	var out: Array[Vector3i] = []
	for cp in _active_chunks:
		out.append(cp)
	return out


# Debug tool (world.gd binds F7): downloads the privately-owned flux buffer for every
# allocated page and prints total |flux| + nonzero-lane count per chunk, plus a grand
# total, to console. On-demand only — full page-buffer readback, never call per-frame.
# Complements WorldOrchestrator.debug_dump_fluid_stats() (which shows fill, i.e. what's
# there): this shows flux, i.e. what's about to move — including on cells that no longer
# hold any fill, which is exactly the signature of the "stale flux survives voxel death"
# bug (a chunk reporting 0 liquid voxels but nonzero |flux| here means exactly that).
func debug_dump_flux_stats() -> void:
	if not _flux_buf.is_valid():
		print("[FluxDebug] flux buffer not ready")
		return
	var cs := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var lanes_per_page := vpc * 3
	var total_bytes := _flux_page_capacity * lanes_per_page * 4
	var data: PackedByteArray = _compute.download_range(_flux_buf, 0, total_bytes)

	var grand_abs := 0
	var grand_nonzero := 0
	print("[FluxDebug] === snapshot: %d page(s) allocated ===" % _flux_chunk_table.size())
	for cp in _flux_chunk_table:
		var page: int = _flux_chunk_table[cp]
		var base := page * lanes_per_page * 4
		var chunk_abs := 0
		var chunk_nonzero := 0
		for i in range(lanes_per_page):
			var v := data.decode_s32(base + i * 4)
			if v != 0:
				chunk_nonzero += 1
				chunk_abs += abs(v)
		if chunk_nonzero > 0:
			print("[FluxDebug]   chunk %s (page %d): total_|flux|=%d  nonzero_lanes=%d" % [cp, page, chunk_abs, chunk_nonzero])
		grand_abs += chunk_abs
		grand_nonzero += chunk_nonzero
	print("[FluxDebug] TOTAL |flux|=%d across %d nonzero lane(s)" % [grand_abs, grand_nonzero])


func step() -> void:
	if _flux_pipeline.is_empty():
		return
	if _active_chunks.is_empty():
		# Idle-rescue (2026-07-17, bench-validated in momentum_demand.py's step()): the sim
		# is asleep but demand mass is still queued. Run one rescue resolve cycle (walk with
		# the orphaned-origin down-probe enabled — see momentum_resolve.comp); any deposit
		# marks its chunk dirty, which re-activates it below, so a large backlog keeps the
		# sim awake and drains a bit every tick until the queue empties.
		if use_punch and _punch_backlog > 0 and _orchestrator.is_grid_valid():
			_rescue_punch_backlog()
		return
	if not _orchestrator.is_grid_valid():
		return

	# Safety net: add_candidates() normally allocates pages, but if it ran before the
	# grid was valid (e.g. very first tick), retry here before the dispatch needs them.
	_ensure_flux_page_table()
	_ensure_density_touch_capacity()
	_ensure_momentum_claim_capacity()
	_ensure_fill_history_capacity()
	_ensure_punch_demand_accum_capacity()
	for cp in _active_chunks:
		if _flux_page_table_buf.is_valid() and not _flux_chunk_table.has(cp):
			_allocate_flux_page(cp)

	var grid_min  := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()

	var active_list := _active_chunks.keys()

	# Flux dispatch covers active + all 6 face neighbours so cross-chunk flux reads/writes
	# (Scale's neighbour-owned-face lookups, Transport's changelog application) land within
	# a single dispatch.
	var flux_list := _build_flux_chunk_list(active_list)
	var flux_count := flux_list.size()

	# Phi dispatch range (2026-07-08) — the WIDER superset (see _build_phi_chunk_list's
	# header comment for why: the sleep scheduler is itself connector-blind, and this wider
	# range is the fix). phi_list ⊇ flux_list always, so every page-lifecycle step below
	# now keys off phi_list instead of flux_list — a chunk can legitimately be phi-only
	# (liquid-component member, no flux flow this tick) and its page must stay mapped for
	# Seed/FloodStep even though Accelerate/Transport skip it.
	var phi_list := _build_phi_chunk_list(flux_list)
	var phi_count := phi_list.size()

	# Every chunk in the PHI dispatch range needs a page, not just _active_chunks — Pass A
	# gates its entire per-axis loop on my_page (2026-07-02 fix made air cells compute their
	# own owned faces, but that computation is still gated on the CURRENT voxel's own page),
	# and PhiSeed/PhiFloodStep need pages across the whole liquid component, not just the
	# 1-hop flux halo. Positive-direction flow crossing a chunk boundary stores on the
	# DONOR's face (already paged, since the donor's chunk is active) and works fine either
	# way. Negative-direction flow — including straight down, gravity is -Y — stores on the
	# RECEIVER's own owned face: if that neighbour chunk has never held water, it has no
	# page, so Pass A returns immediately for every voxel there and nothing is ever
	# computed, let alone materialized — there's no changelog entry to trigger discovery,
	# so this doesn't self-heal like the positive-direction case does. Allocating pages for
	# the full phi range up front closes that hole (and, being a superset, the flux
	# neighbour set too).
	for cp in phi_list:
		if _flux_page_table_buf.is_valid() and not _flux_chunk_table.has(cp):
			_allocate_flux_page(cp)

	# Restore the GPU page-table entry for every dispatched chunk this tick — a previous
	# tick's exclusion loop below may have temporarily unmapped it (written -1) even
	# though the CPU-side page itself was never freed. Pass A's new neighbour-page check
	# (2026-07-02) depends on flux_page_table accurately reflecting THIS TICK's dispatch
	# membership, not just "ever had a page".
	for cp in phi_list:
		var restore_page: int = _flux_chunk_table.get(cp, -1)
		if restore_page >= 0:
			_write_flux_page_table_entry(cp, restore_page)

	# Zero flux data AND unmap the page table entry for any paged chunk NOT in this
	# tick's dispatch range (2026-07-02). Originally this only zeroed flux data, based on
	# the theory that stale-but-paged data was the leak — that fix did NOT resolve the
	# observed leak (confirmed by user testing). The GainTracker diagnostic subsequently
	# found the real mechanism: Pass A's per-axis loop reads a neighbour's REAL voxel
	# data via read_a() regardless of paging/dispatch status, and previously only checked
	# flux_page_table for "has a page" (true for a chunk still in its idle-guard grace
	# period, even though it isn't part of THIS tick's shader dispatch and therefore never
	# runs Transport to pay its side of any transfer). Un-mapping the page table entry
	# (not just zeroing flux data) makes flux_page_table accurately reflect "in this
	# tick's dispatch" — which Pass A's new neighbour-page check (fluid_flux.comp) now
	# relies on to refuse computing flux against an un-dispatched neighbour at all. The
	# CPU-side page allocation itself (_flux_chunk_table) is untouched — this is a
	# per-tick logical unmap, restored every tick a chunk re-enters phi_list (see the
	# restore loop above), not a free. Idle-guard eviction (actual page freeing) is
	# unaffected, still governed separately below.
	#
	# Range is phi_list now, not flux_list (2026-07-08) — a chunk that's phi-only (liquid
	# component member reached by the wider BFS, no flux flow this tick) must stay mapped
	# for Seed/FloodStep; unmapping it here the moment it falls outside the NARROWER
	# flux_list would silently zero its phi data every tick and defeat the scheduler fix
	# _build_phi_chunk_list exists for.
	var in_dispatch_range := {}
	for cp in phi_list:
		in_dispatch_range[cp] = true
	for cp in _flux_chunk_table.keys():
		if not in_dispatch_range.has(cp):
			var page: int = _flux_chunk_table[cp]
			var cs := _chunk_manager.chunk_size
			var vpc := cs * cs * cs
			var zero := PackedByteArray(); zero.resize(vpc * 3 * 4); zero.fill(0)
			_compute.buffer_update(_flux_buf, page * vpc * 3 * 4, zero)
			var phi_zero := PackedByteArray(); phi_zero.resize(vpc * 4); phi_zero.fill(0)
			_compute.buffer_update(_phi_buf_a, page * vpc * 4, phi_zero)
			_compute.buffer_update(_phi_buf_b, page * vpc * 4, phi_zero)
			_write_flux_page_table_entry(cp, -1)

	var merged_bitmask := PackedByteArray(); merged_bitmask.resize(DIRTY_BUF_BYTES); merged_bitmask.fill(0)
	var zero_dirty     := PackedByteArray(); zero_dirty.resize(DIRTY_BUF_BYTES); zero_dirty.fill(0)
	var zero4          := PackedByteArray(); zero4.resize(4); zero4.fill(0)
	var reset_gain     := PackedByteArray(); reset_gain.resize(8)
	reset_gain.encode_s32(0, -1); reset_gain.encode_s32(4, 0)

	# Pre-grow the chunk-list buffer to the largest list dispatched this step (phi_list ⊇
	# flux_list, so phi_count is the max) BEFORE building the uniform set — otherwise a
	# mid-step realloc invalidates binding 5 and silently skips a dispatch. See
	# _ensure_flux_chunk_list_capacity's comment for the full failure mode.
	_ensure_flux_chunk_list_capacity(max(flux_count, phi_count))

	var flux_uniforms := _build_flux_uniforms()

	for _s in range(steps_per_tick):
		_frame += 1

		# Phi-flood (Seed -> FloodStep x(K-1) -> FloodStepFinal, 2026-07-08) -> Pass A
		# (accelerate) -> Transport (fused conservation guard + apply, writes changelog).
		# Persistent: flux_buf is never zeroed between ticks (see fluid_flux.comp header).
		# Each dispatch below gets its own submit+sync — Transport reads flux_buf and
		# voxels_a that Accelerate already finished writing (a genuine cross-workgroup
		# dependency), Accelerate reads the phi buffer PhiFloodStepFinal just wrote, and so
		# on down the chain (Phase 7e Step 9, 2026-07-06: the old Scale-Outflow/Scale-Inflow
		# passes are gone; Transport recomputes their scale factors inline per-cell instead
		# of reading them back from a buffer a prior pass wrote — see fluid_flux.comp's
		# compute_s_out()/compute_s_in()). Transport is the only pass that touches the
		# changelog, so it gets its own reset_changelog()/apply_changelog() pair immediately
		# around its dispatch. Guard: flux_count can be 0 if all active chunks were
		# unloaded since last step() (phi_count is then also 0, since phi_list ⊇ flux_list).
		var flux_bitmask := zero_dirty  # default: no flux changes this sub-step
		if flux_count > 0:
			# Reset the dirty bitmask BEFORE the phi pipeline now (2026-07-08) —
			# PhiFloodStepFinal (pass 4) also writes into it via the pressure-wave wake
			# check (fluid_flux.comp), and that write must survive until the bitmask is
			# downloaded after Transport below, not get wiped by a reset that used to run
			# only immediately before Transport's own dispatch.
			_compute.buffer_update(_dirty_chunks_buf, 0, zero_dirty)

			# Momentum punch (pass 6 claim-zero + pass 9 crater, 2026-07-15, see use_punch).
			# Runs FIRST in the substep, before phi-flood/Accelerate/Transport — mirrors the
			# bench's MomentumDemandFluidSim.step() ordering (punch+resolve, THEN the base
			# substep, THEN buoyancy on whatever's settled). Own reset/apply changelog cycle
			# (Transport hasn't run yet this substep, so nothing else holds the changelog).
			if use_punch:
				_upload_flux_chunk_list(flux_list)
				_upload_flux_params(grid_min, grid_span, _frame, 6)  # (re)zero shared claim
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()

				_orchestrator.reset_changelog()
				_upload_flux_params(grid_min, grid_span, _frame, 9)  # MomentumPunch
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()
				_orchestrator.apply_changelog()

				_resolve_punch_demands(grid_min, grid_span)

			# --- Phi-flood: re-derived from voxels_a every substep, no persisted field.
			# See docs/PHI_FLOOD_PORT_PLAN.md's "ship the safe version first" note: one
			# submit+sync per dispatch here, matching every existing multi-pass effort in
			# this file (including the pressure-projection branch's red-black solve loop);
			# batching into one submit via compute_list_add_barrier is a real, documented
			# Godot RD capability and the planned perf follow-up once this is verified
			# in-editor. ---
			_upload_flux_chunk_list(phi_list)
			_upload_flux_params(grid_min, grid_span, _frame, 2)  # PhiSeed (always writes phi_buf_a)
			_compute.run(_flux_pipeline, flux_uniforms, phi_count * 4, 4, 4)
			_compute.submit()
			_compute.sync()

			# latest_is_a tracks which physical buffer holds the most recently written (and
			# therefore next iteration's source) value — starts true because Seed always
			# writes phi_buf_a. No parity math in the shader; this is plain CPU bookkeeping
			# fed in via phi_read_buf each dispatch (0 = phi_buf_a, 1 = phi_buf_b, matching
			# fluid_flux.comp's phi_read()/phi_flood_candidate()).
			var latest_is_a := true
			var flood_iters: int = max(1, phi_flood_iterations)
			for _fi in range(flood_iters - 1):
				_upload_flux_params(grid_min, grid_span, _frame, 3, 0 if latest_is_a else 1)  # PhiFloodStep
				_compute.run(_flux_pipeline, flux_uniforms, phi_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()
				latest_is_a = not latest_is_a
			_upload_flux_params(grid_min, grid_span, _frame, 4, 0 if latest_is_a else 1)  # PhiFloodStepFinal (+ wake check)
			_compute.run(_flux_pipeline, flux_uniforms, phi_count * 4, 4, 4)
			_compute.submit()
			_compute.sync()
			latest_is_a = not latest_is_a
			var accel_phi_read_buf := 0 if latest_is_a else 1

			# --- Accelerate / Transport: same flux dispatch range as before ---
			_upload_flux_chunk_list(flux_list)
			_upload_flux_params(grid_min, grid_span, _frame, 0, accel_phi_read_buf)  # Accelerate
			_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
			_compute.submit()
			_compute.sync()

			_orchestrator.reset_changelog()
			_upload_flux_params(grid_min, grid_span, _frame, 1)  # Transport
			_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
			_compute.submit()
			_compute.sync()

			flux_bitmask = _compute.download_range(_dirty_chunks_buf, 0, DIRTY_BUF_BYTES)
			_orchestrator.apply_changelog()

			# Fill-history update (pass 10, 2026-07-16, see stuck_rest_ticks) — runs after
			# Transport's changelog is applied and before the buoyancy exchange consults the
			# counters, mirroring momentum_demand.py's step() ordering (base substep, THEN
			# _update_fill_history, THEN buoyancy). Writes only _fill_history_buf (each
			# invocation its own cell — no changelog, no dirty bits, no sync hazards). Skipped
			# entirely when buoyancy is off or the override is disabled: counters go stale,
			# but the content-diff self-resets any stale entry on the first dispatch after
			# re-enabling, so toggling at runtime is safe.
			if use_buoyancy and stuck_rest_ticks >= 0:
				_upload_flux_params(grid_min, grid_span, _frame, 10)  # FillHistoryUpdate
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()

			# Buoyancy exchange (pass 6 claim-zero + pass 8 exchange, 2026-07-14, see
			# use_buoyancy). Runs AFTER Transport each substep. Zeroes the shared claim
			# buffer (pass 6) fresh before its own dispatch; own reset/apply changelog
			# cycle; re-download the dirty bitmask so exchanged chunks commit.
			if use_buoyancy:
				_upload_flux_params(grid_min, grid_span, _frame, 6)  # (re)zero shared claim
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()

				_orchestrator.reset_changelog()
				_upload_flux_params(grid_min, grid_span, _frame, 8)  # BuoyancyExchange
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()
				_orchestrator.apply_changelog()

				flux_bitmask = _compute.download_range(_dirty_chunks_buf, 0, DIRTY_BUF_BYTES)


			# Density-trigger debug tint (pass 5, 2026-07-12) — see
			# debug_density_trigger_tint's declaration. Same flux_list/flux_count
			# dispatch range as Accelerate/Transport (still current — the loop hasn't
			# re-uploaded a different chunk list since). Read-only against voxels_a
			# (already synced by Transport's apply_changelog above); writes only its
			# own scratch buffer, never voxels_a/changelog, so it cannot affect
			# simulation results regardless of whether the toggle is on.
			if debug_density_trigger_tint:
				_upload_flux_params(grid_min, grid_span, _frame, 5)  # DensityTouchDebug
				_compute.run(_flux_pipeline, flux_uniforms, flux_count * 4, 4, 4)
				_compute.submit()
				_compute.sync()

			_debug_print_watch()

			# Automatic conservation check (2026-07-02) — GPU-side, no manual F6 diffing
			# needed. Transport atomicAdds every real (new_fill - old_fill) it applies into
			# _conservation_buf; if the simulation conserves mass perfectly, that sum is
			# exactly 0 for a tick fully contained within the dispatch range (every unit
			# lost by one voxel is gained by another). A nonzero total is a direct,
			# quantified leak/gain measurement — only printed when it actually happens, so
			# this stays silent during normal correct operation.
			#
			# Gated behind debug_conservation_check (default false, added 2026-07-03) —
			# the download_range() calls below are a GPU readback that forces a sync every
			# single sub-step regardless of whether anything gets printed, which measurably
			# hurt performance once the leak investigation was done and this was left
			# running unconditionally.
			#
			# BUG (found 2026-07-03): the reset below used to live INSIDE this gate too —
			# Transport's atomicAdd into _conservation_buf runs every tick regardless of
			# this toggle, so leaving the toggle off let the buffer silently accumulate,
			# unread and unreset, for however long it stayed off. The instant it was
			# flipped back on, the first read dumped the ENTIRE accumulated backlog as if
			# it were a single tick's loss (observed: a reported "45814 lost this tick"
			# that was actually the sum of many quiet ticks, not one real event). Fix: the
			# reset must always run every tick, independent of whether we're reading/
			# printing this tick's value — only the (expensive) read+print is gated.
			if debug_conservation_check:
				var cons: PackedByteArray = _compute.download_range(_conservation_buf, 0, 4)
				var cons_delta := cons.decode_s32(0)
				if cons_delta != 0:
					print("[Conservation] fill %s this tick: %d" % ["gained" if cons_delta > 0 else "lost", abs(cons_delta)])

					# Location tracker (2026-07-02) — the stale-page fix didn't resolve the leak,
					# so pinpoint WHERE the single largest gain happened instead of guessing
					# further. Decode the flat voxel index back into chunk position + local xyz.
					var gt: PackedByteArray = _compute.download_range(_gain_tracker_buf, 0, 8)
					var gain_flat := gt.decode_s32(0)
					var gain_amount := gt.decode_s32(4)
					if gain_flat >= 0:
						var cs := _chunk_manager.chunk_size
						var vpc := cs * cs * cs
						var slot := gain_flat / vpc
						var local := gain_flat % vpc
						var lz := local / (cs * cs)
						var ly := (local % (cs * cs)) / cs
						var lx := local % cs
						var chunk_pos: Variant = _orchestrator._slot_to_chunk.get(slot, null)
						print("[Conservation]   biggest single gain: %d at chunk %s local (%d,%d,%d)" % [gain_amount, chunk_pos, lx, ly, lz])
						# Auto-dump the flagged voxel itself (2026-07-03) — its location moves
						# every tick, so retargeting watch_chunk/watch_local by hand after
						# reading a previous tick's log line always lags a step behind the
						# actual leaking cell. This grabs it live, same tick it's reported.
						if chunk_pos != null:
							_debug_dump_voxel("GainVoxel", chunk_pos, Vector3i(lx, ly, lz))
			_compute.buffer_update(_conservation_buf, 0, zero4)
			_compute.buffer_update(_gain_tracker_buf, 0, reset_gain)

		# OR dirty bits across sub-steps.
		for i in range(DIRTY_BUF_BYTES / 4):
			var a := merged_bitmask.decode_u32(i * 4)
			var c := flux_bitmask.decode_u32(i * 4)
			merged_bitmask.encode_u32(i * 4, a | c)

	var changed: Array[Vector3i] = _decode_dirty_bitmask(merged_bitmask)
	if not changed.is_empty():
		for cp in changed:
			var chunk: Chunk = _chunk_manager.chunks.get(cp)
			if chunk:
				chunk.mark_simulation_active(MaterialDef.SimulationClass.LIQUID)
		_orchestrator.chunks_committed.emit(changed)

	var changed_set: Dictionary = {}
	for cp in changed:
		changed_set[cp] = true
	var still_active: Dictionary = {}
	for cp in _active_chunks:
		if changed_set.has(cp):
			still_active[cp] = 0
		else:
			var idle: int = _active_chunks[cp] + 1
			if idle < 4:
				still_active[cp] = idle
	for cp in changed:
		if not still_active.has(cp):
			still_active[cp] = 0

	# Free flux pages for chunks that just fell out of the active set — eviction rides
	# the same idle-guard counter that already governs GranularSimulation's chunk sleep.
	for cp in _active_chunks:
		if not still_active.has(cp):
			_free_flux_page(cp)

	_active_chunks = still_active

	# Evict pages allocated purely as flux-dispatch neighbours (chunk-boundary fix,
	# 2026-07-02) that never actually became active — without this, sparse paging's core
	# promise ("most loaded chunks never touch fluid and pay zero flux storage") breaks
	# permanently for every chunk adjacent to any water body. A few ticks of grace before
	# freeing: a chunk that materialized its first voxel THIS tick won't show up in
	# _active_chunks until next tick's discovery runs, so freeing immediately would yank
	# its page out from under it mid-flow.
	var still_page_idle: Dictionary = {}
	for cp in _flux_chunk_table.keys():
		if _active_chunks.has(cp):
			continue  # active chunks keep their page via the eviction loop above
		var idle: int = _flux_page_idle.get(cp, 0) + 1
		if idle < 4:
			still_page_idle[cp] = idle
		else:
			_free_flux_page(cp)
	_flux_page_idle = still_page_idle


# --- Internal ---

func get_density_touch_buf() -> RID:
	return _density_touch_buf


# Grows _density_touch_buf to cover every currently-allocated voxel slot (see the field's
# declaration for the sizing rule and the exact bug this must not repeat). Called
# unconditionally every step() — like phi_buf_a/phi_buf_b, this stays allocated and bound
# regardless of debug_density_trigger_tint so the render uniform set's binding list never
# changes shape when the toggle flips; only the pass-5 dispatch itself is gated.
func _ensure_density_touch_capacity() -> void:
	var cs := _chunk_manager.chunk_size
	var needed := _orchestrator.get_voxel_buf_capacity() * cs * cs * cs
	if needed <= _density_touch_capacity and _density_touch_buf.is_valid():
		return
	if _density_touch_buf.is_valid():
		_compute.free_rid(_density_touch_buf)
	_density_touch_capacity = max(needed, 1)
	var buf := PackedByteArray(); buf.resize(_density_touch_capacity * 4); buf.fill(0)
	_density_touch_buf = _compute.buffer_upload(buf)


# Same sizing rule/growth pattern as _ensure_density_touch_capacity() above, for the
# shared CAS claim scratch. Called unconditionally every step() (like density_touch) so
# claim-based passes (buoyancy, and the punch once ported) can be toggled at runtime
# without a binding shape change.
func _ensure_momentum_claim_capacity() -> void:
	var cs := _chunk_manager.chunk_size
	var needed := _orchestrator.get_voxel_buf_capacity() * cs * cs * cs
	if needed <= _momentum_claim_capacity and _momentum_claim_buf.is_valid():
		return
	if _momentum_claim_buf.is_valid():
		_compute.free_rid(_momentum_claim_buf)
	_momentum_claim_capacity = max(needed, 1)
	var buf := PackedByteArray(); buf.resize(_momentum_claim_capacity * 4); buf.fill(0)
	_momentum_claim_buf = _compute.buffer_upload(buf)


# Same sizing rule/growth pattern, for the per-voxel fill history (see stuck_rest_ticks).
# Growth REALLOCATES and zeroes — existing counters are lost on growth, which is benign:
# a zeroed entry's prev-u16 (0 = air/fill-0) mismatches any real liquid, so every live
# cell just restarts its count from 0 on the next pass-10 dispatch (a one-time extra
# ~stuck_rest_ticks wait, only on the tick the world grew past the old capacity).
func _ensure_fill_history_capacity() -> void:
	var cs := _chunk_manager.chunk_size
	var needed := _orchestrator.get_voxel_buf_capacity() * cs * cs * cs
	if needed <= _fill_history_capacity and _fill_history_buf.is_valid():
		return
	if _fill_history_buf.is_valid():
		_compute.free_rid(_fill_history_buf)
	_fill_history_capacity = max(needed, 1)
	var buf := PackedByteArray(); buf.resize(_fill_history_capacity * 4); buf.fill(0)
	_fill_history_buf = _compute.buffer_upload(buf)


# Same sizing rule/growth pattern, for the punch demand-resolve accumulate scratch
# (momentum_resolve.comp pass 0's atomicAdd target) — indexed by full flat voxel index
# exactly like _momentum_claim_buf above.
func _ensure_punch_demand_accum_capacity() -> void:
	var cs := _chunk_manager.chunk_size
	var needed := _orchestrator.get_voxel_buf_capacity() * cs * cs * cs
	if needed <= _punch_demand_accum_capacity and _punch_demand_accum_buf.is_valid():
		return
	if _punch_demand_accum_buf.is_valid():
		_compute.free_rid(_punch_demand_accum_buf)
	_punch_demand_accum_capacity = max(needed, 1)
	var buf := PackedByteArray(); buf.resize(_punch_demand_accum_capacity * 4); buf.fill(0)
	_punch_demand_accum_buf = _compute.buffer_upload(buf)


# Debug readout (2026-07-17, the ~23% submerged-sphere oil "loss" investigation): prints
# the punch demand queue's current count and the total fill it holds per material. F6's
# fluid stats only count WORLD fill — mass sitting in unresolved/requeued demands is
# invisible to it (the bench's conservation checks add pending_demands for exactly this
# reason). A large standing queued_fill after the sim fully settles means the "missing"
# mass is STRANDED backlog (walk failures / resolution starvation — demands requeueing
# until the scheduler slept), not destroyed; an empty queue means a real destruction path.
# On-demand only (F6): downloads up to count*16 bytes.
func debug_dump_punch_queue() -> void:
	if not _punch_demand_queue_buf.is_valid():
		print("[PunchQueue] buffer not ready")
		return
	var cnt_bytes: PackedByteArray = _compute.download_range(_punch_demand_queue_buf, 0, 4)
	var cnt := cnt_bytes.decode_u32(0)
	var stored: int = min(cnt, MAX_PUNCH_DEMAND_ENTRIES)
	var total_by_mat := {}
	if stored > 0:
		var data: PackedByteArray = _compute.download_range(_punch_demand_queue_buf, 4, stored * 16)
		for i in range(stored):
			var packed := data.decode_s32(i * 16 + 12)
			var mat := packed & 0xFF
			var amt := packed >> 8
			total_by_mat[mat] = total_by_mat.get(mat, 0) + amt
	if cnt == 0:
		print("[PunchQueue] count=0 — queue empty, nothing in flight")
		return
	print("[PunchQueue] count=%d (stored=%d, dropped_over_capacity=%d)" % [cnt, stored, cnt - stored])
	for mat in total_by_mat:
		print("[PunchQueue]   mat=%d queued_fill=%d (%.1f voxels' worth)" % [mat, total_by_mat[mat], total_by_mat[mat] / 255.0])
	# Print the first few entries' origins — a stranded demand is diagnosable by aiming
	# F7 at its origin: an origin that is now AIR (orphaned by Transport draining the
	# transient splash pile it sat on) explains a walk that finds zero seeds and requeues
	# forever without progress (see docs/BUG_LOG.md 2026-07-17).
	if stored > 0:
		var data2: PackedByteArray = _compute.download_range(_punch_demand_queue_buf, 4, min(stored, 8) * 16)
		for i in range(min(stored, 8)):
			var ox := data2.decode_s32(i * 16)
			var oy := data2.decode_s32(i * 16 + 4)
			var oz := data2.decode_s32(i * 16 + 8)
			var pk := data2.decode_s32(i * 16 + 12)
			# Read the origin cell's CURRENT material/fill (2026-07-17): the decisive datum
			# for WHY a demand is stuck. origin==AIR ⇒ orphaned (the idle-rescue down-probe
			# case). origin==the demand's own material, buried (has same-material all around,
			# no reachable dischargeable surface) ⇒ a WALK_CAP-reach problem, NOT an orphan —
			# the down-probe finds material one step down and doesn't help. See docs/BUG_LOG.md.
			var om := "?"
			var of := "?"
			var slot: int = _orchestrator._chunk_to_slot.get(_world_to_chunk_pos(Vector3i(ox, oy, oz)), -1)
			if slot >= 0:
				var cs := _chunk_manager.chunk_size
				var vpc := cs * cs * cs
				var lpx := ox - _world_to_chunk_pos(Vector3i(ox, oy, oz)).x * cs
				var lpy := oy - _world_to_chunk_pos(Vector3i(ox, oy, oz)).y * cs
				var lpz := oz - _world_to_chunk_pos(Vector3i(ox, oy, oz)).z * cs
				var bidx := (slot * vpc + lpz * cs * cs + lpy * cs + lpx) * Chunk.VOXEL_STRIDE
				var vd: PackedByteArray = _compute.download_range(_orchestrator.get_voxel_buf(), bidx, 4)
				om = str(vd.decode_u8(0)); of = str(vd.decode_u8(1))
			print("[PunchQueue]   entry %d: origin=(%d, %d, %d) mat=%d amt=%d | origin_cell now: mat=%s fill=%s" % [i, ox, oy, oz, pk & 0xFF, pk >> 8, om, of])


# floor-div chunk position for a world voxel (matches fluid_flux.comp world_to_chunk).
func _world_to_chunk_pos(gpos: Vector3i) -> Vector3i:
	var cs := _chunk_manager.chunk_size
	return Vector3i(floori(float(gpos.x) / cs), floori(float(gpos.y) / cs), floori(float(gpos.z) / cs))


# Drains whatever is currently in _punch_demand_queue_buf (populated by fluid_flux.comp
# pass 9's punch, dispatched immediately before this is called). Structural mirror of the
# density-stacking mechanism's own (deleted) _resolve_density_demands(): zero-download-free
# GPU-side buffer_copy for the carry-forward, fixed max-size dispatch guarded by each
# pass's own `if (idx >= count) return;` so the CPU never needs to know the real count.
const _PUNCH_MAX_DISPATCH_GROUPS := 4096  # ceil(MAX_PUNCH_DEMAND_ENTRIES / 64.0)

func _resolve_punch_demands(grid_min: Vector3i, grid_span: Vector3i, rescue: int = 0) -> void:
	var zero4 := PackedByteArray(); zero4.resize(4); zero4.fill(0)
	_compute.buffer_update(_punch_touched_cells_buf, 0, zero4)
	_compute.buffer_update(_punch_demand_requeue_buf, 0, zero4)

	var uniforms := _build_momentum_resolve_uniforms()

	# Pass 0: walk — bounded BFS, single best (min-y) candidate per demand entry, one
	# invocation per queue entry (momentum_resolve.comp local_size_x=64). `rescue` enables
	# the walk's orphaned-origin down-probe — only ever 1 from _rescue_punch_backlog().
	_upload_momentum_resolve_params(grid_min, grid_span, 0, rescue)
	_compute.run(_momentum_resolve_pipeline, uniforms, _PUNCH_MAX_DISPATCH_GROUPS, 1, 1)
	_compute.submit()
	_compute.sync()

	# Pass 1: apply — clamp to real headroom, write changelog, requeue excess. Reset first
	# so this substep's own already-applied punch writes (pass 9) aren't redundantly
	# re-applied.
	_orchestrator.reset_changelog()
	_upload_momentum_resolve_params(grid_min, grid_span, 1)
	_compute.run(_momentum_resolve_pipeline, uniforms, _PUNCH_MAX_DISPATCH_GROUPS, 1, 1)
	_compute.submit()
	_compute.sync()
	_orchestrator.apply_changelog()

	# Carry genuine excess into the NEXT substep's queue — pure GPU-side copy, matching
	# the same established pattern. Next substep's punch (pass 9) atomicAdds fresh entries
	# on top of whatever count this seeds.
	_compute.buffer_copy(_punch_demand_requeue_buf, _punch_demand_queue_buf, 0, 0, 4)
	_compute.buffer_copy(_punch_demand_requeue_buf, _punch_demand_queue_buf, 4, 4, MAX_PUNCH_DEMAND_ENTRIES * 16)

	# Track the standing backlog (2026-07-17, one 4-byte readback per cycle) — see
	# _punch_backlog's declaration for why this must be known CPU-side.
	var qc: PackedByteArray = _compute.download_range(_punch_demand_queue_buf, 0, 4)
	_punch_backlog = qc.decode_u32(0)


# Idle-rescue for a stranded demand backlog (2026-07-17, see _punch_backlog and step()'s
# idle path; bench twin: momentum_demand.py step()'s idle-rescue drain). Runs one resolve
# cycle with the walk's orphaned-origin down-probe enabled, then re-activates every chunk
# the deposits dirtied so normal simulation resumes and the backlog keeps draining. Safe by
# scheduling construction: this only runs when the active set is empty — nothing dirty for
# 4+ ticks, world settled. The bench rejected BOTH per-cell gates for this probe first
# (unconditional and fill-history-gated each re-opened the waterfall chimney identically at
# peak splash, climb_events=4) before validating this scheduler gate.
func _rescue_punch_backlog() -> void:
	var zero_dirty := PackedByteArray(); zero_dirty.resize(DIRTY_BUF_BYTES); zero_dirty.fill(0)
	_compute.buffer_update(_dirty_chunks_buf, 0, zero_dirty)

	var before := _punch_backlog
	_resolve_punch_demands(_orchestrator.get_grid_min(), _orchestrator.get_grid_span(), 1)

	var bitmask: PackedByteArray = _compute.download_range(_dirty_chunks_buf, 0, DIRTY_BUF_BYTES)
	var reactivated := 0
	for cp in _decode_dirty_bitmask(bitmask):
		_active_chunks[cp] = 0
		reactivated += 1
		if _flux_page_table_buf.is_valid() and not _flux_chunk_table.has(cp):
			_allocate_flux_page(cp)
	# One-shot progress trace (2026-07-17) — a rescue that fires but re-activates 0 chunks
	# and doesn't shrink the queue count is failing to place the demand (down-probe found
	# material immediately but the underlying min-Y walk still can't reach a dischargeable
	# surface within WALK_CAP — a BURIED origin, not the air-orphan the probe targets).
	# Printed only when it makes no progress, so a healthy draining rescue stays quiet.
	if reactivated == 0 and _punch_backlog >= before and _punch_backlog > 0:
		print("[Rescue] fired but made NO progress: backlog=%d (queue count unchanged, 0 chunks re-activated) — origin likely buried, not air-orphaned" % _punch_backlog)


# momentum_resolve.comp's own separate pipeline/uniform set (2026-07-15) — mirrors
# GranularSimulation._build_demand_resolve_uniforms() / the deleted density_resolve.comp's
# own exact binding layout (0=voxels_a, 1=chunk_table, 2=params, 3=queue, 4=accum,
# 5=touched_cells, 6=changelog, 7=requeue, 8=dirty_chunks).
func _build_momentum_resolve_uniforms() -> Array:
	var u := func(binding: int, rid: RID) -> RDUniform:
		var un := RDUniform.new()
		un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		un.binding = binding
		un.add_id(rid)
		return un
	return [
		u.call(0, _orchestrator.get_voxel_buf()),
		u.call(1, _orchestrator.get_chunk_table()),
		u.call(2, _momentum_resolve_params_buf),
		u.call(3, _punch_demand_queue_buf),
		u.call(4, _punch_demand_accum_buf),
		u.call(5, _punch_touched_cells_buf),
		u.call(6, _orchestrator.get_changelog_buf()),
		u.call(7, _punch_demand_requeue_buf),
		u.call(8, _dirty_chunks_buf),
	]


func _upload_momentum_resolve_params(grid_min: Vector3i, grid_span: Vector3i, pass_idx: int, rescue: int = 0) -> void:
	var cs := _chunk_manager.chunk_size
	var p := PackedByteArray(); p.resize(44)
	p.encode_s32(0,  cs)
	p.encode_s32(4,  grid_min.x)
	p.encode_s32(8,  grid_min.y)
	p.encode_s32(12, grid_min.z)
	p.encode_s32(16, grid_span.x)
	p.encode_s32(20, grid_span.y)
	p.encode_s32(24, grid_span.z)
	p.encode_s32(28, pass_idx)
	p.encode_s32(32, settle_threshold)
	# Orphaned-origin idle-rescue flag (offset 36, 2026-07-17) — read by the walk pass only.
	p.encode_s32(36, rescue)
	# Runtime walk cap (offset 40, 2026-07-17, see punch_walk_cap export) — read by the walk.
	p.encode_s32(40, clampi(punch_walk_cap, 1, 128))
	_compute.buffer_update(_momentum_resolve_params_buf, 0, p)


# Allocates the GPU-side page-table mirror, sized to grid_span (chunk_pos → page index,
# -1 = no page). grid_span is fixed once WorldOrchestrator._init_grid() runs and never
# grows afterward (only the voxel slot pool grows via buffer_grow) — same invariant
# WorldOrchestrator._chunk_table already relies on — so this never needs to grow either.
func _ensure_flux_page_table() -> void:
	if _flux_page_table_buf.is_valid():
		return
	if not _orchestrator.is_grid_valid():
		return
	var span := _orchestrator.get_grid_span()
	var total := span.x * span.y * span.z
	if total <= 0:
		return
	var data := PackedByteArray(); data.resize(total * 4); data.fill(0xFF)  # -1 as s32
	_flux_page_table_buf = _compute.buffer_upload(data)


# Grows/creates the page-indexed flux buffer so it can hold at least min_pages pages.
# Phase 7e Step 4: each page now holds 3 signed int lanes per voxel (fx, fy, fz), not 1 —
# stride 3, interleaved — since every liquid voxel owns 3 persistent flux faces.
func _ensure_flux_buf_capacity(min_pages: int) -> void:
	var cs  := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	if not _flux_buf.is_valid():
		var initial = max(FLUX_PAGE_INITIAL_CAPACITY, min_pages)
		var bytes = initial * vpc * 3 * 4
		var data := PackedByteArray(); data.resize(bytes); data.fill(0)
		_flux_buf = _compute.buffer_upload(data)
		# Phi buffers (2026-07-08) — same page count as _flux_buf, stride 1 not 3, grown/
		# freed in lockstep everywhere _flux_buf is (see _phi_buf_a's declaration comment).
		var phi_bytes = initial * vpc * 4
		var phi_data := PackedByteArray(); phi_data.resize(phi_bytes); phi_data.fill(0)
		_phi_buf_a = _compute.buffer_upload(phi_data)
		_phi_buf_b = _compute.buffer_upload(phi_data.duplicate())
		_flux_page_capacity = initial
		for i in range(initial - 1, -1, -1):
			_flux_freelist.append(i)
		return
	if min_pages > _flux_page_capacity:
		var old_bytes := _flux_page_capacity * vpc * 3 * 4
		var new_capacity := _flux_page_capacity * 2
		while new_capacity < min_pages:
			new_capacity *= 2
		var new_bytes := new_capacity * vpc * 3 * 4
		_flux_buf = _compute.buffer_grow(_flux_buf, old_bytes, new_bytes)
		var old_phi_bytes := _flux_page_capacity * vpc * 4
		var new_phi_bytes := new_capacity * vpc * 4
		_phi_buf_a = _compute.buffer_grow(_phi_buf_a, old_phi_bytes, new_phi_bytes)
		_phi_buf_b = _compute.buffer_grow(_phi_buf_b, old_phi_bytes, new_phi_bytes)
		for i in range(new_capacity - 1, _flux_page_capacity - 1, -1):
			_flux_freelist.append(i)
		_flux_page_capacity = new_capacity


# chunk_pos → grid-flat index, identical formula to WorldOrchestrator._chunk_flat_idx.
func _flux_table_idx(chunk_pos: Vector3i) -> int:
	var grid_min  := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()
	var l := chunk_pos - grid_min
	if l.x < 0 or l.x >= grid_span.x or l.y < 0 or l.y >= grid_span.y or l.z < 0 or l.z >= grid_span.z:
		return -1
	return l.x + l.y * grid_span.x + l.z * grid_span.x * grid_span.y


func _write_flux_page_table_entry(chunk_pos: Vector3i, page: int) -> void:
	if not _flux_page_table_buf.is_valid():
		return
	var idx := _flux_table_idx(chunk_pos)
	if idx < 0:
		return
	var b := PackedByteArray(); b.resize(4); b.encode_s32(0, page)
	_compute.buffer_update(_flux_page_table_buf, idx * 4, b)


func _allocate_flux_page(chunk_pos: Vector3i) -> void:
	if _flux_chunk_table.has(chunk_pos):
		return
	if _flux_freelist.is_empty():
		_ensure_flux_buf_capacity(_flux_page_capacity + 1)
	if _flux_freelist.is_empty():
		return  # buffer alloc failed (no compute device) — leave as zero-flux boundary
	var page: int = _flux_freelist.pop_back()
	_flux_chunk_table[chunk_pos] = page
	_write_flux_page_table_entry(chunk_pos, page)


func _free_flux_page(chunk_pos: Vector3i) -> void:
	if not _flux_chunk_table.has(chunk_pos):
		return
	var page: int = _flux_chunk_table[chunk_pos]
	_flux_chunk_table.erase(chunk_pos)
	_flux_freelist.append(page)
	_write_flux_page_table_entry(chunk_pos, -1)
	# Zero the vacated page so a future occupant doesn't inherit stale momentum
	# (Step 6b — flux invalidation applies to page reuse as well as cell reuse).
	var cs  := _chunk_manager.chunk_size
	var vpc := cs * cs * cs
	var zero := PackedByteArray(); zero.resize(vpc * 3 * 4); zero.fill(0)
	_compute.buffer_update(_flux_buf, page * vpc * 3 * 4, zero)
	# Phi buffers ride the same page index (stride 1, not 3) — zero both so a future
	# occupant doesn't inherit stale phi (matters most for PhiSeed's very first iteration
	# read on a freshly-reused page, before this tick's own seed write lands).
	var phi_zero := PackedByteArray(); phi_zero.resize(vpc * 4); phi_zero.fill(0)
	_compute.buffer_update(_phi_buf_a, page * vpc * 4, phi_zero)
	_compute.buffer_update(_phi_buf_b, page * vpc * 4, phi_zero)


func _decode_dirty_bitmask(bitmask: PackedByteArray) -> Array[Vector3i]:
	var slot_to_chunk := _orchestrator._slot_to_chunk
	var out: Array[Vector3i] = []
	var words := bitmask.size() / 4
	for i in range(words):
		var word := bitmask.decode_u32(i * 4)
		if word == 0:
			continue
		for bit in range(32):
			if word & (1 << bit):
				var slot := i * 32 + bit
				var cp: Variant = slot_to_chunk.get(slot, null)
				if cp != null:
					out.append(cp)
	return out


func _build_flux_uniforms() -> Array:
	var u := func(binding: int, rid: RID) -> RDUniform:
		var un := RDUniform.new()
		un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		un.binding = binding
		un.add_id(rid)
		return un
	return [
		u.call(0, _orchestrator.get_voxel_buf()),
		u.call(1, _orchestrator.get_chunk_table()),
		u.call(2, _flux_params_buf),
		u.call(3, _orchestrator.get_material_sim_buf()),
		u.call(4, _orchestrator.get_changelog_buf()),
		u.call(5, _flux_chunk_list_buf),   # active + 6 face neighbours
		u.call(6, _dirty_chunks_buf),
		u.call(7, _flux_buf),
		u.call(8, _flux_page_table_buf),
		u.call(9, _conservation_buf),
		u.call(10, _gain_tracker_buf),
		u.call(11, _phi_buf_a),  # phi-flood (2026-07-08) — always bound, even for passes
		u.call(12, _phi_buf_b),  # (e.g. Transport) that never touch it; same convention
								  # already used for conservation/gain_tracker above.
		u.call(13, _orchestrator.get_material_flow_buf()),  # per-material viscosity (2026-07-10)
		u.call(14, _density_touch_buf),  # density-trigger debug tint (2026-07-12, see
										  # debug_density_trigger_tint) — always bound,
										  # only pass 5 ever writes it.
		u.call(15, _momentum_claim_buf),  # shared CAS claim scratch (2026-07-14) — always
										  # bound, used by claim-based resolution passes
										  # (pass 6 zeroes it, pass 8 buoyancy claims it).
		u.call(16, _punch_demand_queue_buf),  # write-only from pass 9 (MomentumPunch);
											   # drained by momentum_resolve.comp's own
											   # separate pipeline/uniform set. Always
											   # bound so toggling use_punch never
											   # changes this uniform set's shape.
		u.call(21, _orchestrator.get_material_density_buf()),  # per-material density
												  # (2026-07-12) — generalizes the
												  # trigger away from hardcoded WATER/OIL.
		u.call(22, _fill_history_buf),  # per-voxel fill history (2026-07-16, see
										 # stuck_rest_ticks) — written by pass 10,
										 # read by buoy_at_rest(). Always bound.
	]


func _upload_flux_params(grid_min: Vector3i, grid_span: Vector3i, frame: int, pass_idx: int, phi_read_buf: int = 0) -> void:
	var cs := _chunk_manager.chunk_size
	var p := PackedByteArray(); p.resize(84)
	p.encode_s32(0,  cs)
	p.encode_s32(4,  grid_min.x)
	p.encode_s32(8,  grid_min.y)
	p.encode_s32(12, grid_min.z)
	p.encode_s32(16, grid_span.x)
	p.encode_s32(20, grid_span.y)
	p.encode_s32(24, grid_span.z)
	p.encode_s32(28, frame)
	p.encode_s32(32, pass_idx)
	p.encode_float(36, damp)
	p.encode_float(40, g_dt)
	p.encode_float(44, gravity_bias)
	p.encode_s32(48, settle_threshold)
	# Phi-flood (2026-07-08) — which buffer (0=phi_buf_a, 1=phi_buf_b) this dispatch reads
	# as source. Irrelevant for Transport (pass 1, never touches phi) and for PhiSeed
	# (pass 2, always writes phi_buf_a regardless) — harmless to upload a default there.
	p.encode_s32(52, phi_read_buf)
	# Viscosity (2026-07-09) — global flow scalar, only read by Pass A. See the export.
	p.encode_float(56, flow)
	# Debug tint mode (offset 60) — only read by pass 5. See the tint_mode export.
	p.encode_s32(60, tint_mode)
	# Buoyancy rest tolerance (offset 64) — read by pass 8 + pass-5 buoyancy tint.
	p.encode_s32(64, buoy_rest_flux)
	# Momentum punch knobs (offsets 68/72/76) — read by pass 5 mode 2 + the punch pass.
	p.encode_s32(68, punch_threshold)
	p.encode_s32(72, punch_scale)
	p.encode_s32(76, max_reach)
	# Fill-history rest override (offset 80) — read by buoy_at_rest() (pass 8 + pass-5
	# buoyancy tint); pass 10 maintains the backing buffer. See stuck_rest_ticks.
	p.encode_s32(80, stuck_rest_ticks)
	_compute.buffer_update(_flux_params_buf, 0, p)


func _build_flux_chunk_list(active_list: Array) -> Array:
	# Active chunks + all 6 face neighbours so cross-chunk flux reads/writes (Scale's
	# neighbour-owned-face lookups, Transport's changelog application) all land within a
	# single dispatch. Vertical neighbours were added when Pass A folded the Y axis into
	# flux (Phase 7e Step 4) — before that, only horizontal neighbours mattered.
	var seen := {}
	var out  := []
	var offsets: Array[Vector3i] = [
		Vector3i( 0,  0,  0),
		Vector3i( 1,  0,  0), Vector3i(-1,  0,  0),
		Vector3i( 0,  0,  1), Vector3i( 0,  0, -1),
		Vector3i( 0,  1,  0), Vector3i( 0, -1,  0),
	]
	for cp: Vector3i in active_list:
		for offset in offsets:
			var nb: Vector3i = cp + offset
			if not seen.has(nb) and _chunk_manager.chunks.has(nb):
				seen[nb] = true
				out.append(nb)
	return out


# Phi-flood dispatch range (2026-07-08) — deliberately WIDER than _build_flux_chunk_list's
# 1-hop halo. This is the fix for the second structural bug found porting phi-flood to the
# faithful Python replica: the sleep scheduler is itself connector-blind. On a long
# connector the discharge endpoint (the far arm's surface) can sit several chunks outside
# the 1-hop flux dispatch frontier while every cell INSIDE that frontier is already
# saturated — zero transfers occur there, nothing marks a chunk dirty, and the idle-guard
# puts the whole rig to sleep with the imbalance still intact (reproduced in the Python
# port: arm A/B left permanently lopsided, `active=0`). Fix: flood the FULL connected
# liquid component (BFS via face-adjacency through chunks that currently hold liquid,
# `Chunk.has_simulation(SimulationClass.LIQUID)` — Chunks/chunk_instance.gd:21, a bitmask
# already ref-counted on every voxel mutation, no new bookkeeping needed) starting from
# flux_list, not just flux_list's own 1-hop halo. The flood is READ-ONLY on voxels_a, so
# covering chunks outside the flux/Transport dispatch range is safe — they simply don't get
# an Accelerate/Transport dispatch this tick, only Seed/FloodStep. Capped at
# PHI_FLOOD_MAX_CHUNKS so a single giant lake can't blow up one tick's dispatch cost: over
# the cap, the flood just stops expanding for that tick (degraded convergence at the
# frontier, never a hang) and picks back up next tick as the BFS re-runs from scratch.
const PHI_FLOOD_MAX_CHUNKS := 4096

func _build_phi_chunk_list(flux_list: Array) -> Array:
	var seen := {}
	var out  := []
	var offsets: Array[Vector3i] = [
		Vector3i( 1,  0,  0), Vector3i(-1,  0,  0),
		Vector3i( 0,  0,  1), Vector3i( 0,  0, -1),
		Vector3i( 0,  1,  0), Vector3i( 0, -1,  0),
	]
	# Seed the BFS frontier with flux_list itself (guarantees phi_list ⊇ flux_list even for
	# an active chunk whose own liquid content is momentarily nil, e.g. this tick's edit) —
	# Accelerate/Transport need valid phi data everywhere they read voxels, and only expand
	# past a chunk if IT holds liquid (a chunk with none is a valid stopping point: nothing
	# on the far side of it needs phi awareness through a non-liquid wall).
	var frontier: Array = []
	for cp: Vector3i in flux_list:
		if not seen.has(cp):
			seen[cp] = true
			out.append(cp)
			frontier.append(cp)
	while not frontier.is_empty() and out.size() < PHI_FLOOD_MAX_CHUNKS:
		var next_frontier: Array = []
		for cp: Vector3i in frontier:
			var chunk: Chunk = _chunk_manager.chunks.get(cp)
			if chunk == null or not chunk.has_simulation(MaterialDef.SimulationClass.LIQUID):
				continue  # a chunk with no liquid is a valid stopping point for the BFS
			for offset in offsets:
				var nb: Vector3i = cp + offset
				if seen.has(nb) or not _chunk_manager.chunks.has(nb):
					continue
				seen[nb] = true
				out.append(nb)
				next_frontier.append(nb)
				if out.size() >= PHI_FLOOD_MAX_CHUNKS:
					break
			if out.size() >= PHI_FLOOD_MAX_CHUNKS:
				break
		frontier = next_frontier
	return out


# Grow _flux_chunk_list_buf to hold `count` chunks, reallocating (new RID) if needed.
# MUST be called for the largest list this step BEFORE _build_flux_uniforms() (2026-07-17
# bugfix): compute_helper.run() builds its uniform set from the RIDs in the uniforms array
# at dispatch time, so a realloc between building flux_uniforms and a run() leaves binding 5
# (this buffer) pointing at a freed RID — uniform_set_create then returns invalid and the
# WHOLE dispatch is silently SKIPPED ("Storage buffer supplied (binding: 5) is invalid").
# That skips Phi/Accelerate/Transport for that tick, diverging the sim run-to-run — the
# source of the non-deterministic "walking origin" and unreliable DebugSnapshot. Pre-growing
# up front (max of flux_count/phi_count) guarantees no realloc happens after the uniforms
# are built, so the cached RIDs stay valid for every dispatch in the step.
func _ensure_flux_chunk_list_capacity(count: int) -> void:
	if count <= _flux_chunk_list_capacity and _flux_chunk_list_buf.is_valid():
		return
	if _flux_chunk_list_buf.is_valid():
		_compute.free_rid(_flux_chunk_list_buf)
	_flux_chunk_list_capacity = max(count, _flux_chunk_list_capacity * 2)
	var blank := PackedByteArray(); blank.resize(_flux_chunk_list_capacity * 12); blank.fill(0)
	_flux_chunk_list_buf = _compute.buffer_upload(blank)


func _upload_flux_chunk_list(chunk_positions: Array) -> void:
	var count := chunk_positions.size()
	_ensure_flux_chunk_list_capacity(count)
	var data := PackedByteArray(); data.resize(count * 12)
	for i in count:
		var cp: Vector3i = chunk_positions[i]
		data.encode_s32(i * 12,     cp.x)
		data.encode_s32(i * 12 + 4, cp.y)
		data.encode_s32(i * 12 + 8, cp.z)
	_compute.buffer_update(_flux_chunk_list_buf, 0, data)


func _load_flux_shader() -> String:
	var f := FileAccess.open("res://shaders/fluid_flux.comp", FileAccess.READ)
	if f: return f.get_as_text()
	push_error("FluidSimulation: failed to load fluid_flux.comp")
	return ""


func _load_momentum_resolve_shader() -> String:
	var f := FileAccess.open("res://shaders/momentum_resolve.comp", FileAccess.READ)
	if f: return f.get_as_text()
	push_error("FluidSimulation: failed to load momentum_resolve.comp")
	return ""
