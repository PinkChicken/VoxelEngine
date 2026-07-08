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
var _flux_params_buf: RID             # 56-byte params for fluid_flux.comp (incl. pass, damp, g_dt, gravity_bias, settle_threshold, phi_read_buf)
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

const DIRTY_BUF_BYTES := 64


func _ready() -> void:
	_orchestrator  = get_node("/root/world/WorldOrchestrator")
	_chunk_manager = get_node("/root/world/ChunkManager")
	_compute = preload("res://Utils/compute_helper.gd").new()
	if _compute.has_device():
		_flux_pipeline = _compute.compile_shader(_load_flux_shader(), 4)
		if _flux_pipeline.is_empty():
			push_error("FluidSimulation: fluid_flux.comp compile failed")
		var fp := PackedByteArray(); fp.resize(56); fp.fill(0)  # damp, g_dt, gravity_bias, settle_threshold, phi_read_buf
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
	var runtime := get_parent()
	if runtime.has_method("register_simulation"):
		runtime.register_simulation(self)

	Performance.add_custom_monitor("Voxel/fluid_active_chunks", func(): return _active_chunks.size())


func _exit_tree() -> void:
	if Performance.has_custom_monitor("Voxel/fluid_active_chunks"):
		Performance.remove_custom_monitor("Voxel/fluid_active_chunks")
	for rid in [_flux_params_buf, _flux_chunk_list_buf, _dirty_chunks_buf, _flux_buf, _flux_page_table_buf, _conservation_buf, _gain_tracker_buf, _phi_buf_a, _phi_buf_b]:
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
	return not _active_chunks.is_empty()


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
	if _active_chunks.is_empty() or _flux_pipeline.is_empty():
		return
	if not _orchestrator.is_grid_valid():
		return

	# Safety net: add_candidates() normally allocates pages, but if it ran before the
	# grid was valid (e.g. very first tick), retry here before the dispatch needs them.
	_ensure_flux_page_table()
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
	]


func _upload_flux_params(grid_min: Vector3i, grid_span: Vector3i, frame: int, pass_idx: int, phi_read_buf: int = 0) -> void:
	var cs := _chunk_manager.chunk_size
	var p := PackedByteArray(); p.resize(56)
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


func _upload_flux_chunk_list(chunk_positions: Array) -> void:
	var count := chunk_positions.size()
	if count > _flux_chunk_list_capacity:
		if _flux_chunk_list_buf.is_valid():
			_compute.free_rid(_flux_chunk_list_buf)
		_flux_chunk_list_capacity = max(count, _flux_chunk_list_capacity * 2)
		var blank := PackedByteArray(); blank.resize(_flux_chunk_list_capacity * 12); blank.fill(0)
		_flux_chunk_list_buf = _compute.buffer_upload(blank)
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
