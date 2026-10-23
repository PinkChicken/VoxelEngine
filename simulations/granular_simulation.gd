class_name GranularSimulation extends VoxelSimulation

# GPU dispatcher — step() compiles to one compute dispatch per active chunk.
# Active chunks are tracked at chunk granularity (not voxel): add_candidates()
# converts world positions to chunk positions. Chunks exit the active set when
# they produce no changelog entries (all granular voxels stable). They re-enter
# when chunks_committed fires for a neighbour and discovery calls add_candidates.

var _compute                          # ComputeHelper
var _pipeline: Dictionary = {}
var _zero_pipeline: Dictionary = {}   # zero_slots.comp — clears active chunk slots in voxels_b
var _global_params_buf: RID           # 32-byte global params, uploaded once per step
var _chunk_list_buf: RID              # N×3 ints: active chunk positions (granular pass)
var _chunk_list_capacity: int = 0
var _zero_chunk_list_buf: RID         # active + below-neighbors (zero pass)
var _zero_chunk_list_capacity: int = 0
var _dirty_chunks_buf: RID           # bitmask: 1 bit per voxel-buffer slot, written by shader atomicOr
var _orchestrator: WorldOrchestrator
var _chunk_manager: ChunkManager
var _active_chunks: Dictionary = {}  # Vector3i → int (consecutive idle frames, 0 = active this frame)
var _frame: int = 0

# Sub-steps per simulation tick — higher = faster-falling sand, more GPU cost. Each
# sub-step is one checkerboard-parity dispatch (granular_sim.comp gates on
# `params.frame % 2`), so grains only get a chance to move on every OTHER sub-step;
# at 1 (the old fixed behaviour) a full tick only advances one parity, which reads as
# slow motion for a voxel size this small. Mirrors FluidSimulation.steps_per_tick.
@export var steps_per_tick: int = 1

# MLS-MPM migration (2026-08-12, brief §37) — false skips register_simulation() below, so
# SimulationRuntime never ticks this system, while the node itself stays alive in the scene
# tree (world.gd's many debug tools — U-tube/density/sphere test-rig builders, F6/F7/T/H/N/J
# — all reference $FluidSimulation/$GranularSimulation directly and would break if the nodes
# were removed from world.tscn instead). Reversible: flip back to true to restore CA.
@export var enabled: bool = true

# 64 bytes covers 512 slots (512 / 32 * 4) — exactly the full fixed world (8³ chunks).
# If buffer_grow ever pushes capacity past 512, _dirty_chunks_buf must also grow.
const DIRTY_BUF_BYTES := 64

# --- phi_demand mechanism (2026-07-10, docs/PHI_FILL_SAND_PLAN.md) ---
# OFF by default: existing try_displace_priority behaviour (whole-voxel lateral-search
# swap) is completely unaffected when this is false. Bench-validated in Python
# (tools/fluid_prototypes/python/phi_demand.py, phi_demand_parallel.py) before this
# GLSL port — see the plan doc for the full design and the parallel-execution
# validation. NOT yet verified in-editor; this is the first in-engine test of this
# mechanism, hence the toggle rather than an outright replacement.
@export var use_phi_demand_displacement: bool = false
# Same "correctable deficit" tolerance fluid_flux.comp's phi wake logic uses
# (FluidSimulation.settle_threshold). Duplicated here rather than reading
# FluidSimulation directly, to avoid a new cross-node dependency for v1 — should be
# kept in sync with FluidSimulation.settle_threshold if that's ever retuned.
@export var demand_settle_threshold: int = 1

var _demand_resolve_pipeline: Dictionary = {}
var _demand_mode_params_buf: RID       # {int enabled} — read by granular_sim.comp binding 9
var _demand_queue_buf: RID             # {count, entries[]} — written by granular_sim.comp binding 10
var _demand_resolve_params_buf: RID    # demand_resolve.comp's own Params struct
var _demand_accum_buf: RID             # per-voxel-slot uint scratch, sized to voxel buf capacity
var _demand_accum_capacity: int = 0    # slots currently backing _demand_accum_buf
var _touched_cells_buf: RID            # {count, entries[]}
var _demand_requeue_buf: RID           # {count, entries[]} — carried into next substep's queue
const MAX_DEMAND_ENTRIES := 65536      # must match granular_sim.comp / demand_resolve.comp MAX_*
var _pending_requeue: PackedByteArray = PackedByteArray()  # raw entries carried from prior substep
var _pending_requeue_count: int = 0


func _ready() -> void:
	_orchestrator  = get_node("/root/world/WorldOrchestrator")
	_chunk_manager = get_node("/root/world/ChunkManager")
	_compute = preload("res://Utils/compute_helper.gd").new()
	if _compute.has_device():
		_pipeline = _compute.compile_shader(_load_shader(), 4)
		if _pipeline.is_empty():
			push_error("GranularSimulation: shader compile failed")
		_zero_pipeline = _compute.compile_shader(_load_zero_shader(), 4)
		if _zero_pipeline.is_empty():
			push_error("GranularSimulation: zero_slots shader compile failed")
		var p := PackedByteArray(); p.resize(32); p.fill(0)
		_global_params_buf = _compute.buffer_upload(p)
		# Seed chunk lists; both grow as needed.
		_chunk_list_capacity = 64
		var cl := PackedByteArray(); cl.resize(_chunk_list_capacity * 12); cl.fill(0)
		_chunk_list_buf = _compute.buffer_upload(cl)
		_zero_chunk_list_capacity = 128  # active + below-neighbors
		var zl := PackedByteArray(); zl.resize(_zero_chunk_list_capacity * 12); zl.fill(0)
		_zero_chunk_list_buf = _compute.buffer_upload(zl)
		var db := PackedByteArray(); db.resize(DIRTY_BUF_BYTES); db.fill(0)
		_dirty_chunks_buf = _compute.buffer_upload(db)

		# --- phi_demand mechanism buffers (see the export comment above) ---
		_demand_resolve_pipeline = _compute.compile_shader(_load_demand_resolve_shader(), 4)
		if _demand_resolve_pipeline.is_empty():
			push_error("GranularSimulation: demand_resolve shader compile failed")
		var dmp := PackedByteArray(); dmp.resize(4)
		dmp.encode_s32(0, 1 if use_phi_demand_displacement else 0)
		_demand_mode_params_buf = _compute.buffer_upload(dmp)
		var dq := PackedByteArray(); dq.resize(4 + MAX_DEMAND_ENTRIES * 16); dq.fill(0)
		_demand_queue_buf = _compute.buffer_upload(dq)
		# 36 bytes: demand_resolve.comp's Params struct is 9 ints (chunk_size, grid_min
		# xyz, grid_span xyz, pass, settle_threshold), NOT the 8-int/32-byte shape
		# granular_sim.comp's own SimParams uses -- do not "fix" this to match that one,
		# they are deliberately different buffers (see the binding-9 comment above).
		var drp := PackedByteArray(); drp.resize(36); drp.fill(0)
		_demand_resolve_params_buf = _compute.buffer_upload(drp)
		var tc := PackedByteArray(); tc.resize(4 + MAX_DEMAND_ENTRIES * 12); tc.fill(0)
		_touched_cells_buf = _compute.buffer_upload(tc)
		var rq := PackedByteArray(); rq.resize(4 + MAX_DEMAND_ENTRIES * 16); rq.fill(0)
		_demand_requeue_buf = _compute.buffer_upload(rq)
		# _demand_accum_buf sized lazily in step() once the voxel buffer's real
		# capacity is known (mirrors FluidSimulation's own flux-buffer growth pattern).
	var runtime := get_parent()
	if enabled and runtime.has_method("register_simulation"):
		runtime.register_simulation(self)

	Performance.add_custom_monitor("Voxel/active_chunks", func(): return _active_chunks.size())


func _exit_tree() -> void:
	Performance.remove_custom_monitor("Voxel/active_chunks")
	if _global_params_buf.is_valid():
		_compute.free_rid(_global_params_buf)
	if _chunk_list_buf.is_valid():
		_compute.free_rid(_chunk_list_buf)
	if _zero_chunk_list_buf.is_valid():
		_compute.free_rid(_zero_chunk_list_buf)
	if _dirty_chunks_buf.is_valid():
		_compute.free_rid(_dirty_chunks_buf)
	if _demand_mode_params_buf.is_valid():
		_compute.free_rid(_demand_mode_params_buf)
	if _demand_queue_buf.is_valid():
		_compute.free_rid(_demand_queue_buf)
	if _demand_resolve_params_buf.is_valid():
		_compute.free_rid(_demand_resolve_params_buf)
	if _demand_accum_buf.is_valid():
		_compute.free_rid(_demand_accum_buf)
	if _touched_cells_buf.is_valid():
		_compute.free_rid(_touched_cells_buf)
	if _demand_requeue_buf.is_valid():
		_compute.free_rid(_demand_requeue_buf)


func get_simulation_class() -> int:
	return MaterialDef.SimulationClass.GRANULAR


func add_candidates(chunk_positions: Array[Vector3i]) -> void:
	for cp in chunk_positions:
		_active_chunks[cp] = 0  # reset idle counter whenever a chunk re-enters


func has_work() -> bool:
	return not _active_chunks.is_empty()


func get_active_chunk_positions() -> Array[Vector3i]:
	var out: Array[Vector3i] = []
	for cp in _active_chunks:
		out.append(cp)
	return out


func step() -> void:
	if _active_chunks.is_empty() or _pipeline.is_empty():
		return
	if not _orchestrator.is_grid_valid():
		return

	var grid_min  := _orchestrator.get_grid_min()
	var grid_span := _orchestrator.get_grid_span()

	# Dispatch range is fixed for the whole tick (re-derived next real step() call) —
	# same convention FluidSimulation uses for its flux/phi lists across sub-steps.
	var active_list := _active_chunks.keys()
	var active_count := active_list.size()
	_upload_chunk_list(active_list)

	# Zero active chunks AND their below-neighbors so grains can fall cross-chunk
	# via CAS without hitting stale B data in non-active chunks.
	var zero_list := _build_zero_list(active_list)
	_upload_zero_chunk_list(zero_list)
	var zero_uniforms := _build_zero_uniforms()
	var uniforms := _build_uniforms()

	var zero_dirty     := PackedByteArray(); zero_dirty.resize(DIRTY_BUF_BYTES); zero_dirty.fill(0)
	var merged_bitmask := PackedByteArray(); merged_bitmask.resize(DIRTY_BUF_BYTES); merged_bitmask.fill(0)

	# Always reflect the CURRENT toggle state, not just when true -- if this only ever
	# wrote 1 (inside an `if true` branch) and never 0, flipping the export off at
	# runtime after having it on would leave the GPU-side flag stuck at 1.
	var dmp := PackedByteArray(); dmp.resize(4)
	dmp.encode_s32(0, 1 if use_phi_demand_displacement else 0)
	_compute.buffer_update(_demand_mode_params_buf, 0, dmp)
	if use_phi_demand_displacement:
		_ensure_demand_accum_capacity()
	var demand_touched_chunks: Dictionary = {}  # Vector3i -> true, unioned into `changed` below

	for _s in range(steps_per_tick):
		_frame += 1

		# Reset dirty bitmask and changelog before this sub-step's dispatch.
		_compute.buffer_update(_dirty_chunks_buf, 0, zero_dirty)
		_orchestrator.reset_changelog()
		_upload_global_params(grid_min, grid_span, _frame)

		_compute.run(_zero_pipeline, zero_uniforms, zero_list.size() * 4, 4, 4)
		_compute.submit()
		_compute.sync()

		# Granular simulation — each grain atomically claims its destination via CAS.
		_compute.run(_pipeline, uniforms, active_count * 4, 4, 4)
		_compute.submit()
		_compute.sync()

		# Download dirty bitmask immediately post-sync — just a memcopy, no GPU stall.
		var bitmask: PackedByteArray = _compute.download_range(_dirty_chunks_buf, 0, DIRTY_BUF_BYTES)

		# Patch _voxel_buf_a on the GPU from the changelog (no CPU extraction).
		_orchestrator.apply_changelog()

		if use_phi_demand_displacement:
			_resolve_demands(grid_min, grid_span, demand_touched_chunks)

		# OR dirty bits across sub-steps — a chunk that only moved on ONE parity this
		# tick must still count as changed.
		for i in range(DIRTY_BUF_BYTES / 4):
			var a := merged_bitmask.decode_u32(i * 4)
			var c := bitmask.decode_u32(i * 4)
			merged_bitmask.encode_u32(i * 4, a | c)

	# Decode merged bitmask → dirty chunk positions.
	var changed: Array[Vector3i] = _decode_dirty_bitmask(merged_bitmask)
	# demand_resolve.comp doesn't share granular_sim.comp's dirty_chunks atomicOr
	# mechanism (it's a fresh, differently-shaped dispatch) — chunks it touched are
	# tracked separately in _resolve_demands() and unioned in here.
	for cp in demand_touched_chunks:
		if not changed.has(cp):
			changed.append(cp)
	if not changed.is_empty():
		# Stamp the simulation mask on every chunk sand moved into or out of.
		# apply_changelog only patches voxels_a (GPU) and never updates block_data or
		# _simulation_mask, so destination chunks would silently have mask=0 and be
		# skipped by the neighbor discovery path on future commits.
		for cp in changed:
			var chunk: Chunk = _chunk_manager.chunks.get(cp)
			if chunk:
				chunk.mark_simulation_active(MaterialDef.SimulationClass.GRANULAR)
		_orchestrator.chunks_committed.emit(changed)

	# Stability: a chunk must appear in zero dirty bitmask frames for FOUR consecutive
	# dispatch frames before it can sleep. Four frames covers both checkerboard parities
	# twice, giving boundary grains two full opportunities to find a free path.
	var changed_set: Dictionary = {}
	for cp in changed:
		changed_set[cp] = true
	var still_active: Dictionary = {}
	for cp in _active_chunks:
		if changed_set.has(cp):
			still_active[cp] = 0           # activity this frame: reset idle count
		else:
			var idle: int = _active_chunks[cp] + 1
			if idle < 4:
				still_active[cp] = idle    # keep alive until both parities checked twice
			# idle >= 4: confirmed stable — drop the chunk
	# Directly add every dirty chunk regardless of its CPU simulation mask.
	# apply_changelog() only patches the GPU buffer — it does not update block_data or
	# _simulation_mask. Chunks that received sand via the changelog would otherwise
	# never be discovered by SimulationRuntime (mask check returns 0). The dirty
	# bitmask IS the ground truth: any slot marked dirty had a granular grain move
	# into or out of it this tick, so it must be active next tick.
	for cp in changed:
		if not still_active.has(cp):
			still_active[cp] = 0
	_active_chunks = still_active


# --- Internal ---

func _build_uniforms() -> Array:
	var u := func(binding: int, rid: RID) -> RDUniform:
		var un := RDUniform.new()
		un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		un.binding = binding
		un.add_id(rid)
		return un
	return [
		u.call(0, _orchestrator.get_voxel_buf()),
		u.call(1, _orchestrator.get_voxel_buf_b()),
		u.call(2, _orchestrator.get_chunk_table()),
		u.call(3, _global_params_buf),
		u.call(4, _orchestrator.get_material_sim_buf()),
		u.call(5, _orchestrator.get_changelog_buf()),
		u.call(6, _chunk_list_buf),
		u.call(7, _dirty_chunks_buf),
		u.call(8, _orchestrator.get_sim_active_buf()),
		u.call(9, _demand_mode_params_buf),
		u.call(10, _demand_queue_buf),
	]


func _upload_global_params(grid_min: Vector3i, grid_span: Vector3i, frame: int) -> void:
	var cs := _chunk_manager.chunk_size
	var p := PackedByteArray(); p.resize(32)
	p.encode_s32(0,  cs)
	p.encode_s32(4,  grid_min.x)
	p.encode_s32(8,  grid_min.y)
	p.encode_s32(12, grid_min.z)
	p.encode_s32(16, grid_span.x)
	p.encode_s32(20, grid_span.y)
	p.encode_s32(24, grid_span.z)
	p.encode_s32(28, frame)
	_compute.buffer_update(_global_params_buf, 0, p)


func _upload_chunk_list(chunk_positions: Array) -> void:
	var count := chunk_positions.size()
	# Grow the GPU buffer if needed (doubles each time).
	if count > _chunk_list_capacity:
		if _chunk_list_buf.is_valid():
			_compute.free_rid(_chunk_list_buf)
		_chunk_list_capacity = max(count, _chunk_list_capacity * 2)
		var blank := PackedByteArray(); blank.resize(_chunk_list_capacity * 12); blank.fill(0)
		_chunk_list_buf = _compute.buffer_upload(blank)
	var data := PackedByteArray(); data.resize(count * 12)
	for i in count:
		var cp: Vector3i = chunk_positions[i]
		data.encode_s32(i * 12,     cp.x)
		data.encode_s32(i * 12 + 4, cp.y)
		data.encode_s32(i * 12 + 8, cp.z)
	_compute.buffer_update(_chunk_list_buf, 0, data)


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


func _build_zero_list(active_list: Array) -> Array:
	var seen := {}
	var out  := []
	# Same offsets as fluid: self + below + 4 diagonal-below + 4 horizontal.
	# Diagonal-below covers cross-chunk diagonal falls and water lateral displacement.
	# Horizontal covers water displacement at sand's own level.
	var offsets: Array[Vector3i] = [
		Vector3i( 0,  0,  0),
		Vector3i( 0, -1,  0),
		Vector3i( 1, -1,  0), Vector3i(-1, -1,  0),
		Vector3i( 0, -1,  1), Vector3i( 0, -1, -1),
		Vector3i( 1,  0,  0), Vector3i(-1,  0,  0),
		Vector3i( 0,  0,  1), Vector3i( 0,  0, -1),
	]
	for cp: Vector3i in active_list:
		for offset in offsets:
			var nb: Vector3i = cp + offset
			if not seen.has(nb) and _chunk_manager.chunks.has(nb):
				seen[nb] = true
				out.append(nb)
	return out


func _upload_zero_chunk_list(chunk_positions: Array) -> void:
	var count := chunk_positions.size()
	if count > _zero_chunk_list_capacity:
		if _zero_chunk_list_buf.is_valid():
			_compute.free_rid(_zero_chunk_list_buf)
		_zero_chunk_list_capacity = max(count, _zero_chunk_list_capacity * 2)
		var blank := PackedByteArray(); blank.resize(_zero_chunk_list_capacity * 12); blank.fill(0)
		_zero_chunk_list_buf = _compute.buffer_upload(blank)
	var data := PackedByteArray(); data.resize(count * 12)
	for i in count:
		var cp: Vector3i = chunk_positions[i]
		data.encode_s32(i * 12,     cp.x)
		data.encode_s32(i * 12 + 4, cp.y)
		data.encode_s32(i * 12 + 8, cp.z)
	_compute.buffer_update(_zero_chunk_list_buf, 0, data)


func _build_zero_uniforms() -> Array:
	var u := func(binding: int, rid: RID) -> RDUniform:
		var un := RDUniform.new()
		un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		un.binding = binding
		un.add_id(rid)
		return un
	return [
		u.call(1, _orchestrator.get_voxel_buf_b()),
		u.call(2, _orchestrator.get_chunk_table()),
		u.call(3, _global_params_buf),
		u.call(4, _orchestrator.get_sim_active_buf()),
		u.call(6, _zero_chunk_list_buf),
	]


func _load_shader() -> String:
	var f := FileAccess.open("res://shaders/granular_sim.comp", FileAccess.READ)
	if f: return f.get_as_text()
	push_error("GranularSimulation: failed to load granular_sim.comp")
	return ""


func _load_zero_shader() -> String:
	var f := FileAccess.open("res://shaders/zero_slots.comp", FileAccess.READ)
	if f: return f.get_as_text()
	push_error("GranularSimulation: failed to load zero_slots.comp")
	return ""


func _load_demand_resolve_shader() -> String:
	var f := FileAccess.open("res://shaders/demand_resolve.comp", FileAccess.READ)
	if f: return f.get_as_text()
	push_error("GranularSimulation: failed to load demand_resolve.comp")
	return ""


# --- phi_demand mechanism (2026-07-10, docs/PHI_FILL_SAND_PLAN.md) ---
# Not yet verified in-editor -- see the use_phi_demand_displacement export comment.

func _ensure_demand_accum_capacity() -> void:
	# FIX (re-confirmed 2026-07-12, docs/BUG_LOG.md "_ensure_demand_accum_capacity() sized
	# its scratch buffer to the chunk-slot count..."): get_voxel_buf_capacity() returns the
	# number of CHUNK SLOTS the voxel buffer is sized for, not the voxel count — but
	# demand_accum is indexed by the FULL FLAT VOXEL INDEX (slot * vpc + local), same as
	# voxels_a itself. Must multiply by chunk_size^3 or every accumulate() write beyond the
	# first ~512 voxels lands out of bounds and is silently discarded by the GPU (confirmed
	# via direct buffer probe in the original investigation — no crash, no validation error,
	# just demand_accum reading back 0 for any real flat index). This was written once
	# before, bench-verified, then reverted along with unrelated density-stacking work in
	# the same file — see the bug log entry for why it needs re-applying independently.
	var cs := _chunk_manager.chunk_size
	var needed := _orchestrator.get_voxel_buf_capacity() * cs * cs * cs
	if needed <= _demand_accum_capacity and _demand_accum_buf.is_valid():
		return
	if _demand_accum_buf.is_valid():
		_compute.free_rid(_demand_accum_buf)
	_demand_accum_capacity = max(needed, 1)
	var buf := PackedByteArray(); buf.resize(_demand_accum_capacity * 4); buf.fill(0)
	_demand_accum_buf = _compute.buffer_upload(buf)
	# Freshly allocated -> already all-zero, matching demand_resolve.comp's own
	# self-clearing contract (pass_apply zeroes each touched slot after applying it,
	# so the buffer never needs a full re-zero on ordinary ticks, only on (re)growth).


func _build_demand_resolve_uniforms() -> Array:
	var u := func(binding: int, rid: RID) -> RDUniform:
		var un := RDUniform.new()
		un.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		un.binding = binding
		un.add_id(rid)
		return un
	return [
		u.call(0, _orchestrator.get_voxel_buf()),
		u.call(1, _orchestrator.get_chunk_table()),
		u.call(2, _demand_resolve_params_buf),
		u.call(3, _demand_queue_buf),
		u.call(4, _demand_accum_buf),
		u.call(5, _touched_cells_buf),
		u.call(6, _orchestrator.get_changelog_buf()),
		u.call(7, _demand_requeue_buf),
	]


func _upload_demand_resolve_params(grid_min: Vector3i, grid_span: Vector3i, pass_idx: int) -> void:
	var cs := _chunk_manager.chunk_size
	var p := PackedByteArray(); p.resize(36)
	p.encode_s32(0,  cs)
	p.encode_s32(4,  grid_min.x)
	p.encode_s32(8,  grid_min.y)
	p.encode_s32(12, grid_min.z)
	p.encode_s32(16, grid_span.x)
	p.encode_s32(20, grid_span.y)
	p.encode_s32(24, grid_span.z)
	p.encode_s32(28, pass_idx)
	p.encode_s32(32, demand_settle_threshold)
	_compute.buffer_update(_demand_resolve_params_buf, 0, p)


# Resolves this substep's demand queue (granular_sim.comp's try_displace_demand writes
# into it, bound at binding 10 of the granular dispatch above) via demand_resolve.comp's
# two passes, folding in any leftover requeued from the PREVIOUS substep first. Updates
# `touched_chunks` (Vector3i -> true) with every chunk any resolved cell belongs to, so
# the caller can fold it into this tick's `changed` set for chunks_committed.
func _resolve_demands(grid_min: Vector3i, grid_span: Vector3i, touched_chunks: Dictionary) -> void:
	# Pre-seed this substep's queue with last substep's leftover (if any), THEN let
	# granular_sim.comp's own writes (already dispatched above, same substep) sit on
	# top -- both are already present in _demand_queue_buf's entries by this point;
	# only the COUNT needs bumping to include the pre-seeded entries, since granular's
	# atomicAdd(demand_queue.count, ...) started counting from whatever was there
	# before its own dispatch. To keep this simple and avoid a read-modify-write race
	# with the shader's own atomicAdd, the requeue is folded in BEFORE granular's next
	# dispatch instead (see the end of this function) -- this call only drains and
	# resolves whatever is in the buffer right now.
	var count_buf: PackedByteArray = _compute.download_range(_demand_queue_buf, 0, 4)
	var count: int = count_buf.decode_u32(0)
	if count <= 0:
		_pending_requeue = PackedByteArray()
		_pending_requeue_count = 0
		return
	if count > MAX_DEMAND_ENTRIES:
		push_warning("GranularSimulation: demand queue overflow (%d); clamping to %d" % [count, MAX_DEMAND_ENTRIES])
		count = MAX_DEMAND_ENTRIES

	var zero4 := PackedByteArray(); zero4.resize(4); zero4.fill(0)
	_compute.buffer_update(_touched_cells_buf, 0, zero4)
	_compute.buffer_update(_demand_requeue_buf, 0, zero4)

	var uniforms := _build_demand_resolve_uniforms()

	# Pass 0: walk + accumulate (one invocation per demand entry).
	_upload_demand_resolve_params(grid_min, grid_span, 0)
	_compute.run(_demand_resolve_pipeline, uniforms, int(ceil(float(count) / 64.0)), 1, 1)
	_compute.submit()
	_compute.sync()

	var touched_count_buf: PackedByteArray = _compute.download_range(_touched_cells_buf, 0, 4)
	var touched_count: int = touched_count_buf.decode_u32(0)
	if touched_count <= 0:
		_pending_requeue = PackedByteArray()
		_pending_requeue_count = 0
		return
	if touched_count > MAX_DEMAND_ENTRIES:
		touched_count = MAX_DEMAND_ENTRIES

	# Pass 1 writes fresh changelog entries -- reset first so apply_changelog() below
	# doesn't also re-apply this substep's already-applied granular entries (harmless
	# if it did, since they're idempotent same-value writes, but wasteful).
	_orchestrator.reset_changelog()
	_upload_demand_resolve_params(grid_min, grid_span, 1)
	_compute.run(_demand_resolve_pipeline, uniforms, int(ceil(float(touched_count) / 64.0)), 1, 1)
	_compute.submit()
	_compute.sync()
	_orchestrator.apply_changelog()

	# Track touched chunks for chunks_committed (demand_resolve doesn't share
	# granular_sim.comp's dirty_chunks atomicOr mechanism -- see the call site).
	var cs := _chunk_manager.chunk_size
	var touched_data: PackedByteArray = _compute.download_range(_touched_cells_buf, 4, touched_count * 12)
	for i in touched_count:
		var x := touched_data.decode_s32(i * 12)
		var y := touched_data.decode_s32(i * 12 + 4)
		var z := touched_data.decode_s32(i * 12 + 8)
		var cp := Vector3i(floori(float(x) / cs), floori(float(y) / cs), floori(float(z) / cs))
		touched_chunks[cp] = true

	# Carry genuine excess into the NEXT substep's queue -- never silently dropped
	# (see demand_resolve.comp's file header: an earlier version that clamped and
	# discarded here measured real mass loss under real concurrent contention).
	var requeue_count_buf: PackedByteArray = _compute.download_range(_demand_requeue_buf, 0, 4)
	var requeue_count: int = requeue_count_buf.decode_u32(0)
	if requeue_count > 0:
		if requeue_count > MAX_DEMAND_ENTRIES:
			requeue_count = MAX_DEMAND_ENTRIES
		var requeue_data: PackedByteArray = _compute.download_range(_demand_requeue_buf, 4, requeue_count * 16)
		# Pre-seed _demand_queue_buf for the NEXT call to this function (next substep,
		# or next tick if this was the last substep): write the requeued entries at
		# the start of the entries array and set count to match. Any demands
		# granular_sim.comp's NEXT dispatch queues will atomicAdd on top via its own
		# count-allocated slots, which start at this seeded count, not 0.
		_compute.buffer_update(_demand_queue_buf, 4, requeue_data)
		var cnt := PackedByteArray(); cnt.resize(4)
		cnt.encode_u32(0, requeue_count)
		_compute.buffer_update(_demand_queue_buf, 0, cnt)
		_pending_requeue_count = requeue_count
	else:
		# Nothing carried forward -- zero the queue count so the NEXT dispatch starts
		# counting fresh from 0 (otherwise this substep's already-resolved entries
		# would be re-read next time, double-resolving them).
		_compute.buffer_update(_demand_queue_buf, 0, zero4)
		_pending_requeue_count = 0
