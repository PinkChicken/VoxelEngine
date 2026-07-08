"""
phi_flood.py — PhiFloodFluidSim: connected-vessel fix via a piezometric
pressure flood (docs/POSSIBLE_SOLUTIONS.md option C, with two additions taken
from option D and the Chorin bench's density-correction lesson).

THE FIELD.  For every liquid voxel p a pressure P(p) (fill units, same scale
as head()) is re-derived from the CURRENT fill field every tick:

    seeds:  every free-surface cell (fill<255 OR the cell above is not the
            same liquid) seeds P = fill  — exactly head()'s value there.
    flood:  P(p) = max( seed(p), best over 6 same-liquid neighbours q ),
            with deficit(p) = FILL_CAP - fill(p):
            q ABOVE p (down-step):   P(q) + fill(p)  - deficit(p)
            q BELOW p (up-step):     P(q) - FILL_CAP - deficit(p)
            q BESIDE p (horizontal): P(q) - EPS_H    - deficit(p)

Own-column paths reproduce head() EXACTLY for settled bodies (seed=fill at the
surface, +fill per full cell going down == head()'s column sum; the deficit
terms vanish on full cells), so P == head() for any isolated body at rest and
the corrected accel is bit-identical to the base system there — the fix only
activates where a LATERAL path carries more pressure than the local column
(the connected-vessel case) or where a submerged cell is deficient.

Two wake-loop bugs were found and fixed against the initial cost model (see
inline comments in _compute_phi): the down-step must ALSO subtract the
receiver's deficit (else a slightly-deficient submerged cell inherits the full
static ramp and can never be corrected), and the up-step must cost the full
FILL_CAP rather than fill(q) (else a chain of slightly-deficient cells
self-consistently inflates P into a permanently over-pressured, uncorrectable
state). Both leave full cells — i.e. all settled bodies — unaffected.

Why this dodges the two documented traps:

 * Build B / falsification-ladder #3 ("max-propagation flattens the driving
   gradient"): that field had zero horizontal cost, so it was uniform at each
   elevation and the connector gradient was identically 0.  EPS_H is a small
   per-hop hydraulic slope: the tall arm's ramp invades the connector DECAYING
   by EPS_H per step, so adjacent connector cells always differ by EPS_H and
   flux forms along the whole connector.  Steady-state equalization error is
   bounded by EPS_H * path_len / FILL_CAP cells (sub-voxel for sane rigs).

 * Thin-smear / staircase (ladder #2, projection outcome): the up-step and
   horizontal costs subtract the receiving cell's fill DEFICIT (FILL_CAP -
   fill).  A cell that has just donated upward reads lower P than its full
   neighbours, so refill flux forms behind it — the "bucket brigade" through a
   saturated column actually carries mass instead of the top cell draining
   into a smear.  This is the same physical term as the Chorin bench's
   load-bearing density-correction source, in flood form.

THE INJECTION.  extra_correction_pass() re-runs Pass A's per-face update with
dh = P(p) - P(nb) instead of head(p) - head(nb), starting from the SAME
pre-Accelerate flux value (snapshotted in _pass_accelerate), with identical
gating, decay, clamp and rounding.  Nothing downstream changes: eff_flux /
compute_s_out / compute_s_in / Transport untouched, so conservation is
inherited from the base system.

Reactivity is automatic: P is re-derived from voxels_a every tick, so a second
pour becomes new seeds/weight immediately — no persisted field to go stale
(the projection attempt's reactivity failure) and no air-layer domain
extension needed (poured water is liquid cells, hence in the flood domain).

KNOWN ISSUE (found 2026-07-08, in-engine testing — the "flat lake ripple"
bug). Placing one full voxel directly on top of an already-full flat lake
radiates a spurious pressure wave across the WHOLE lake in a single tick (a
wide "crater" ring), wildly out of proportion to one voxel's mass. Root
cause: the seed rule above ("fill<255 or the cell above isn't the same
liquid") conflates two physically different situations. A cell exposed to AIR
is the true local free surface — any real excess there just raises the water
level, never manifests as trapped pressure — while a cell sealed under ROCK
(e.g. the U-tube connector) is NOT a true surface, an interior point of the
connected body that legitimately needs its P inflated by a taller connected
column. The pile's genuinely higher head at the one true interface (its
submerged former-top cell vs. a flat neighbour's still-open top) keeps
"leaking" sideways through the flat neighbour's OWN valid seed value via
max() — hop after hop, each losing only EPS_H=4 — something stock head()
(which never looks sideways at all) could never do.

A fix ("protect true air-exposed surfaces from ever being inflated") was
built and REVERTED the same day: `test_pile_ripple.py` confirmed it contains
the ripple, but it also broke the core U-tube mechanism outright (equalize/
reactivity both back to FAIL) — see `diagnose_utube.py phi`'s connector scan,
which showed phi hard-stopping at the base of the still-empty target arm. The
reason is structural, not a tuning slip: that arm-base cell is ALSO "exposed
to air" (nothing has arrived there yet), so protecting it caps its dh at
exactly its own fill — precisely what stock head() already gives, which is
exactly what fails this test. A single cell's local state (fill, what's above
it) cannot distinguish "a coincidental neighbour already correctly at rest"
(the lake bug) from "a dry arm base that genuinely needs this exact inflation
to know it should rise" (the U-tube fix) — both are a cell exposed to air
receiving an inflating horizontal candidate from a same-elevation neighbour;
only information about what's happening FURTHER AWAY (a real connected
imbalance vs. none) distinguishes them, which is exactly the non-local
information the flood exists to carry. Left open — see docs/BUG_LOG.md and
docs/PHI_FLOOD_PORT_PLAN.md for the full write-up and candidate directions
(e.g. bounding disturbance MAGNITUDE/SPEED rather than blocking the mechanism
outright, so a single small edit can't outrun stock head()'s own
one-hop-per-tick diffusion while a genuine large connected-body imbalance
still gets through). `test_pile_ripple.py` is kept in this folder as the
reproduction/regression case for whoever picks this up.
"""

from typing import Dict, Optional, Tuple

from fluid_engine import (
    FluidSim, FluxParams, VoxelWorld,
    FILL_CAP, AXIS_DIR, SIM_LIQUID, gl_round, clamp,
)

# Horizontal hydraulic slope, fill-units per hop. Small enough that the
# steady-state error (EPS_H * path_len / 255 voxels) stays sub-voxel on any
# sane rig; large enough that connector accel (EPS_H * g_dt) integrates to a
# clearly super-MIN_FLOW persistent flux.
DEFAULT_EPS_H = 4
# Hard iteration ceiling for the flood (each iteration propagates one hop).
# The flood early-outs the moment a sweep changes nothing, so this is a
# safety cap, not a per-tick cost.
DEFAULT_MAX_ITERS = 256
# Pressure-wave wake margin (fill units): a REAL free-surface cell (air above
# or partially filled — sealed-under-solid cells never qualify, so legitimate
# static pressure under rock can't cause churn) whose flooded phi exceeds its
# own fill by more than this wakes its chunk. This is the fix for the
# discovered scheduler blindness (2026-07-08): on a long connector the
# discharge endpoint (the far arm's surface) can sit several chunks outside
# the dispatch frontier while everything INSIDE the frontier is saturated —
# zero transfers happen, nothing is marked dirty, and the idle-guard puts the
# rig to sleep with the imbalance intact. The flood therefore runs over the
# whole loaded liquid set (it is read-only on voxels_a, so sleeping chunks
# are safe to read) and over-pressured real surfaces are woken explicitly.
# At equilibrium the EPS_H slope makes the invading ramp UNDERSHOOT (the
# local seed phi=fill wins at balanced surfaces), so this signal is exactly
# zero at rest and sleep still happens.
DEFAULT_WAKE_MARGIN = 64


class PhiFloodFluidSim(FluidSim):
    def __init__(self, world: VoxelWorld, params: Optional[FluxParams] = None,
                 eps_h: int = DEFAULT_EPS_H, max_iters: int = DEFAULT_MAX_ITERS,
                 wake_margin: int = DEFAULT_WAKE_MARGIN):
        super().__init__(world, params)
        self.correction_enabled = True
        self.eps_h = eps_h
        self.max_iters = max_iters
        self.wake_margin = wake_margin
        self._pre_accel_flux = None
        self._flux_list_snapshot = []
        # Diagnostics
        self.last_phi: Dict[Tuple[int, int, int], int] = {}
        self.last_flood_iters = 0

    # Snapshot the pre-Accelerate flux so the correction can redo Pass A's
    # update (decay + accel + clamp + round) from the same starting value.
    def _pass_accelerate(self, flux_list, cs: int, vpc: int) -> None:
        self._pre_accel_flux = list(self.flux_buf)
        self._flux_list_snapshot = list(flux_list)
        super()._pass_accelerate(flux_list, cs, vpc)

    # ------------------------------------------------------------------
    # The flood.
    # ------------------------------------------------------------------

    def _compute_phi(self, flux_list):
        """Returns (cells, P): cells maps gpos -> (mat, fill) for every LOADED
        liquid voxel (not just this tick's dispatch range — the flood is
        read-only on voxels_a, and it must be able to reach a discharge
        endpoint that sleeps outside the dispatch frontier; see
        DEFAULT_WAKE_MARGIN); P maps gpos -> flooded pressure."""
        world = self.world
        cs = world.chunk_size
        mat_sim = world.material_sim

        cells: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
        for cp, slot in world.chunk_to_slot.items():
            arr = world.voxels[slot]
            base = (cp[0] * cs, cp[1] * cs, cp[2] * cs)
            for local, u in enumerate(arr):
                m = u & 0xFF
                if m == 0 or mat_sim[m] != SIM_LIQUID:
                    continue
                lx = local % cs
                ly = (local // cs) % cs
                lz = local // (cs * cs)
                cells[(base[0] + lx, base[1] + ly, base[2] + lz)] = (m, (u >> 8) & 0xFF)

        # Seeds: free-surface cells carry P = fill (== head() there).
        P: Dict[Tuple[int, int, int], int] = {}
        seed: Dict[Tuple[int, int, int], int] = {}
        for pos, (m, f) in cells.items():
            above = (pos[0], pos[1] + 1, pos[2])
            ac = cells.get(above)
            if f < FILL_CAP or ac is None or ac[0] != m:
                seed[pos] = f
                P[pos] = f
            else:
                P[pos] = 0  # valid lower bound: true P >= fill >= 0

        eps = self.eps_h
        iters = 0
        for _ in range(self.max_iters):
            iters += 1
            changed = False
            newP = {}
            for pos, (m, f) in cells.items():
                best = seed.get(pos, 0)
                deficit = FILL_CAP - f
                for axis in range(3):
                    d = AXIS_DIR[axis]
                    for sgn in (1, -1):
                        q = (pos[0] + sgn * d[0], pos[1] + sgn * d[1], pos[2] + sgn * d[2])
                        qc = cells.get(q)
                        if qc is None or qc[0] != m:
                            continue
                        pq = P[q]
                        # The receiving cell's fill DEFICIT is subtracted on
                        # EVERY step direction (down included — fixed
                        # 2026-07-08 after a locked wake-loop repro: without
                        # it, a slightly-deficient SUBMERGED cell inherits the
                        # full static ramp from above, every neighbouring face
                        # reads perfect equilibrium, and the deficit can never
                        # be corrected while the wake rule keeps firing on it).
                        # Full cells (deficit 0) are unaffected, so the
                        # own-column == head() identity still holds for
                        # settled bodies.
                        # Up-steps cost the full elevation quantum FILL_CAP,
                        # NOT fill(q) (fixed 2026-07-08, second wake-loop
                        # repro): -fill(q) under-charges each hop through a
                        # slightly-deficient cell, so a chain of them
                        # self-consistently inflates P — over-pressured by the
                        # wake rule yet with every face reading < 0.5 accel,
                        # i.e. an uncorrectable permanent wake-loop. For full
                        # cells fill(q) == FILL_CAP, so settled bodies are
                        # unaffected.
                        if axis == 1:
                            if sgn == 1:   # q above p: down-step
                                cand = pq + f - deficit
                            else:          # q below p: up-step
                                cand = pq - FILL_CAP - deficit
                        else:              # horizontal
                            cand = pq - eps - deficit
                        if cand > best:
                            best = cand
                if best != P[pos]:
                    changed = True
                newP[pos] = best
            P = newP
            if not changed:
                break

        self.last_flood_iters = iters
        self.last_phi = P
        return cells, P

    # ------------------------------------------------------------------
    # The injection: redo Pass A's lane update with dh = P(p) - P(nb).
    # ------------------------------------------------------------------

    def extra_correction_pass(self) -> None:
        world = self.world
        cs = world.chunk_size
        vpc = world.vpc
        mat_sim = world.material_sim
        damp = self.params.damp
        g_dt = self.params.g_dt
        gravity_bias = self.params.gravity_bias

        flux_list = self._flux_list_snapshot
        cells, P = self._compute_phi(flux_list)

        # Pressure-wave wake (see DEFAULT_WAKE_MARGIN): an over-pressured REAL
        # free surface (air above, or partial fill) is a discharge endpoint
        # that must be dispatched even if its chunk sleeps beyond the current
        # frontier. Takes effect next tick (mirrors the engine's add_candidates
        # / dirty-bitmask wake path).
        for pos, (m, f) in cells.items():
            if P[pos] - f <= self.wake_margin:
                continue
            above = (pos[0], pos[1] + 1, pos[2])
            am = world.read_a(above) & 0xFF
            # Dischargeable: air above (can push up/out) or partial with a
            # CORRECTABLE deficit (can accept refill). A sealed full cell is
            # pure static pressure and never wakes; neither does a submerged
            # cell within REST_TOL of full — a deficit of 1 unit rounds to
            # zero accel and one of settle_threshold-1 is invisible to the
            # sleep scheduler, so waking on it would churn forever against a
            # correction the substrate cannot express.
            rest_tol = max(1, self.params.settle_threshold - 1)
            if am != 0 and f >= FILL_CAP - rest_tol:
                continue
            cp = world.world_to_chunk(pos)
            if cp not in self._active_chunks:
                self.add_active(cp)

        def pval(pos, mat):
            c = cells.get(pos)
            if c is None or c[0] != mat:
                return 0
            return P[pos]

        pre = self._pre_accel_flux
        for cp in flux_list:
            page = self._flux_chunk_table.get(cp, -1)
            if page < 0:
                continue
            slot = world.chunk_to_slot[cp]
            arr = world.voxels[slot]
            for lz in range(cs):
                for ly in range(cs):
                    for lx in range(cs):
                        local = lz * cs * cs + ly * cs + lx
                        gpos = (cp[0] * cs + lx, cp[1] * cs + ly, cp[2] * cs + lz)
                        u = arr[local]
                        mat_id = u & 0xFF
                        my_is_liquid = (mat_id != 0 and mat_sim[mat_id] == SIM_LIQUID)
                        my_is_air = (mat_id == 0)
                        # Fast skip: a voxel that is neither liquid nor air, or
                        # has no liquid in its own cell and no liquid +axis
                        # neighbour, was zeroed / left alone by Accelerate and
                        # phi==head==0 on both sides — nothing to correct.
                        if not my_is_liquid and not my_is_air:
                            continue

                        for axis in range(3):
                            d = AXIS_DIR[axis]
                            nb = (gpos[0] + d[0], gpos[1] + d[1], gpos[2] + d[2])
                            nv = world.read_a(nb)
                            nm = nv & 0xFF

                            nb_is_liquid = (nm != 0 and mat_sim[nm] == SIM_LIQUID)
                            nb_is_air = (nm == 0)

                            # Identical gating to Pass A: only faces Accelerate
                            # actually integrated get rewritten.
                            nb_cp = world.world_to_chunk(nb)
                            nb_page = self.flux_page_table.get(nb_cp, -1) if world.is_loaded(nb_cp) else -1
                            if nb_page < 0:
                                continue
                            if not nb_is_liquid and not nb_is_air:
                                continue
                            if not my_is_liquid and not nb_is_liquid:
                                continue
                            if my_is_liquid and nb_is_liquid and mat_id != nm:
                                continue

                            active_mat = mat_id if my_is_liquid else nm

                            dh_phi = pval(gpos, active_mat) - pval(nb, active_mat)
                            dh_head = self.head(gpos, active_mat) - self.head(nb, active_mat)
                            if dh_phi == dh_head:
                                continue  # lane already holds the identical update

                            lane = page * vpc * 3 + local * 3 + axis
                            if axis == 1:
                                accel = (float(dh_phi) - float(FILL_CAP)) * g_dt * gravity_bias
                                axis_cap = float(FILL_CAP) * gravity_bias
                            else:
                                accel = float(dh_phi) * g_dt
                                axis_cap = float(FILL_CAP)

                            old_f = float(pre[lane])
                            decayed = int(old_f * damp)
                            new_f = clamp(decayed + accel, -axis_cap, axis_cap)
                            self.flux_buf[lane] = gl_round(new_f)
