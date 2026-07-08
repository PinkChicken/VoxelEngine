"""
LevelSurface reference model v3 — replaces v2's vertical-fill "submerged" test
(SUBMERGED_MIN / proj_level_submerged_damping) with a PRESSURE-DIFFERENCE test.

Why: v2's damping asked "how much fill is directly above me" as a proxy for "does the
pressure solve already own this cell's redistribution." That proxy is wrong whenever a
buried low-fill seam sits inside an otherwise-uniformly-stacked blob (confirmed in-engine:
a fully-submerged 255-fill cell with genuinely open ~11-fill lateral neighbours never
transferred to them at all, because vertical fill above it alone was enough to zero its
damping and fully disable LevelSurface for that cell -- regardless of whether the lateral
neighbours had real room).

The actual question is not "how much is stacked above me" but "does the pressure solve
already have a real gradient here." BuildRHS/solve_red/solve_black already compute that
every tick (self.P, inherited from utube_projection.Sim) -- so ask it directly:
  pair_damping = clamp(1 - |P_self - P_neighbour| / PRESSURE_MIN, 0, 1)
A free-surface/dry neighbour reads as P=0 (matches the existing Dirichlet convention).

This has a pleasant, testable consequence: in every scenario this project has ALREADY
validated (flat uneven puddle, droplet, thin film over a reservoir), BuildRHS's divergence
is zero or near-zero everywhere, so the solved P is uniformly ~0 -- meaning
|P_self - P_neighbour| ~ 0 for every pair, and this new formula reduces to full strength
(damping ~ 1) automatically, identical to v2's behaviour in those cases. Only the
buried-seam / genuinely-different-column-height cases should differ, and they should differ
in the RIGHT direction: a real column-height difference (U-tube before it settles) has a
real |P_self - P_neighbour|, correctly deferring to the projection's own Correct push; a
same-total-stacking buried seam has near-zero |P_self - P_neighbour| even though raw fill
differs a lot, correctly letting LevelSurface act where the projection structurally cannot
see anything.

Subclasses utube_projection.py's Sim (not SimHeightLevel -- that's v2's ancestor and still
carries the vertical SUBMERGED_MIN concept this file replaces).
"""
import importlib.util, pathlib

_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("utube_projection", _here / "utube_projection.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
AIR, LIQUID, SOLID, FILL_CAP = _base.AIR, _base.LIQUID, _base.SOLID, _base.FILL_CAP

LEVEL_DEGREE = 4.0

NB_SURFACE, NB_DRY, NB_NONE = 0, 1, 2


class SimLevelSurfaceV3(_base.Sim):
    def __init__(self, W, H):
        super().__init__(W, H)
        self.LEVEL_RATE = 0.20
        self.LEVEL_MIN = 1.0
        self.PRESSURE_MIN = 2.0  # |P_self - P_neighbour| beyond this = a real gradient, defer to the solve
        self._last_moved = 0.0

    def _pressure_at(self, x, y, cls):
        # Matches proj_pressure_at's convention: only liquid (SURFACE) cells have a real
        # solved P; anything else (dry, sealed, off-grid) reads as the free-surface P=0.
        if cls != NB_SURFACE:
            return 0.0
        return self.P[y][x]

    def _classify(self, x, y):
        if not (0 <= x < self.W and 0 <= y < self.H):
            return NB_NONE, 0
        m = self.mat[y][x]
        if m == LIQUID:
            return NB_SURFACE, LIQUID
        if m == AIR:
            floor_sealed = (y - 1 < 0) or self.mat[y - 1][x] == SOLID
            return (NB_DRY, 0) if floor_sealed else (NB_NONE, 0)
        return NB_NONE, 0  # solid

    def level_surface(self):
        H, W = self.H, self.W
        deltas = [[0.0] * W for _ in range(H)]
        donor_mats = [[None] * W for _ in range(H)]

        for y in range(H):
            for x in range(W):
                m = self.mat[y][x]
                if m == LIQUID:
                    self_cls = NB_SURFACE
                elif m == AIR:
                    floor_sealed = (y - 1 < 0) or self.mat[y - 1][x] == SOLID
                    self_cls = NB_DRY if floor_sealed else NB_NONE
                else:
                    self_cls = NB_NONE
                if self_cls == NB_NONE:
                    continue

                self_fill = self.fill[y][x] if self_cls == NB_SURFACE else 0.0
                p_self = self._pressure_at(x, y, self_cls)
                delta = 0
                donor_mat = None

                for dx in (-1, 1):
                    nx = x + dx
                    nb_cls, nb_mat = self._classify(nx, y)
                    if nb_cls == NB_NONE:
                        continue
                    if self_cls == NB_DRY and nb_cls == NB_DRY:
                        continue
                    if self_cls == NB_SURFACE and nb_cls == NB_SURFACE and nb_mat != LIQUID:
                        continue

                    nb_fill = self.fill[y][nx] if nb_cls == NB_SURFACE else 0.0
                    diff = self_fill - nb_fill
                    if abs(diff) < self.LEVEL_MIN:
                        continue

                    p_nb = self._pressure_at(nx, y, nb_cls)
                    pair_damping = max(0.0, min(1.0, 1.0 - abs(p_self - p_nb) / self.PRESSURE_MIN))

                    pair_delta = round(diff * (self.LEVEL_RATE / LEVEL_DEGREE) * pair_damping)
                    delta -= pair_delta

                    if pair_delta != 0 and self_cls == NB_DRY and nb_cls == NB_SURFACE:
                        donor_mat = LIQUID

                deltas[y][x] = delta
                donor_mats[y][x] = donor_mat

        moved = 0
        for y in range(H):
            for x in range(W):
                delta = deltas[y][x]
                if delta == 0:
                    continue
                is_dry = self.mat[y][x] == AIR
                old_fill = self.fill[y][x] if not is_dry else 0.0
                new_fill = max(0.0, min(FILL_CAP, old_fill + delta))
                if new_fill == old_fill:
                    continue
                if new_fill <= 1e-9:
                    self.mat[y][x] = AIR
                    self.fill[y][x] = 0.0
                elif is_dry:
                    if donor_mats[y][x] is None:
                        continue
                    self.mat[y][x] = LIQUID
                    self.fill[y][x] = new_fill
                else:
                    self.fill[y][x] = new_fill
                moved += abs(new_fill - old_fill)
        return moved

    def tick(self):
        super().tick()
        m = self.level_surface()
        self._last_moved = max(self._last_moved, m)
