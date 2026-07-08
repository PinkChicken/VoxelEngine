"""
LevelSurface reference model — mirrors shaders/fluid_flux.comp's params.pass == 6 block
EXACTLY as of the 2026-07-06 fix pass (five bugs found and fixed in one debugging session,
see docs/BUG_LOG.md). This is NOT the original height_level_prototype.py design bench —
that file stays as the historical record of how 5b was chosen. This file exists so the
CURRENT shipped GLSL logic has a Python-testable twin; keep it in sync whenever
fluid_flux.comp's pass 6 changes, or this stops being a meaningful regression tool.

The five fixes this mirrors:
  1. LEVEL_DEGREE=4.0 normalization — un-normalized 4-direction summing let an exposed
     droplet lose up to 4x LEVEL_RATE per tick instead of LEVEL_RATE.
  2. Classify-by-what's-actually-there, not by the asker's own material — a DRY cell asking
     about a liquid neighbour must not gate on matching its own (air) material.
  3. SUBMERGED_MIN threshold — "submerged" means fill directly above EXCEEDS a threshold,
     not merely nonzero (a thin residual film must not permanently lock a reservoir out).
  4. Continuous damping instead of a boolean submerged gate — a boolean re-evaluated every
     tick against naturally-noisy input (settle jitter) flickers a cell's whole contribution
     on/off; a continuous factor bounds the ripple to the size of the noise instead.
  5. Per-pair integer rounding instead of aggregate-then-round — summing every active
     neighbour's float contribution and rounding ONCE breaks pairwise cancellation the
     moment a cell has 2+ active neighbours (routine in any real puddle), producing a real,
     consistently-directional mass leak. Uses Python's round() (banker's rounding,
     round-half-to-even) to match GLSL's roundEven() — deliberately not plain round()/GLSL
     round(), whose tie-breaking is implementation-defined and not guaranteed symmetric
     under negation, which this fix's cancellation guarantee depends on.

Only X is available as a lateral axis in this 2D (x, y-up) bench, matching every other file
in this folder — LEVEL_DEGREE is still divided by 4.0 (matching the real 3D engine's X+Z
lateral count) even though only 2 of those 4 "slots" can ever be active here. That's
intentional: it reproduces the actual engine constant, not an idealized 2D-only value.
"""
import importlib.util, pathlib

_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("utube_projection", _here / "utube_projection.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
AIR, LIQUID, SOLID, FILL_CAP = _base.AIR, _base.LIQUID, _base.SOLID, _base.FILL_CAP

LEVEL_DEGREE = 4.0

# Classification results, matching LEVEL_NB_* in fluid_flux.comp.
NB_SURFACE, NB_DRY, NB_NONE = 0, 1, 2


class SimLevelSurfaceV2(_base.Sim):
    def __init__(self, W, H):
        super().__init__(W, H)
        self.LEVEL_RATE = 0.20
        self.LEVEL_MIN = 1.0
        self.SUBMERGED_MIN = 2.0
        self._last_moved = 0.0

    # ---- mirrors proj_level_submerged_damping ----
    def _damping(self, x, y, mat):
        if not (0 <= y + 1 < self.H) or self.mat[y + 1][x] != mat:
            return 1.0
        fill_above = self.fill[y + 1][x]
        return max(0.0, min(1.0, 1.0 - fill_above / self.SUBMERGED_MIN))

    # ---- mirrors proj_level_classify (no submerged gating, no material pre-filter) ----
    def _classify(self, x, y):
        if not (0 <= x < self.W and 0 <= y < self.H):
            return NB_NONE, 0
        m = self.mat[y][x]
        if m == LIQUID:
            return NB_SURFACE, LIQUID  # single-liquid-type bench: material is always LIQUID
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
                self_cls = NB_NONE
                self_damping = 1.0
                if m == LIQUID:
                    self_cls = NB_SURFACE
                    self_damping = self._damping(x, y, LIQUID)
                elif m == AIR:
                    floor_sealed = (y - 1 < 0) or self.mat[y - 1][x] == SOLID
                    self_cls = NB_DRY if floor_sealed else NB_NONE
                if self_cls == NB_NONE:
                    continue
                if self_cls == NB_SURFACE and self_damping <= 0.0:
                    continue

                self_fill = self.fill[y][x] if self_cls == NB_SURFACE else 0.0
                delta = 0  # BUG (found 2026-07-06, in-engine): must be an INTEGER accumulator,
                # not a float -- see the per-pair rounding note below.
                donor_mat = None

                for dx in (-1, 1):
                    nx = x + dx
                    nb_cls, nb_mat = self._classify(nx, y)
                    if nb_cls == NB_NONE:
                        continue
                    if self_cls == NB_DRY and nb_cls == NB_DRY:
                        continue
                    if self_cls == NB_SURFACE and nb_cls == NB_SURFACE and nb_mat != LIQUID:
                        continue  # single-liquid-type bench: always matches, kept for parity

                    nb_fill = self.fill[y][nx] if nb_cls == NB_SURFACE else 0.0
                    diff = self_fill - nb_fill
                    if abs(diff) < self.LEVEL_MIN:
                        continue

                    nb_damping = self._damping(nx, y, LIQUID) if nb_cls == NB_SURFACE else 1.0
                    pair_damping = min(self_damping, nb_damping)

                    # BUG (found 2026-07-06, in-engine): the original version summed every
                    # active pair's float contribution and rounded ONCE at the end. Once a
                    # cell has 2+ active neighbours (routine in any real 2D/3D puddle, just
                    # never exercised by the old 1D-single-neighbour test cases), round(a+b)
                    # != round(a)+round(b) in general, so the aggregate-then-rounded total has
                    # no guaranteed relationship to what each neighbour independently rounds
                    # on its own side -- a real, structural, consistently-directional leak,
                    # not ordinary rounding noise. This Python bench COULD NOT have caught it
                    # before this fix: it used continuous floats throughout with no integer
                    # quantization step at all, unlike the real 8-bit-fill GLSL engine. Fix:
                    # round PER PAIR, immediately, before summing -- Python's round() uses
                    # banker's rounding (round-half-to-even), which is symmetric under
                    # negation (round(x) == -round(-x) exactly), matching GLSL's roundEven()
                    # (deliberately NOT round(), whose tie-breaking is implementation-defined
                    # and not guaranteed symmetric).
                    pair_delta = round(diff * (self.LEVEL_RATE / LEVEL_DEGREE) * pair_damping)
                    delta -= pair_delta

                    if pair_delta != 0 and self_cls == NB_DRY and nb_cls == NB_SURFACE:
                        donor_mat = LIQUID

                deltas[y][x] = delta
                donor_mats[y][x] = donor_mat

        moved = 0.0
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
                        continue  # no valid donor this tick -- spurious, skip (mirrors Transport)
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
