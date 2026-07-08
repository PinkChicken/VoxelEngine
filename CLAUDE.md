# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A voxel sandbox engine built in **Godot 4** (GDScript) with GPU-accelerated raymarched rendering. The engine stores 4-byte voxels on the CPU (`block_data`) and renders them via a GPU compute shader raymarcher. Granular (sand) simulation runs on the GPU via `granular_sim.comp`.

## Working with the User

The user works nights — don't treat the hour as a reason to suggest wrapping up or picking things up "tomorrow." Keep going through checkpoints/testing at their pace; they'll say when they want to stop.

## Running the Project

Open the project in Godot 4 and press F5 (or the Play button) to run `world.tscn` as the main scene. There are no CLI build commands — all development is done through the Godot editor. The `@tool` annotation on several scripts means they also run in the editor (ChunkManager, Chunk, TerrainGetter).

## ARCHITECTURE.md / docs/BUG_LOG.md — Keep It Current

**After every code change, update `ARCHITECTURE.md` (and `docs/BUG_LOG.md` for bugs) to match.** This is mandatory, not optional.

`docs/BUG_LOG.md` was split out of `ARCHITECTURE.md`'s old Section 4 (2026-07-02) once that file grew too large to comfortably read in one sitting. It is the *only* file that moved — everything else stays in `ARCHITECTURE.md`.

What to update:
- Mark phases/steps as `✅ COMPLETE` when they are finished (`ARCHITECTURE.md`)
- Add new bugs to **`docs/BUG_LOG.md`** with: symptom, root cause, fix, and any rule to remember
- Log failed approaches in the bug log under the relevant bug (so we don't retry them)
- Update **Section 5 — Current To-Do** in `ARCHITECTURE.md`: check off completed items, add newly discovered issues
- Update phase descriptions in `ARCHITECTURE.md` to reflect what was actually built vs. what was planned

Together, `ARCHITECTURE.md` and `docs/BUG_LOG.md` are the single source of truth for design decisions, completed work, known bugs, and future plans. Claude reads both at the start of each session to reconstruct context.

## Architecture (Current State)

Read `ARCHITECTURE.md` for the full design. Key points:

### Scene Graph (`world.tscn`)
- **WorldOrchestrator** — owns all GPU buffers, sole writer for voxel mutations, drives simulation
- **ChunkManager** — owns all `Chunk` instances, chunk lifecycle, terrain generation
- **RaymarchRenderer** — GPU raymarcher, pure consumer of buffers owned by WorldOrchestrator
- **MaterialDB** — holds `MaterialDef` resources, exposes GPU material buffer
- **SimulationRuntime** (child of WorldOrchestrator) — schedules simulations, owns `_active` list
- **GranularSimulation** (child of SimulationRuntime) — GPU sand simulation
- **CollisionSystem** (in world.tscn) — GPU surface extraction, PhysicsServer3D shape management
- **EditDispatcher** (child of WorldOrchestrator, created in `_ready()`) — GPU edit shader dispatcher

### Voxel Storage
`Chunk` (`Chunks/chunk_instance.gd`) stores voxel data as `block_data: PackedByteArray`, 4 bytes per voxel:

| Byte | Field | Notes |
|------|-------|-------|
| 0 | material_id | 0 = air |
| 1 | fill | 255 = full, 0 = empty |
| 2 | temperature | signed, encode_s8/decode_s8 |
| 3 | flags | engine flags |

Index: `(z * cs² + y * cs + x) * VOXEL_STRIDE`. The 4-byte layout is identical to GPU `u32` little-endian — uploaded with no conversion.

**GPU is the primary source of truth.** `block_data` is populated once at chunk generation and is only re-synced at save time. Never read `block_data` expecting current runtime state.

### GPU Buffer Ownership (WorldOrchestrator)
| Buffer | Purpose |
|--------|---------|
| `_voxel_buf_a` | Canonical voxel state — renderer and simulation read this |
| `_voxel_buf_b` | Simulation scratch — granular CAS writes here, never swapped |
| `_chunk_table` | Dense grid: chunk position → slot index (s32, -1 = not loaded) |
| `_changelog_buf` | GPU-written `(flat_idx, new_value)` pairs from edit/sim shaders |
| `_material_sim_buf` | 256 u32s: material_id → SimulationClass |
| `_sim_active_buf` | 1 u32 per voxel slot; non-zero = grain moved this tick (particle, not solid) |

### Key GDScript Gotchas
- **`Vector3i(0,0,0)` is falsy.** `dict.get(key)` returning a zero vector evaluates as falsy. Always use `dict.get(key, null)` and compare `== null`. Never use `if not chunk_pos:` when the value could be a zero vector.
- **`fill()` on a zero-length PackedByteArray is a no-op.** Always `resize()` before `fill(0)`.
- **Background threads can call `rebuild_counts_and_mask` before deferred init.** `resize()` before `fill()` is required everywhere.

### Key GLSL Gotchas
- `flat` is a reserved interpolation qualifier — cannot be used as a variable name in shaders.
- Non-ASCII characters (em dashes, curly quotes) in GLSL comments corrupt tokenization on some drivers.

### Player Controls
- Left/right click: add/remove voxel
- `B`: spawn sphere of selected material
- `1-4` / mouse wheel: select material
- `F`: toggle fly mode
- `C`: toggle collision debug visualization
- `Escape`: release mouse

## Key Paths
- `ARCHITECTURE.md` — full design, phase history, bug log, current to-do
- `WorldOrchestrator.gd` — GPU buffer management, mutation commit, edit API wrappers
- `simulations/edit_dispatcher.gd` — GPU edit shader, player tool dispatch
- `simulations/granular_simulation.gd` — GPU sand simulation dispatcher
- `simulations/simulation_runtime.gd` — simulation scheduling, chunk discovery
- `simulations/collision_system.gd` — GPU surface extraction, shape management
- `shaders/raymarch.comp` — voxel raymarcher
- `shaders/granular_sim.comp` — GPU sand simulation
- `shaders/edit_apply.comp` — GPU edit (add/remove/replace shapes)
- `shaders/zero_slots.comp` — zeros voxels_b before each granular dispatch
- `shaders/patch_apply.comp` — patches voxels_a from simulation changelog
- `shaders/surface_extract.comp` — extracts surface voxels for collision
- `Chunks/chunk_instance.gd` — chunk storage, voxel accessors, simulation mask
- `Chunks/chunk_manager.gd` — chunk lifecycle, terrain generation, streaming
- `material_def.gd` / `material_db.gd` — material definitions and GPU buffer
- `Utils/compute_helper.gd` — RenderingDevice wrapper
- `addons/terrain_regenerator/` — editor plugin to trigger terrain regeneration

## File Notes
- `.uid` files are Godot 4 resource UID files — keep them in version control
- `.tmp` files are Godot crash-recovery backups — ignored via `.gitignore`
