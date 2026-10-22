# Inchworm Sim: Design

## Goal (~2 weeks)
Logical world + MuJoCo viewer that can replay a sequence of
actions (move, place, remove) toward a target structure.

## Conventions
- Axes: z is up. Origin (0,0,0) is the corner of the ground plane.
- Grid cells indexed from 0. Block edge = ___ m.
- Faces: +x, -x, +y, -y, +z, -z

## Entities
- Block: id, grid position, face connection
- Inchworm: id, block location and face its on for each EE
    - Only attach to faces

## Actions
Step(direction), placeBlock(face), removeBlock(face)
    connect(endeffector), disconnect(endeffector)

## Validity rules (v1)
1. Target cell must be in bounds and empty
2. Placed block must touch ground or another block
3. 

## Layers and ownership
- core/ (Person A), physics/ (Person B). Core never imports mujoco.

## Open questions
- ...

## Decisions log
- 2026-10-22: Initial design in progress