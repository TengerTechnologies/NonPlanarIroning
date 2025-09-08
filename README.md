--readme still in the works--

This script is experimental, use with care!

The script allows you to do nonplanar ironing passes with both your nozzle or a separate dedicated tool. (You can also use it for nonplanar toplayers if you set the extrusion amount to 100% and the z-offset to your layerhight)ple

It has basic collision detection and tooloffsets/bounds. 

If you set your toollenght/nozzlelenght you can split the layers into passes which get processesed separately to enable the nonplanar ironing of the whole print, even with a normal nozzle. You can also let it calculate the best places to split the model for access to ironable regions (f.e hidden areas under overhangs etc.)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/G4OjYKChAd8/0.jpg)](https://www.youtube.com/watch?v=G4OjYKChAd8)


Use the script inside your slicer in the postprocessing script:
```"C:\pathToPython\python.exe" "C:\pathToScript\nonplanarironing.py" -argument xy -argument xy```

(Make sure to turn off bgcode)

Currently the script is tested in Prusaslicer and Marlin flavoured gcode. It shoul in theory work in other slicers such as Orca but it is untested.

(The script is also built without libraries like f.e numpy so that less experienced users shouldn't need to install anything) -> there will be a performance optimized version in future.

# CLI parameters

---

## Required

| Flag         | Type | Meaning                                         |
| ------------ | ---- | ----------------------------------------------- |
| `input_file` | path | Path to the **.gcode** file to modify in-place. |

---

## Debug / logging

| Flag          | Type     |                Default | Meaning                                                                            |
| ------------- | -------- | ---------------------: | ---------------------------------------------------------------------------------- |
| `-debug`      | bool     |                **off** | Verbose logging of decisions (masks, counts, timings).                             |
| `-debugReach` | bool     |                **off** | Extra logs about reachability (bands, clearance, clamping).                        |
| `-vizOut`     | str/path | **`c:\`** | Folder to write PNG visualizations of heightfields and masks (created if missing). |

---

## Geometry & sampling

| Flag               | Type  |  Default | Meaning                                                                                                                                                      |
| ------------------ | ----- | -------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-cellSize`        | mm    | **0.20** | Resolution of the heightfield grid (XY). Smaller = finer detail, slower.                                                                                     |
| `-sampleStep`      | mm    | **0.15** | Step along each G1/arc when rasterizing to the grid. Smaller = denser sampling.                                                                              |
| `-smoothBoxRadius` | cells |    **1** | Box-blur radius (in **cells**) to smooth the heightfield and reduce stair-steps. (Changes with cell size)                                                    |
| `-fillRadius`      | mm    |  **0.0** | Gray-dilation radius to “borrow” height from donors and fill narrow top gaps. (Use small value if you see sudden toolpath drops)                             |

---

## Ironing geometry (line shape)

| Flag             | Type |  Default | Meaning                                                                                                                                                                                       |
| ---------------- | ---- | -------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-nozzleDiam`    | mm   | **0.40** | Normal nozzle diameter (used for flow)                                                                                                                                                        |
| `-ironingWidth`  | mm   | **0.40** | Effective width of an ironing pass (used for footprints and flow).                                                                                                                            |
| `-ironingHeight` | mm   | **0.05** | Target "squish" height while ironing ->basically the effectiv ironing layerhight (used for flow).                                                                                             |
| `-ironingPitch`  | mm   | **0.20** | Spacing between scanlines / passes. (not used on adaptive and concentric toolpath -> use stepover instead)                                                                                    |
| `-angle`         | °    |  **0.0** | Primary scanline orientation; `0°` = along X, `90°` = along Y. (the surface pattern)                                                                                                          |
| `-maxAngleDeg`   | °    | **90.0** | Maximum local surface slope (from horizontal) that is allowed for **surface** ironing. Above this is treated as wall/non-iron. (use this for your max angle of the nozzle to avoid collisions)|

---

## Z envelope & reach

| Flag           | Type |   Default | Meaning                                                                                                                      |
| -------------- | ---- | --------: | ---------------------------------------------------------------------------------------------------------------------------- |
| `-zOffset`     | mm   | **-0.05** | Vertical offset applied to computed ironing Z (negative = push slightly into the surface).                                   |
| `-maxDepth`    | mm   |  **0.30** | Maximum sudden downwards movement allowed from the local surface inside a pass (safety clamp).                               |
| `-reachDepth`  | mm   | **10.00** | Tool reach from the **global top** downwards; bands and points deeper than this are considered unreachable unless segmented. |
| `-erodeRadius` | mm   |   **0.0** | Morphological erosion of the surface mask to retract from sharp outer edges.                                                 |

---

## Side / ball clearance (collision envelopes)

| Flag           | Type | Default | Meaning                                                                                                |
| -------------- | ---- | ------: | ------------------------------------------------------------------------------------------------------ |
| `-shaftCheck`  | bool | **off** | Enable lateral “shaft” collision checking.                                                             |
| `-shaftRadius` | mm   | **4.0** | Radius of the **widest cylindrical part** that must not hit the print (e.g., hex-nut or upper shaft).  |
| `-ballRadius`  | mm   | **4.0** | Radius of a spherical tip/ball; enforces a constant margin on steep/curvy areas. Set `0` to disable.   |

---

## Extrusion & feed

| Flag            | Type   |    Default | Meaning                                                                      |
| --------------- | ------ | ---------: | ---------------------------------------------------------------------------- |
| `-feedrate`     | mm/min | **1200.0** | Travel/feed during ironing moves (G1 `F` value).                             |
| `-flowPercent`  | ratio  |   **0.15** | Relative flow during ironing. `0.15` ≈ **15 %** of the flow.                 |
| `-filamentDiam` | mm     |   **1.75** | Filament diameter used for converting line volume ↔ extrusion.               |

> Normal “classic ironing” starting point is `-ironingPitch 0.10 -flowPercent 0.15`.

---

## Path strategy

| Flag        | Type |      Default | Meaning                                                                                      |
| ----------- | ---- | -----------: | -------------------------------------------------------------------------------------------- |
| `-strategy` | enum | **scanline** | Toolpath style: `scanline`, `adaptive` (coarser in flats, finer on detail), or `concentric`. |
| `-stepover` | mm   |     **0.20** | Step for `adaptive`/`concentric` planners (similar to pitch).                                |
| `-minLoop`  | mm   |      **0.3** | Discard polylines shorter than this (noise guard).                                           |

> While this whole script is experimental, adaptibe and concetric path generation is suuuuuuuper experimental and not tested much 
---

## Emission / control

| Flag              | Type |     Default | Meaning                                                                                                                                                                 |
| ----------------- | ---- | ----------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-splitSegments`  | bool |     **off** | Emit ironing **between** print segments (bands), not only at the end.                                                                                                   |
| `-ironTool`       | str  |    **None** | Optional tool to switch to for ironing (e.g. `T1`). Switched back afterward. (depricated, better use pre/post iron snippets for whole toolchange gcode snipped)         |
| `-bandOrder`      | enum | **topdown** | Process bands from `topdown` (default) or `bottomup`.                                                                                                                   |
| `-collisionScope` | enum |  **global** | Collision envelope from the `global` model or only what’s `printed` up to the current band. (Use to make globally unreachable regions, reachable in segment)            |

---

## Adaptive banding (reach-driven segmentation for automatic segment finder based on tool maxReach)

| Flag             | Type     |  Default | Meaning                                                                            |
| ---------------- | -------- | -------: | ---------------------------------------------------------------------------------- |
| `-adaptiveSplit` | bool     |  **off** | Enable adaptive band placement (seek bands that newly unlock reachable areas).     |
| `-bandStep`      | mm       |  **0.0** | Fixed band spacing (if non-zero and `-adaptiveSplit` is **off**).                  |
| `-minNewFrac`    | fraction | **0.02** | Minimum **fraction** of new ironable cells to accept a band during adaptive split. |
| `-minNewAbs`     | cells    |   **25** | Minimum **absolute** number of new cells to accept a band during adaptive split.   |

---

## Travel

| Flag               | Type | Default | Meaning                                                                        |
| ------------------ | ---- | ------: | ------------------------------------------------------------------------------ |
| `-travelClearance` | mm   | **0.2** | Z hop above the band’s local top for non-extruding travels to avoid obstacles. |

---

## Per-band path regeneration

| Flag                 | Type | Default | Meaning                                                                                            |
| -------------------- | ---- | ------: | -------------------------------------------------------------------------------------------------- |
| `-regenPerBandPaths` | bool | **off** | Recompute toolpaths per band (recommended when using adaptive bands or `printed` collision scope). |

> This takes a lot of time to compute
---

## Tool offsets (mounting offsets of the ironing tool)

| Flag           | Type |  Default | Meaning                                                                    |
| -------------- | ---- | -------: | -------------------------------------------------------------------------- |
| `-ironOffsetX` | mm   | **0.0** | Tool X offset from the primary nozzle (positive = tool is to the +X side).  |
| `-ironOffsetY` | mm   | **0.0** | Tool Y offset from the primary nozzle.                                      |
| `-ironOffsetZ` | mm   | **0.0** | Tool Z offset (positive = tool tip is **above** the nozzle reference).      |

---

## Custom G-code snippets (toolchange, prep, cleanup)

| Flag        | Type                |                        Default | Meaning                                                                      |
| ----------- | ------------------- | -----------------------------: | ---------------------------------------------------------------------------- |
| `-preIron`  | str / `@file.gcode` |        **``** | Snippet emitted **before** ironing (inline string or `@path/to/file.gcode`).                  |
| `-postIron` | str / `@file.gcode` | **``**        | Snippet emitted **after** ironing.                                                            |

> **Quoting (Windows / bash):**
> Use single quotes around the whole argument and escape inner quotes, e.g.
> `-preIron 'M118 P2 "G1 Z-32"'`

---

### A few good starting presets

* **Classic top ironing:**
  `-strategy scanline -ironingPitch 0.10 -flowPercent 0.15 -maxAngleDeg 15`

* **Nonplanar smoothing (gentle slopes):**
  `-maxAngleDeg 45 -shaftCheck -shaftRadius 4 -ballRadius 2 -erodeRadius 0.3`

* **Reach-limited cavities with banding:**
  `-splitSegments -adaptiveSplit -reachDepth 10 -collisionScope printed -regenPerBandPaths`  

Tweak `cellSize`, `sampleStep`, and `smoothBoxRadius` for quality vs. speed








