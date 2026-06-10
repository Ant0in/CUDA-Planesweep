# CUDA Planesweep Optimizations

This note summarizes the optimizations added during the CUDA planesweep work.

## 1. CPU Planesweep Moved To CUDA

The original expensive part was the nested loop over:

```text
cameras -> depth planes -> y -> x -> matching window
```

We moved the cost computation to CUDA. One GPU thread computes one `(x, y, depth)` candidate.

## 2. Cost Cube Computed In Parallel

The GPU fills:

```text
cost_cube[depth][y][x]
```

Pixels and depth planes are independent, so they are computed in parallel.

## 3. Flattened Camera Data

The GPU uses `DeviceCam` instead of CPU-side objects like `std::vector` or `cv::Mat`.

```cpp
float K[9], R[9], t[3];
```

This is simpler and more CUDA-friendly.

## 4. Constant Memory For Camera Parameters

Camera parameters are read by all threads, so they are stored in CUDA constant memory:

```cpp
__constant__ DeviceCam c_ref_cam;
__constant__ DeviceCam c_src_cams[];
```

This is efficient because many threads read the same values.

## 5. Source Cameras Fused Into One Kernel

Instead of launching one kernel per source camera, each thread loops over source cameras and keeps the best cost.

This reduces kernel launch overhead and avoids repeatedly updating the same cost cube cell.

## 6. Removed Cost-Cube Initialization Kernel

Instead of initializing the whole cost cube to `255`, each thread starts with:

```cpp
best_cost = 255.0f;
```

Then it writes the final best cost once.

## 7. Shared Memory For Reference Image Tiles

Neighboring threads compare overlapping reference-image windows.

A block loads a padded reference tile into shared memory, and all threads in the block reuse it.

This avoids repeatedly reading the same reference pixels from global memory.

## 8. Read-Only Cache For Image Reads

Image data is read-only, so source/reference image reads use:

```cpp
__ldg(...)
```

This helps cached image reads, especially for projected source pixels.

## 9. Z-Plane Reuse Inside Blocks

The kernel uses:

```cpp
ZPlanesPerBlock = 2
```

So one CUDA block handles more than one depth plane and can reuse the same shared-memory reference tile.

## 10. Better Block Shape

The block shape is:

```cpp
dim3 block(32, 8, ZPlanesPerBlock);
```

This gives a good number of threads per block while keeping memory access more row-friendly.

## 11. Safer GPU Memory Management

`DeviceBuffer<T>` automatically frees CUDA memory when it goes out of scope.

This avoids forgetting `cudaFree(...)` and makes error paths safer.

## 12. Avoided An Extra Huge CPU Copy

The full cost cube is copied directly from GPU memory into `cv::Mat` planes.

Earlier, it could have gone through an extra giant temporary CPU vector first.

## 13. GPU Min-Depth Extraction

The `find_min` path was parallelized.

One CUDA thread owns one pixel, scans all depth planes for that pixel, and writes the best depth.

## 14. Fast Min Path Avoids Downloading The Full Cost Cube

For `min` mode:

```text
GPU sweep -> GPU argmin -> download final depth map only
```

This avoids copying the huge cost cube back to CPU.

## 15. Graph Cut Kept CPU-Side

Graph cut was not moved to GPU.

This implementation uses dynamic graph structures and maxflow operations with strong dependencies. It is not impossible in theory, but this CPU implementation is not GPU-friendly.

## 16. Reduced Noisy Prints

Per-camera upload prints and per-depth graph-cut layer prints were removed.

This keeps output readable and avoids unnecessary console overhead.

## 17. Timing Instrumentation

Timers were added for:

- CUDA camera upload
- CUDA sweep
- CUDA min-depth extraction
- graph-cut extraction
- total execution time

This helps show where time is actually spent.

## Why `32 x 8` Blocks Instead Of `16 x 16`?

Both have 256 threads per depth slice:

```text
32 * 8  = 256
16 * 16 = 256
```

The difference is the shape.

Images are stored row-major, so neighboring `x` values are next to each other in memory. With `32 x 8`, a warp naturally covers a long horizontal run of pixels, which usually gives better coalesced memory access.

With `16 x 16`, warps are more likely to span multiple rows, which can make memory access a bit less clean.

So the short answer is:

```text
32 x 8 keeps warps more horizontal, which matches image memory layout better.
```

It is not always guaranteed to be faster, but it is a good practical choice for image kernels. The real proof would be profiling both versions.
