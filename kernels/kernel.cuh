#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include "cuda_utils.cuh"
#include "device_types.cuh"
#include "../src/constants.hpp"

/**
 * @brief Converts the CPU-side camera into the tiny plain struct CUDA can digest.
 * @param camera The OpenCV/std::vector camera object we use on the host.
 * @return The same parameters, flattened into boring arrays because kernels like boring arrays.
 */
DeviceCam make_device_cam(cam const &camera);

namespace PlaneSweepKernel {

    /**
     * @brief Reads one image byte, using the read-only cache when the GPU supports it.
     * @param image Device pointer to the image channel.
     * @param idx Flat pixel index in the image.
     * @return The pixel value at that index.
     */
    __device__ __forceinline__ unsigned char read_image_pixel(unsigned char const *__restrict__ image, int idx);

    /**
     * @brief Computes the matching cost for one pixel, one depth plane, and one source camera.
     * @param ref_tile Shared-memory tile around the reference pixel.
     * @param src_image Source image stored in device memory.
     * @param ref Reference camera parameters.
     * @param src Source camera parameters.
     * @param x Pixel x-coordinate in the reference view.
     * @param y Pixel y-coordinate in the reference view.
     * @param zi Depth plane index.
     * @param z_planes Total number of depth planes.
     * @param z_near Nearest depth bound.
     * @param z_far Farthest depth bound.
     * @param window Matching window size, usually something tiny like 5.
     * @param tile_width Width of the shared-memory reference tile.
     * @param tile_center_x Pixel center x-coordinate inside the tile.
     * @param tile_center_y Pixel center y-coordinate inside the tile.
     * @return Average absolute difference cost, lower means better match.
     */
    __device__ float compute_matching_cost(unsigned char const *__restrict__ ref_tile, unsigned char const *__restrict__ src_image,
        DeviceCam const &ref, DeviceCam const &src, int x, int y, int zi, int z_planes, float z_near, float z_far, int window,
        int tile_width, int tile_center_x, int tile_center_y);

    /**
     * @brief Main planesweep kernel, one thread handles one reference pixel and one depth plane.
     * @param ref_image Reference image channel in device memory.
     * @param cost_cube Output cost cube, laid out as depth-major planes.
     * @param source_count Number of source cameras to compare against.
     * @param z_planes Total number of depth candidates.
     * @param z_near Nearest depth bound.
     * @param z_far Farthest depth bound.
     * @param window Matching window size.
     */
    __global__ void sweep_plane_all_cameras_kernel(unsigned char const *__restrict__ ref_image, float *__restrict__ cost_cube,
        int source_count, int z_planes, float z_near, float z_far, int window);

    /**
     * @brief Extracts the lowest-cost depth for each pixel directly on the GPU.
     * @param cost_cube Cost cube laid out as depth-major planes.
     * @param depth Output 8-bit depth map in device memory.
     * @param width Image width.
     * @param height Image height.
     * @param z_planes Total number of depth planes to scan.
     */
    __global__ void find_min_depth_kernel(float const *__restrict__ cost_cube, unsigned char *__restrict__ depth,
        int width, int height, int z_planes);

    /**
     * @brief Host wrapper that uploads images, launches the CUDA sweep, and brings the cost cube back.
     * @param ref Reference camera, aka the view we are estimating depth for.
     * @param cam_vector All cameras, including the reference one.
     * @param window Matching window size, defaulting to 3 because small windows keep things sane.
     * @return A vector of OpenCV float images, one image per depth plane.
     */
    std::vector<cv::Mat> sweeping_plane_cuda(cam const &ref, std::vector<cam> const &cam_vector, int window = 3);

    /**
     * @brief Host wrapper for the fast path: sweep on GPU, find min on GPU, download only the depth map.
     * @param ref Reference camera, aka the view we are estimating depth for.
     * @param cam_vector All cameras, including the reference one.
     * @param window Matching window size.
     * @return 8-bit depth map where each pixel stores the best depth plane.
     */
    cv::Mat sweeping_plane_cuda_min_depth(cam const &ref, std::vector<cam> const &cam_vector, int window = 3);
    
}
