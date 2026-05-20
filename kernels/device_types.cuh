#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief This struct represents a camera in a format suitable for our CUDA kernels.
 * It contains the intrinsic and extrinsic parameters of the camera, as well as their inverses for efficient computation.
 */
struct DeviceCam {

	int width;
	int height;

	float K[9];
	float R[9];
	float t[3];
	float K_inv[9];
	float R_inv[9];
	float t_inv[3];

};

constexpr int MaxSourceCams = 16;
constexpr int ZPlanesPerBlock = 2;

extern __constant__ DeviceCam c_ref_cam;
extern __constant__ DeviceCam c_src_cams[MaxSourceCams];
extern __constant__ unsigned char *c_src_images[MaxSourceCams];
