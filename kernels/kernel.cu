
#include "kernel.cuh"

using namespace PlaneSweepKernel;

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Device-side constant memory for camera parameters and source images, because it's faster to read from there
__constant__ DeviceCam c_ref_cam;
__constant__ DeviceCam c_src_cams[MaxSourceCams];
__constant__ unsigned char *c_src_images[MaxSourceCams];


using cuda_utils::DeviceBuffer;

/**
 * @brief Converts a CPU camera into the GPU-friendly DeviceCam struct.
 * @param camera Camera with OpenCV images and std::vector calibration data.
 * @return Flattened camera parameters, because CUDA kernels do not want the fancy host stuff.
 */
DeviceCam make_device_cam(cam const &camera) {

	DeviceCam out{};
	out.width = camera.width;
	out.height = camera.height;

	// Flatten the 3x3 matrices and 3x1 vectors into arrays for easier GPU access
	for (int i = 0; i < 9; ++i)
	{
		out.K[i] = static_cast<float>(camera.p.K[i]);
		out.R[i] = static_cast<float>(camera.p.R[i]);
		out.K_inv[i] = static_cast<float>(camera.p.K_inv[i]);
		out.R_inv[i] = static_cast<float>(camera.p.R_inv[i]);
	}

	for (int i = 0; i < 3; ++i)
	{
		out.t[i] = static_cast<float>(camera.p.t[i]);
		out.t_inv[i] = static_cast<float>(camera.p.t_inv[i]);
	}

	return out;

}

/**
 * @brief Reads one image byte with the read-only cache path when possible.
 * @param image Device pointer to an image channel.
 * @param idx Flat pixel index.
 * @return Pixel value at idx.
 */
__device__ __forceinline__ unsigned char PlaneSweepKernel::read_image_pixel(unsigned char const *__restrict__ image, int idx) {
	// initially i was using something else but i got advice that __ldg would be faster for image reads
	// and it seems ok in my case so here we are (i'll justify this in the oral maybe idk)
	return __ldg(image + idx);
}

/**
 * @brief Computes the cost for a single pixel/depth/source-camera pair (is it really a pair if there is 3 elements?).
 * @param ref_tile Shared-memory tile of the reference image around the block.
 * @param src_image Source image on the GPU.
 * @param ref Reference camera parameters.
 * @param src Source camera parameters.
 * @param x Reference image x-coordinate.
 * @param y Reference image y-coordinate.
 * @param zi Current depth plane index.
 * @param z_planes Total number of depth planes.
 * @param z_near Near depth bound.
 * @param z_far Far depth bound.
 * @param window Matching window size.
 * @param tile_width Width of the shared-memory reference tile.
 * @param tile_center_x Center x-coordinate for this thread inside the tile.
 * @param tile_center_y Center y-coordinate for this thread inside the tile.
 * @return Average absolute pixel difference for that candidate.
 */
__device__ float PlaneSweepKernel::compute_matching_cost(unsigned char const *__restrict__ ref_tile, unsigned char const *__restrict__ src_image,
	DeviceCam const &ref, DeviceCam const &src, int x, int y, int zi, int z_planes, float z_near, float z_far,
	int window, int tile_width, int tile_center_x, int tile_center_y) {

	// so ok the math here is a bit of a pain but the idea is that we first unproject the reference pixel at the current depth plane
	// to get its 3D coordinates in the world, then we reproject that 3D point into the source camera to see where it would land on
	// the source image, and then we can do a little windowed comparison around that projected point to get our cost for this
	// depth plane and source camera

	const float z = z_near * z_far / (z_near + (static_cast<float>(zi) / static_cast<float>(z_planes)) * (z_far - z_near));

	const float X_ref = (ref.K_inv[0] * x + ref.K_inv[1] * y + ref.K_inv[2]) * z;
	const float Y_ref = (ref.K_inv[3] * x + ref.K_inv[4] * y + ref.K_inv[5]) * z;
	const float Z_ref = (ref.K_inv[6] * x + ref.K_inv[7] * y + ref.K_inv[8]) * z;

	const float X = ref.R_inv[0] * X_ref + ref.R_inv[1] * Y_ref + ref.R_inv[2] * Z_ref - ref.t_inv[0];
	const float Y = ref.R_inv[3] * X_ref + ref.R_inv[4] * Y_ref + ref.R_inv[5] * Z_ref - ref.t_inv[1];
	const float Z = ref.R_inv[6] * X_ref + ref.R_inv[7] * Y_ref + ref.R_inv[8] * Z_ref - ref.t_inv[2];

	const float X_proj = src.R[0] * X + src.R[1] * Y + src.R[2] * Z - src.t[0];
	const float Y_proj = src.R[3] * X + src.R[4] * Y + src.R[5] * Z - src.t[1];
	const float Z_proj = src.R[6] * X + src.R[7] * Y + src.R[8] * Z - src.t[2];

	float x_proj = src.K[0] * X_proj / Z_proj + src.K[1] * Y_proj / Z_proj + src.K[2];
	float y_proj = src.K[3] * X_proj / Z_proj + src.K[4] * Y_proj / Z_proj + src.K[5];

	x_proj = x_proj < 0.0f || x_proj >= src.width ? 0.0f : roundf(x_proj);
	y_proj = y_proj < 0.0f || y_proj >= src.height ? 0.0f : roundf(y_proj);

	const int src_center_x = static_cast<int>(x_proj);
	const int src_center_y = static_cast<int>(y_proj);
	const int radius = window / 2;

	int l_min = -radius;
	int l_max = radius;
	int k_min = -radius;
	int k_max = radius;

	l_min = x + l_min < 0 ? -x : l_min;
	l_max = x + l_max >= ref.width ? ref.width - 1 - x : l_max;
	k_min = y + k_min < 0 ? -y : k_min;
	k_max = y + k_max >= ref.height ? ref.height - 1 - y : k_max;

	l_min = src_center_x + l_min < 0 ? -src_center_x : l_min;
	l_max = src_center_x + l_max >= src.width ? src.width - 1 - src_center_x : l_max;
	k_min = src_center_y + k_min < 0 ? -src_center_y : k_min;
	k_max = src_center_y + k_max >= src.height ? src.height - 1 - src_center_y : k_max;

	if (l_min > l_max || k_min > k_max)
		return 255.0f;

	float cost = 0.0f;
	int samples = 0;

	for (int k = k_min; k <= k_max; ++k)
	{
		const int ref_row = (tile_center_y + k) * tile_width;
		const int src_row = (src_center_y + k) * src.width;
		for (int l = l_min; l <= l_max; ++l)
		{
			const float ref_value = static_cast<float>(ref_tile[ref_row + tile_center_x + l]);
			const float src_value = static_cast<float>(read_image_pixel(src_image, src_row + src_center_x + l));
			cost += fabsf(ref_value - src_value);
			++samples;
		}
	}

	return samples > 0 ? cost / static_cast<float>(samples) : 255.0f;

}

/**
 * @brief CUDA kernel doing the actual sweep over pixels, depth planes, and source cameras.
 * @param ref_image Reference image channel in device memory.
 * @param cost_cube Output cost cube, stored as depth planes one after another.
 * @param source_count Number of source cameras currently uploaded.
 * @param z_planes Total number of depth planes to test.
 * @param z_near Near depth bound.
 * @param z_far Far depth bound.
 * @param window Matching window size.
 */
__global__ void PlaneSweepKernel::sweep_plane_all_cameras_kernel(unsigned char const *__restrict__ ref_image, float *__restrict__ cost_cube,
	int source_count, int z_planes, float z_near, float z_far, int window) {

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int zi = blockIdx.z * blockDim.z + threadIdx.z;

	extern __shared__ unsigned char ref_tile[];
	const int radius = window / 2;
	const int tile_width = blockDim.x + 2 * radius;
	const int tile_height = blockDim.y + 2 * radius;
	const int tile_count = tile_width * tile_height;
	const int tile_origin_x = static_cast<int>(blockIdx.x * blockDim.x) - radius;
	const int tile_origin_y = static_cast<int>(blockIdx.y * blockDim.y) - radius;
	const int linear_thread = threadIdx.y * blockDim.x + threadIdx.x;
	const int block_threads = blockDim.x * blockDim.y;

	if (threadIdx.z == 0)
	{
		for (int tile_idx = linear_thread; tile_idx < tile_count; tile_idx += block_threads)
		{
			const int tile_x = tile_idx % tile_width;
			const int tile_y = tile_idx / tile_width;
			const int global_x = tile_origin_x + tile_x;
			const int global_y = tile_origin_y + tile_y;

			ref_tile[tile_idx] = (global_x >= 0 && global_x < c_ref_cam.width && global_y >= 0 && global_y < c_ref_cam.height)
				? read_image_pixel(ref_image, global_y * c_ref_cam.width + global_x)
				: 0;
		}
	}

	// make sure the whole tile is loaded before any thread tries to read from it
	__syncthreads();

	if (x >= c_ref_cam.width || y >= c_ref_cam.height || zi >= z_planes)
		return;

	float best_cost = 255.0f;

	for (int src_idx = 0; src_idx < source_count; ++src_idx) {
		const float cost = compute_matching_cost(ref_tile, c_src_images[src_idx], c_ref_cam, c_src_cams[src_idx],
			x, y, zi, z_planes, z_near, z_far, window, tile_width, threadIdx.x + radius, threadIdx.y + radius);
		best_cost = fminf(best_cost, cost);
	}

	const size_t idx = (static_cast<size_t>(zi) * c_ref_cam.height + y) * c_ref_cam.width + x;
	cost_cube[idx] = best_cost;

}


/**
 * @brief CPU-side entry point for the CUDA planesweep.
 * @param ref Reference camera/view.
 * @param cam_vector All available cameras, the reference included.
 * @param window Matching window size.
 * @return Cost cube as OpenCV Mats so the rest of the old CPU pipeline can keep chilling.
 */
std::vector<cv::Mat> PlaneSweepKernel::sweeping_plane_cuda(cam const &ref, std::vector<cam> const &cam_vector, int window) {

	if (ref.YUV.empty())
		throw std::runtime_error("[e] Reference camera has no image channels");

	int expected_source_count = 0;

	for (auto const &camera : cam_vector) {
		if (camera.name != ref.name)
			++expected_source_count;
	}

	if (expected_source_count == 0)
		throw std::runtime_error("[e] No source cameras available for CUDA planesweep");
	if (expected_source_count > MaxSourceCams)
		throw std::runtime_error("[e] Too many source cameras for CUDA planesweep constant-memory table");

	const size_t pixel_count = static_cast<size_t>(ref.width) * ref.height;
	const size_t cost_count = pixel_count * ZPlanes;

	cv::Mat ref_y = ref.YUV[0].isContinuous() ? ref.YUV[0] : ref.YUV[0].clone();

	DeviceBuffer<unsigned char> d_ref_image(pixel_count);
	DeviceBuffer<float> d_cost_cube(cost_count);
	std::vector<DeviceBuffer<unsigned char>> src_image_buffers;
	std::vector<unsigned char *> h_src_images;
	std::vector<DeviceCam> h_src_cams;
	src_image_buffers.reserve(expected_source_count);
	h_src_images.reserve(expected_source_count);
	h_src_cams.reserve(expected_source_count);

	cuda_utils::checkCuda(cudaMemcpy(d_ref_image.get(), ref_y.ptr<unsigned char>(), pixel_count * sizeof(unsigned char),
		cudaMemcpyHostToDevice));

	const DeviceCam d_ref = make_device_cam(ref);
	const dim3 block(32, 8, ZPlanesPerBlock);
	const dim3 grid(
		(ref.width + block.x - 1) / block.x,
		(ref.height + block.y - 1) / block.y,
		(ZPlanes + block.z - 1) / block.z);
	const int radius = window / 2;
	const size_t shared_bytes = (block.x + 2 * radius) * (block.y + 2 * radius) * sizeof(unsigned char);

	for (auto const &camera : cam_vector) {

		if (camera.name == ref.name)
			continue;
		if (camera.YUV.empty())
			throw std::runtime_error("[e] Source camera has no image channels: " + camera.name);

		printf("[i] CUDA upload cam: %s\n", camera.name.c_str());

		const size_t camera_pixel_count = static_cast<size_t>(camera.width) * camera.height;
		cv::Mat cam_y = camera.YUV[0].isContinuous() ? camera.YUV[0] : camera.YUV[0].clone();

		src_image_buffers.emplace_back(camera_pixel_count);
		unsigned char *d_cam_image = src_image_buffers.back().get();
		cuda_utils::checkCuda(cudaMemcpy(d_cam_image, cam_y.ptr<unsigned char>(), camera_pixel_count * sizeof(unsigned char),
			cudaMemcpyHostToDevice));

		h_src_images.push_back(d_cam_image);
		h_src_cams.push_back(make_device_cam(camera));
	}

	cuda_utils::checkCuda(cudaMemcpyToSymbol(c_ref_cam, &d_ref, sizeof(DeviceCam)));
	cuda_utils::checkCuda(cudaMemcpyToSymbol(c_src_images, h_src_images.data(), h_src_images.size() * sizeof(unsigned char *)));
	cuda_utils::checkCuda(cudaMemcpyToSymbol(c_src_cams, h_src_cams.data(), h_src_cams.size() * sizeof(DeviceCam)));

	printf("[i] CUDA sweep: %zu source cams, %i depth planes\n", h_src_images.size(), ZPlanes);
	sweep_plane_all_cameras_kernel<<<grid, block, shared_bytes>>>(
		d_ref_image.get(),
		d_cost_cube.get(),
		static_cast<int>(h_src_images.size()),
		ZPlanes,
		ZNear,
		ZFar,
		window);

	cuda_utils::checkCuda(cudaGetLastError());
	cuda_utils::checkCuda(cudaDeviceSynchronize());

	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int zi = 0; zi < ZPlanes; ++zi)
	{
		cost_cube[zi] = cv::Mat(ref.height, ref.width, CV_32FC1);
		cuda_utils::checkCuda(cudaMemcpy(
			cost_cube[zi].ptr<float>(),
			d_cost_cube.get() + static_cast<size_t>(zi) * pixel_count,
			pixel_count * sizeof(float),
			cudaMemcpyDeviceToHost));
	}

	return cost_cube;
}
