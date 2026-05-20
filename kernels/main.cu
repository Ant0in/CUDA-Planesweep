#include "main.cuh"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct DeviceCam
{
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

__constant__ DeviceCam c_ref_cam;
__constant__ DeviceCam c_src_cams[MaxSourceCams];
__constant__ unsigned char *c_src_images[MaxSourceCams];

void check_cuda(cudaError_t err, char const *call)
{
	if (err != cudaSuccess)
		throw std::runtime_error(std::string(call) + " failed: " + cudaGetErrorString(err));
}

#define CUDA_CHECK(call) check_cuda((call), #call)

DeviceCam make_device_cam(cam const &camera)
{
	DeviceCam out{};
	out.width = camera.width;
	out.height = camera.height;

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

__device__ float compute_matching_cost(
	unsigned char const *__restrict__ ref_image,
	unsigned char const *__restrict__ src_image,
	DeviceCam const &ref,
	DeviceCam const &src,
	int x,
	int y,
	int zi,
	int z_planes,
	float z_near,
	float z_far,
	int window)
{
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
		const int ref_row = (y + k) * ref.width;
		const int src_row = (src_center_y + k) * src.width;
		for (int l = l_min; l <= l_max; ++l)
		{
			const float ref_value = static_cast<float>(ref_image[ref_row + x + l]);
			const float src_value = static_cast<float>(src_image[src_row + src_center_x + l]);
			cost += fabsf(ref_value - src_value);
			++samples;
		}
	}

	return samples > 0 ? cost / static_cast<float>(samples) : 255.0f;
}

__global__ void sweep_plane_all_cameras_kernel(
	unsigned char const *__restrict__ ref_image,
	float *__restrict__ cost_cube,
	int source_count,
	int z_planes,
	float z_near,
	float z_far,
	int window)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int zi = blockIdx.z;

	if (x >= c_ref_cam.width || y >= c_ref_cam.height || zi >= z_planes)
		return;

	float best_cost = 255.0f;
	for (int src_idx = 0; src_idx < source_count; ++src_idx)
	{
		const float cost = compute_matching_cost(
			ref_image,
			c_src_images[src_idx],
			c_ref_cam,
			c_src_cams[src_idx],
			x,
			y,
			zi,
			z_planes,
			z_near,
			z_far,
			window);
		best_cost = fminf(best_cost, cost);
	}

	const size_t idx = (static_cast<size_t>(zi) * c_ref_cam.height + y) * c_ref_cam.width + x;
	cost_cube[idx] = best_cost;
}
}

// Those functions are an example on how to call cuda functions from the main.cpp

__global__ void dev_test_vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	C[i] = A[i] + B[i];
}

void wrap_test_vectorAdd() {
	printf("Vector Add:\n");

	int N = 3;
	int a[] = { 1, 2, 3 };
	int b[] = { 1, 2, 3 };
	int c[] = { 0, 0, 0 };

	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice);

	dev_test_vecAdd <<<1, N>>> (dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	for (int i = 0; i < N; ++i) {
		printf("%i + %i = %i\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

std::vector<cv::Mat> sweeping_plane_cuda(cam const &ref, std::vector<cam> const &cam_vector, int window)
{
	if (ref.YUV.empty())
		throw std::runtime_error("Reference camera has no image channels");

	int expected_source_count = 0;
	for (auto const &camera : cam_vector)
	{
		if (camera.name != ref.name)
			++expected_source_count;
	}
	if (expected_source_count == 0)
		throw std::runtime_error("No source cameras available for CUDA planesweep");
	if (expected_source_count > MaxSourceCams)
		throw std::runtime_error("Too many source cameras for CUDA planesweep constant-memory table");

	const size_t pixel_count = static_cast<size_t>(ref.width) * ref.height;
	const size_t cost_count = pixel_count * ZPlanes;

	cv::Mat ref_y = ref.YUV[0].isContinuous() ? ref.YUV[0] : ref.YUV[0].clone();

	unsigned char *d_ref_image = nullptr;
	float *d_cost_cube = nullptr;
	std::vector<unsigned char *> h_src_images;
	std::vector<DeviceCam> h_src_cams;
	h_src_images.reserve(expected_source_count);
	h_src_cams.reserve(expected_source_count);

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ref_image), pixel_count * sizeof(unsigned char)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_cost_cube), cost_count * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_ref_image, ref_y.ptr<unsigned char>(), pixel_count * sizeof(unsigned char), cudaMemcpyHostToDevice));

	const DeviceCam d_ref = make_device_cam(ref);
	const dim3 block(16, 16);
	const dim3 grid(
		(ref.width + block.x - 1) / block.x,
		(ref.height + block.y - 1) / block.y,
		ZPlanes);

	for (auto const &camera : cam_vector)
	{
		if (camera.name == ref.name)
			continue;
		if (camera.YUV.empty())
			throw std::runtime_error("Source camera has no image channels: " + camera.name);

		printf("CUDA upload cam: %s\n", camera.name.c_str());

		const size_t camera_pixel_count = static_cast<size_t>(camera.width) * camera.height;
		cv::Mat cam_y = camera.YUV[0].isContinuous() ? camera.YUV[0] : camera.YUV[0].clone();

		unsigned char *d_cam_image = nullptr;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_cam_image), camera_pixel_count * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(d_cam_image, cam_y.ptr<unsigned char>(), camera_pixel_count * sizeof(unsigned char), cudaMemcpyHostToDevice));

		h_src_images.push_back(d_cam_image);
		h_src_cams.push_back(make_device_cam(camera));
	}

	CUDA_CHECK(cudaMemcpyToSymbol(c_ref_cam, &d_ref, sizeof(DeviceCam)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_src_images, h_src_images.data(), h_src_images.size() * sizeof(unsigned char *)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_src_cams, h_src_cams.data(), h_src_cams.size() * sizeof(DeviceCam)));

	printf("CUDA sweep: %zu source cams, %i depth planes\n", h_src_images.size(), ZPlanes);
	sweep_plane_all_cameras_kernel<<<grid, block>>>(
		d_ref_image,
		d_cost_cube,
		static_cast<int>(h_src_images.size()),
		ZPlanes,
		ZNear,
		ZFar,
		window);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int zi = 0; zi < ZPlanes; ++zi)
	{
		cost_cube[zi] = cv::Mat(ref.height, ref.width, CV_32FC1);
		CUDA_CHECK(cudaMemcpy(
			cost_cube[zi].ptr<float>(),
			d_cost_cube + static_cast<size_t>(zi) * pixel_count,
			pixel_count * sizeof(float),
			cudaMemcpyDeviceToHost));
	}

	CUDA_CHECK(cudaFree(d_ref_image));
	for (unsigned char *image : h_src_images)
		CUDA_CHECK(cudaFree(image));
	CUDA_CHECK(cudaFree(d_cost_cube));

	return cost_cube;
}
