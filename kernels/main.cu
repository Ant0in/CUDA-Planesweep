#include "main.cuh"

#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>

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

__global__ void init_cost_cube(float *cost_cube, size_t count, float value)
{
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count)
		cost_cube[i] = value;
}

__global__ void sweep_plane_kernel(
	unsigned char const *ref_image,
	unsigned char const *cam_image,
	float *cost_cube,
	DeviceCam ref,
	DeviceCam src,
	int z_planes,
	float z_near,
	float z_far,
	int window)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int zi = blockIdx.z;

	if (x >= ref.width || y >= ref.height || zi >= z_planes)
		return;

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

	float cost = 0.0f;
	float samples = 0.0f;
	const int radius = window / 2;

	for (int k = -radius; k <= radius; ++k)
	{
		for (int l = -radius; l <= radius; ++l)
		{
			const int ref_x = x + l;
			const int ref_y = y + k;
			const int src_x = static_cast<int>(x_proj) + l;
			const int src_y = static_cast<int>(y_proj) + k;

			if (ref_x < 0 || ref_x >= ref.width)
				continue;
			if (ref_y < 0 || ref_y >= ref.height)
				continue;
			if (src_x < 0 || src_x >= src.width)
				continue;
			if (src_y < 0 || src_y >= src.height)
				continue;

			const float ref_value = static_cast<float>(ref_image[ref_y * ref.width + ref_x]);
			const float src_value = static_cast<float>(cam_image[src_y * src.width + src_x]);
			cost += fabsf(ref_value - src_value);
			samples += 1.0f;
		}
	}

	cost = samples > 0.0f ? cost / samples : 255.0f;

	const size_t idx = (static_cast<size_t>(zi) * ref.height + y) * ref.width + x;
	cost_cube[idx] = fminf(cost_cube[idx], cost);
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

	const size_t pixel_count = static_cast<size_t>(ref.width) * ref.height;
	const size_t cost_count = pixel_count * ZPlanes;

	cv::Mat ref_y = ref.YUV[0].isContinuous() ? ref.YUV[0] : ref.YUV[0].clone();

	unsigned char *d_ref_image = nullptr;
	unsigned char *d_cam_image = nullptr;
	float *d_cost_cube = nullptr;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ref_image), pixel_count * sizeof(unsigned char)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_cost_cube), cost_count * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_ref_image, ref_y.ptr<unsigned char>(), pixel_count * sizeof(unsigned char), cudaMemcpyHostToDevice));

	const int init_block = 256;
	const int init_grid = static_cast<int>((cost_count + init_block - 1) / init_block);
	init_cost_cube<<<init_grid, init_block>>>(d_cost_cube, cost_count, 255.0f);
	CUDA_CHECK(cudaGetLastError());

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

		printf("CUDA Cam: %s\n", camera.name.c_str());

		const size_t camera_pixel_count = static_cast<size_t>(camera.width) * camera.height;
		cv::Mat cam_y = camera.YUV[0].isContinuous() ? camera.YUV[0] : camera.YUV[0].clone();

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_cam_image), camera_pixel_count * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(d_cam_image, cam_y.ptr<unsigned char>(), camera_pixel_count * sizeof(unsigned char), cudaMemcpyHostToDevice));

		sweep_plane_kernel<<<grid, block>>>(
			d_ref_image,
			d_cam_image,
			d_cost_cube,
			d_ref,
			make_device_cam(camera),
			ZPlanes,
			ZNear,
			ZFar,
			window);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaFree(d_cam_image));
		d_cam_image = nullptr;
	}

	std::vector<float> h_cost_cube(cost_count);
	CUDA_CHECK(cudaMemcpy(h_cost_cube.data(), d_cost_cube, cost_count * sizeof(float), cudaMemcpyDeviceToHost));

	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int zi = 0; zi < ZPlanes; ++zi)
	{
		cost_cube[zi] = cv::Mat(ref.height, ref.width, CV_32FC1);
		std::memcpy(
			cost_cube[zi].ptr<float>(),
			h_cost_cube.data() + static_cast<size_t>(zi) * pixel_count,
			pixel_count * sizeof(float));
	}

	CUDA_CHECK(cudaFree(d_ref_image));
	CUDA_CHECK(cudaFree(d_cost_cube));

	return cost_cube;
}
