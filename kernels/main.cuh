#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

#include "../src/constants.hpp"

// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();

std::vector<cv::Mat> sweeping_plane_cuda(cam const &ref, std::vector<cam> const &cam_vector, int window = 3);
