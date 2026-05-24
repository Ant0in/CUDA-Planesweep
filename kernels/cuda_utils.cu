
#include "cuda_utils.cuh"
#include "../src/constants.hpp"

#include <iostream>
#include <cstdlib>


namespace cuda_utils {

    /**
     * @brief Checks a CUDA call and exits loudly if it failed.
     * @param err CUDA status code returned by the runtime.
     */
    void checkCuda(cudaError_t err) {

        // so i think in the slides it was done with a macro but I'm not sure I like that tbh
        if (err != cudaSuccess) {
            std::cerr << EA_RED << "[e] CUDA Error: " << cudaGetErrorString(err) << EA_DEFAULT << std::endl;
            exit(EXIT_FAILURE);
        }

    }

    /**
     * @brief Prints the CUDA device info we care about when debugging performance.
     * It is mostly a tiny sanity check that the program is seeing the GPU we think it is seeing.
     */
    void printDeviceInfo() {

        int deviceCount;
        checkCuda(cudaGetDeviceCount(&deviceCount));
        std::cout << "[i] Number of CUDA devices: " << deviceCount << std::endl;

        // for each device I'll print some info, but yeah whatever who's rich enough to have multiple gpus lmao
        // i'm working on a fucking 1060 6gb, what are you expecting like a 5090 or something? im literally broke kekw
        for (int i = 0; i < deviceCount; ++i) {

            cudaDeviceProp deviceProp;
            checkCuda(cudaGetDeviceProperties(&deviceProp, i));

            std::cout << "[i] Device " << i << ": " << deviceProp.name << std::endl;
            std::cout << EA_GRAY << ">> Total Global Memory: " << EA_DEFAULT << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << EA_GRAY << ">> CUDA Cores: " << EA_DEFAULT << deviceProp.multiProcessorCount * 128 << std::endl; // 128 is the number of CUDA cores per SM for a 1060 lol kind of a hack
            std::cout << EA_GRAY << ">> Compute Capability: " << EA_DEFAULT << deviceProp.major << "." << deviceProp.minor << std::endl;
            std::cout << EA_GRAY << ">> Shared Memory per Block: " << EA_DEFAULT << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
            std::cout << EA_GRAY << ">> Registers per Block: " << EA_DEFAULT << deviceProp.regsPerBlock << std::endl;
            std::cout << EA_GRAY << ">> Warp Size: " << EA_DEFAULT << deviceProp.warpSize << std::endl;
            std::cout << EA_GRAY << ">> Max Threads per Block: " << EA_DEFAULT << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << EA_GRAY << ">> Clock Rate: " << EA_DEFAULT << deviceProp.clockRate / 1000 << " MHz" << std::endl;
            std::cout << EA_GRAY << ">> Multi-Processor Count: " << EA_DEFAULT << deviceProp.multiProcessorCount << std::endl;
            std::cout << EA_GRAY << ">> Max Grid Size: " << EA_DEFAULT << "("
                      << deviceProp.maxGridSize[0] << ", "
                      << deviceProp.maxGridSize[1] << ", "
                      << deviceProp.maxGridSize[2] << ")\n"
                      << EA_DEFAULT 
                      << std::endl;

        }

    }

}
