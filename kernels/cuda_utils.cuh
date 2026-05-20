
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>


/**
 * This namespace contains utility functions for CUDA operations, such as error checking and device information retrieval.
 * It also contains a DeviceBuffer class that manages memory on the GPU, which is something I like to use to avoid bad
 * practices (im a forgetful goober)
 * I believe using a namespace is a good practice, usually I would put that in a class but that got me yelled at by some prof 
 * which I'm not sure who that was lol. Maybe was it f307? idk
 */
namespace cuda_utils {

    /**
     * @brief Checks the result of a CUDA operation and prints an error message if it failed.
     * @param err The error code returned by a CUDA operation.
     */
    void checkCuda(cudaError_t err);

    
    /**
     * @brief Prints information about the CUDA device. Not very useful but it's so damn pretty uwu >w<
     */
    void printDeviceInfo();


    template<typename T>
    class DeviceBuffer {

        private:

            T *ptr_ = nullptr;

            /**
             * @brief Frees the owned GPU allocation if there is one.
             * Small private cleanup helper, mostly here so my memory does not leak when I forgor L me
             */
            void reset() {

                if (ptr_) {
                    checkCuda(cudaFree(ptr_));
                    ptr_ = nullptr;
                }

            }

        public:

            /**
             * @brief Empty buffer, no allocation yet.
             */
            DeviceBuffer() = default;

            /**
             * @brief Allocates a device buffer with count elements right away.
             * @param count Number of T elements to allocate on the GPU.
             */
            explicit DeviceBuffer(size_t count) {allocate(count);}

            /**
             * @brief Releases the device allocation when the wrapper goes out of scope.
             */
            ~DeviceBuffer() {reset();}

            /**
             * @brief Copying is disabled because two owners for one cudaMalloc sounds like pain.
             */
            DeviceBuffer(DeviceBuffer const &) = delete;

            /**
             * @brief Copy assignment is disabled for the same ownership reason as above.
             */
            DeviceBuffer& operator=(DeviceBuffer const &) = delete;

            /**
             * @brief Moves ownership from another buffer without copying GPU memory.
             * @param other Buffer we steal from, politely.
             */
            DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_) {other.ptr_ = nullptr;}

            /**
             * @brief Move-assigns another buffer, freeing the current one first if needed.
             * @param other Buffer we take ownership from.
             * @return This buffer, now owning the other pointer.
             */
            DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {

                if (this != &other) {
                    reset();
                    ptr_ = other.ptr_;
                    other.ptr_ = nullptr;
                }

                return *this;
            }
            
            /**
             * @brief Allocates count elements on the GPU, replacing any old allocation.
             * @param count Number of T elements to allocate.
             */
            void allocate(size_t count) {
                reset();
                checkCuda(cudaMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(T)));
            }

            /**
             * @brief Returns the raw device pointer for CUDA calls.
             * @return The owned GPU pointer, or nullptr if empty.
             */
            T *get() const {return ptr_;}
            
    };

}
