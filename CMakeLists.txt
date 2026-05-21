#------------------------------------------------------------------------------ -
#
#Copyright © 2021 Université Libre de Bruxelles (ULB)
#Copyright © 2021 Universidad Politécnica de Madrid (UPM)
#
#Authors : Jaime Sancho, Daniele Bonatto, Laurie Van Bogaert, Eline Soetens
#Contact : Gauthier.Lafruit@ulb.ac.be
#
#Plane Sweep for INFO-H-503 CUDA project
#This file is used to generate buildable versions of Plane Sweep
#GraphCut code comes from 
#S. Rogge et al., "MPEG-I Depth Estimation Reference Software," 2019 International Conference on 3D #Immersion (IC3D), Brussels, Belgium, 2019, pp. 1-6, doi: 10.1109/IC3D48390.2019.8975995.
#
#Permission is hereby granted, free of charge, to the members of the Moving Picture
#Experts Group (MPEG) obtaining a copy of this software and associated documentation
#files (the "Software"), to use the Software exclusively within the framework of the
#MPEG - I (immersive) and MPEG - I Visual activities, for the sole purpose of
#developing the MPEG - I standard.This permission includes without limitation the
#rights to use, copy, modify and merge copies of the Software, and explicitly
#excludes the rights to publish, distribute, sublicense, sell, embed into a product
#or a service and/or otherwise commercially exploit copies of the Software without
#the written consent of the owner (UPM, ULB).
#
#This permission is provided subject to the following conditions :
#The above copyright notice and this permission notice shall be included in all
#copies, substantial portions or derivative works of this file.
#
#------------------------------------------------------------------------------ -*/


cmake_minimum_required(VERSION 3.17)

set(PROJECT_NAME "PlaneSweep")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# I think it's possible to download OpenCV via CMake, because I remember doing something like that
# from projet 2 in ba2, but I don't think it's very useful here tbh
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source build detected!")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# C++ / CUDA standards (17 in this case)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# CUDA architectures
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Kernels library

file(GLOB kernel_plane_sweep_headers ./kernels/*.cuh)
file(GLOB kernel_plane_sweep_src ./kernels/*.cu)

add_library(${PROJECT_NAME}CUDA
    ${kernel_plane_sweep_headers}
    ${kernel_plane_sweep_src}
)

set_target_properties(${PROJECT_NAME}CUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(${PROJECT_NAME}CUDA PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-lineinfo>
)

target_link_libraries(${PROJECT_NAME}CUDA PRIVATE
    CUDA::cudart
    ${OpenCV_LIBS}
)

target_include_directories(${PROJECT_NAME}CUDA PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# Source files

file(GLOB source_files_planesweep
    ./src/*.cpp
    ./src/*.hpp
    ./src/*.h
)

add_executable(${PROJECT_NAME}
    ${source_files_planesweep}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-lineinfo>
    -Wno-register
    -Wno-write-strings
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PROJECT_NAME}CUDA
    CUDA::cudart
    ${OpenCV_LIBS}
)

# Finally copy some data to the build directory
# this was in the original CMakeLists.txt but it's really not necessary since I copy the binary to the root of
# the project, but I guess it doesn't hurt to keep it here
file(COPY "res" DESTINATION ${CMAKE_BINARY_DIR})

