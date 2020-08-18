project(gpu_acceleration)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA)

set(CUDA_NVCC_FLAGS -O3;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(gpu_acceleration STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
