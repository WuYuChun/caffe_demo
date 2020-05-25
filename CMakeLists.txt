cmake_minimum_required(VERSION 2.8)

project(Caffe_Demo)

FIND_PACKAGE(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

add_definitions(-std=c++11)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CUDA_NVCC_FLAGS -G;-g)

include_directories(/usr/local/cuda-8.0/targets/x86_64-linux/include)

include_directories(${PROJECT_SOURCE_DIR}/third_lib/caffe_include/include)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/common)

link_directories(${PROJECT_SOURCE_DIR}/third_lib/caffe_lib/lib)
link_directories(/usr/lib/x86_64-linux-gnu)

add_subdirectory(tools)
add_subdirectory(common)
add_subdirectory(src)
add_subdirectory(test)
                                            
