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

include_directories(/usr/local/cuda-8.0/targets/x86_64-linux/include)

include_directories(${PROJECT_SOURCE_DIR}/third_lib/caffe_include/include)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

link_directories(${PROJECT_SOURCE_DIR}/third_lib/caffe_lib/lib)
link_directories(/usr/lib/x86_64-linux-gnu)

add_library(semantic_division SHARED src/semantic_division.cpp)
add_library(vgg_16 SHARED src/vgg16.cpp)
cuda_add_library(psd SHARED src/psd.cpp src/cuda_function.cu)


add_executable(semantic_division_test test/semantic_division_test.cpp  )
target_link_libraries(semantic_division_test semantic_division
                      ${OpenCV_LIBS} caffe ${Boost_FILESYSTEM_LIBRARY} 
                       ${Boost_SYSTEM_LIBRARY} glog)

add_executable(vgg16_test test/vgg16_test.cpp )
target_link_libraries(vgg16_test vgg_16
                      ${OpenCV_LIBS} caffe ${Boost_FILESYSTEM_LIBRARY} 
                      ${Boost_SYSTEM_LIBRARY} glog)    
                      
add_executable(parsing_test test/parsing_test.cpp )
target_link_libraries(parsing_test vgg_16
                      ${OpenCV_LIBS} caffe ${Boost_FILESYSTEM_LIBRARY} 
                                            ${Boost_SYSTEM_LIBRARY} glog)                          

add_executable(psd_test test/psd_test.cpp )
target_link_libraries(psd_test psd
                      ${OpenCV_LIBS} caffe ${Boost_FILESYSTEM_LIBRARY} 
                                            ${Boost_SYSTEM_LIBRARY} glog)     


add_subdirectory(tools)                                            
