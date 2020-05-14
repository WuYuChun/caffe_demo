cmake_minimum_required(VERSION 2.8)

project(Caffe_Demo)

FIND_PACKAGE(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
#find_package(Glog REQUIRED)

include_directories(/usr/local/cuda-8.0/targets/x86_64-linux/include)

include_directories(${PROJECT_SOURCE_DIR}/third_lib/caffe_include/include)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

link_directories(/workspace/caffe_demo/third_lib/caffe_lib/lib)
link_directories(/usr/lib/x86_64-linux-gnu)

message("----debug: ${PROJECT_SOURCE_DIR}")

#link_directories(${CUDA_PATH}/lib/x64)
#link_directories(${PROJECT_SOURCE_DIR}/lib)

SET(GLOG_LIBRARIES ${OpenCV_LIBS})



add_executable(semantic_division src/semantic_division_test.cpp src/semantic_division.cpp )
target_link_libraries(semantic_division 
                      ${OpenCV_LIBS} caffe ${Boost_FILESYSTEM_LIBRARY} 
                       ${Boost_SYSTEM_LIBRARY} glog)
