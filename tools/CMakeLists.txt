cmake_minimum_required(VERSION 2.8)

message("---debug: ${OpenCV_LIBS}")
add_executable(read_mnist_data_to_img read_mnist_data_to_img.cpp)
target_link_libraries(read_mnist_data_to_img ${OpenCV_LIBS} )   