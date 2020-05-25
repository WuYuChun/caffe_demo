cmake_minimum_required(VERSION 2.8)


add_executable(semantic_division_test semantic_division_test.cpp  )
target_link_libraries(semantic_division_test semantic_division
                      ${OpenCV_LIBS} caffe ${Boost_FILESYSTEM_LIBRARY} 
                       ${Boost_SYSTEM_LIBRARY} glog)

add_executable(vgg16_test vgg16_test.cpp )
target_link_libraries(vgg16_test vgg_16
                      ${OpenCV_LIBS} caffe ${Boost_FILESYSTEM_LIBRARY} 
                      ${Boost_SYSTEM_LIBRARY} glog)    
                      
add_executable(parsing_test parsing_test.cpp )
target_link_libraries(parsing_test vgg_16
                      ${OpenCV_LIBS} caffe ${Boost_FILESYSTEM_LIBRARY} 
                                            ${Boost_SYSTEM_LIBRARY} glog)                          

add_executable(psd_test psd_test.cpp )
target_link_libraries(psd_test psd common
                      ${OpenCV_LIBS} caffe ${Boost_FILESYSTEM_LIBRARY} 
                                            ${Boost_SYSTEM_LIBRARY} glog)     


                                            
