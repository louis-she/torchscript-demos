cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist)

find_package(Torch REQUIRED PATHS ./libtorch)
find_package(OpenCV REQUIRED PATHS ./opencv/build REQUIRED)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")

set_property(TARGET mnist PROPERTY CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
