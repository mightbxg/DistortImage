cmake_minimum_required(VERSION 3.10)

project(DistortImage)

find_package(OpenCV)

link_libraries(${OpenCV_LIBS})
add_executable(calibrate calibrate.cpp)
add_executable(undistort undistort.cpp)
add_executable(distort distort.cpp)
