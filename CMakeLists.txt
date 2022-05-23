cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(custom_average)

find_package(Torch REQUIRED)

add_library(custom_average inference/src/custom_average.cpp)
target_link_libraries(custom_average "${TORCH_LIBRARIES}")
set_property(TARGET custom_average PROPERTY CXX_STANDARD 14)
