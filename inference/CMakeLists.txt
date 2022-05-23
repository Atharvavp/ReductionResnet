cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(custom_average)

find_package(Torch REQUIRED)

add_library(custom_average src/custom_average.cpp)
target_link_libraries(custom_average "${TORCH_LIBRARIES}")
set_property(TARGET custom_average PROPERTY CXX_STANDARD 14)
set_target_properties(custom_average PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
