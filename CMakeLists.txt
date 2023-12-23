cmake_minimum_required(VERSION 3.16)
project(Hanami VERSION 0.3.3 LANGUAGES C CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(src/libraries)
add_subdirectory(src/Hanami)
