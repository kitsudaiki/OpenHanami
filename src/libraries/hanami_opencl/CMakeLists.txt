cmake_minimum_required(VERSION 3.16)
project(hanami_opencl VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)






add_subdirectory(src)
if(run_tests)
    add_subdirectory(tests)
endif()
