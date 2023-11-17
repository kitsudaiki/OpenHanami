cmake_minimum_required(VERSION 3.16)
project(hanami_args VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hanami_args STATIC
    include/hanami_args/arg_parser.h
    src/arg_parser.cpp
)

target_include_directories(hanami_args PUBLIC
    include/
    src/
    ../hanami_common/include
    ../../third-party-libs/json/include
)

target_link_libraries(hanami_args PUBLIC
    hanami_common
)

if(run_tests)
    add_subdirectory(tests)
endif()
