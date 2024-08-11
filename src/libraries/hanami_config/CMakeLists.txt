cmake_minimum_required(VERSION 3.16)
project(hanami_config VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hanami_config STATIC
    include/hanami_config/config_handler.h
    src/config_handler.cpp
)

target_include_directories(hanami_config PUBLIC
    include/
    src/
    ../hanami_common/include
    ../hanami_ini/include
)

target_link_libraries(hanami_config PUBLIC
    hanami_common
    hanami_ini
)

if(run_tests)
    add_subdirectory(tests)
endif()
