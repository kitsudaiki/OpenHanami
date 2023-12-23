cmake_minimum_required(VERSION 3.16)
project(hanami_sqlite VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hanami_sqlite STATIC
    include/hanami_sqlite/sqlite.h
    src/sqlite.cpp
)

target_include_directories(hanami_sqlite PUBLIC
    include/
    src/
    ../hanami_common/include
    ../../third-party-libs/json/include
)

target_link_libraries(hanami_sqlite PUBLIC
    hanami_common
    sqlite3
)

if(run_tests)
    add_subdirectory(tests)
endif()
