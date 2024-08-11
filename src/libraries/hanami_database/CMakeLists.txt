cmake_minimum_required(VERSION 3.16)
project(hanami_database VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hanami_database STATIC
    include/hanami_database/sql_database.h
    include/hanami_database/sql_table.h
    src/sql_database.cpp
    src/sql_table.cpp
)

target_include_directories(hanami_database PUBLIC
    include/
    src/
    ../hanami_common/include
    ../hanami_sqlite/include
)

target_link_libraries(hanami_database PUBLIC
    hanami_common
    hanami_sqlite
    hanami_crypto
)

if(run_tests)
    add_subdirectory(tests)
endif()
