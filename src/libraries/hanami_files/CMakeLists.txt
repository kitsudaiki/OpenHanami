cmake_minimum_required(VERSION 3.16)
project(hanami_files VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hanami_files STATIC
    include/hanami_files/dataset_files/dataset_file.h
    include/hanami_files/dataset_files/dataset_functions.h
    include/hanami_files/dataset_files/image_dataset_file.h
    include/hanami_files/dataset_files/table_dataset_file.h
    src/dataset_files/dataset_file.cpp
    src/dataset_files/dataset_functions.cpp
    src/dataset_files/image_dataset_file.cpp
    src/dataset_files/table_dataset_file.cpp
)

target_include_directories(hanami_files PUBLIC
    include/
    src/
    ../hanami_common/include
    ../../third-party-libs/json/include
)

target_link_libraries(hanami_files PUBLIC
    hanami_common
)
