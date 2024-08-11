cmake_minimum_required(VERSION 3.16)
project(hanami_crypto VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hanami_crypto STATIC
    include/hanami_crypto/common.h
    include/hanami_crypto/hashes.h
    include/hanami_crypto/signing.h
    include/hanami_crypto/symmetric_encryption.h
    src/common.cpp
    src/hashes.cpp
    src/symmetric_encryption.cpp
)

target_include_directories(hanami_crypto PUBLIC
    include/
    src/
    ../hanami_common/include
)

target_link_libraries(hanami_crypto PUBLIC
    crypto
    cryptopp
    hanami_common
    ssl
)

if(run_tests)
    add_subdirectory(tests)
endif()
