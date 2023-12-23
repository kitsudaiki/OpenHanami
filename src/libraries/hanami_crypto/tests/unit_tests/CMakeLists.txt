add_executable(hanami_crypto_unit_tests
    common_test.cpp
    common_test.h
    hashes_test.cpp
    hashes_test.h
    main.cpp
    symmetric_encryption_test.cpp
    symmetric_encryption_test.h
)

target_include_directories(hanami_crypto_unit_tests PRIVATE
    ../../../../third-party-libs/json/include
    ../../../hanami_common/include
    ../../include
    ../../src
)

target_link_libraries(hanami_crypto_unit_tests PRIVATE
    crypto
    cryptopp
    hanami_common
    hanami_crypto
    ssl
)
