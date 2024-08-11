add_executable(hanami_config_functional_tests
    config_handler_test.cpp
    config_handler_test.h
    main.cpp
)

target_include_directories(hanami_config_functional_tests PRIVATE
    ../../include/
    ../../src/
)

target_link_libraries(hanami_config_functional_tests PRIVATE
    hanami_config
    hanami_common
)
