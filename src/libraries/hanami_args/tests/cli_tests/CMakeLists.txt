add_executable(hanami_args_cli_tests
    main.cpp
)

target_include_directories(hanami_args_cli_tests PRIVATE
    ../../include/
    ../../src/
)

target_link_libraries(hanami_args_cli_tests PRIVATE
    hanami_args
    hanami_common
)
