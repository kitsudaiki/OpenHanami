add_executable(hanami_args_unit_tests
    arg_parser_test.cpp
    arg_parser_test.h
    main.cpp
)

target_include_directories(hanami_args_unit_tests PRIVATE
    ../../include/
    ../../src/
)

target_link_libraries(hanami_args_unit_tests PRIVATE
    hanami_args
    hanami_common
)
