add_executable(hanami_policies_unit_tests
    policy_test.cpp
    policy_test.h
    main.cpp
)

target_include_directories(hanami_policies_unit_tests PRIVATE
    ../../include/
    ../../src/
)

target_link_libraries(hanami_policies_unit_tests PRIVATE
    hanami_policies
    hanami_common
)
