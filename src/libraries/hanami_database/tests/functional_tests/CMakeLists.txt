add_executable(hanami_database_functional_tests
    sql_table_test.cpp
    sql_table_test.h
    test_table.cpp
    test_table.h
    main.cpp
)

target_include_directories(hanami_database_functional_tests PRIVATE
    ../../include/
    ../../src/
    ../../../../third-party-libs/json/include
)

target_link_libraries(hanami_database_functional_tests PRIVATE
    hanami_database
    hanami_common
)
