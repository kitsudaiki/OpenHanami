add_executable(hanami_sqlite_unit_tests
    sqlite_test.cpp
    sqlite_test.h
    main.cpp
)

target_include_directories(hanami_sqlite_unit_tests PRIVATE
    ../../include/
    ../../src/
)

target_link_libraries(hanami_sqlite_unit_tests PRIVATE
    hanami_sqlite
    hanami_common
)
