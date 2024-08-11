add_executable(hanami_ini_unit_tests
    hanami_ini/ini_item_test.cpp
    hanami_ini/ini_item_test.h
    main.cpp
)

target_include_directories(hanami_ini_unit_tests PRIVATE
    ../../include/
    ../../src/
)

target_link_libraries(hanami_ini_unit_tests PRIVATE
    hanami_ini
    hanami_common
)

