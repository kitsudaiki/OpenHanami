qt_add_library(hanami_database
    ../include/hanami_database/sql_database.h
    ../include/hanami_database/sql_table.h
    sql_database.cpp
    sql_table.cpp
)
target_include_directories(hanami_database PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../../third-party-libs/json/include
    ../../hanami_common/include
    ../../hanami_sqlite/include
    ../include
)

target_link_libraries(hanami_database PUBLIC
    # Remove: L../../hanami_common/src
    # Remove: L../../hanami_common/src/debug
    # Remove: L../../hanami_common/src/release
    # Remove: L../../hanami_sqlite/src
    # Remove: L../../hanami_sqlite/src/debug
    # Remove: L../../hanami_sqlite/src/release
    # Remove: qt
    SQLite::SQLite3
    hanami_common
    hanami_sqlite
)

install(TARGETS hanami_database
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
