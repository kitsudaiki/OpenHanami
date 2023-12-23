qt_add_library(hanami_sqlite
    ../include/hanami_sqlite/sqlite.h
    sqlite.cpp
)
target_include_directories(hanami_sqlite PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../../third-party-libs/json/include
    ../../hanami_common/include
    ../include
)

target_link_libraries(hanami_sqlite PUBLIC
    # Remove: L../../hanami_common/src
    # Remove: L../../hanami_common/src/debug
    # Remove: L../../hanami_common/src/release
    # Remove: qt
    SQLite::SQLite3
    hanami_common
)

install(TARGETS hanami_sqlite
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
