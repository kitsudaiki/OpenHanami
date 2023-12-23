qt_add_library(hanami_config
    ../include/hanami_config/config_handler.h
    config_handler.cpp
)
target_include_directories(hanami_config PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../../third-party-libs/json/include
    ../../hanami_common/include
    ../../hanami_ini/include
    ../include
)

target_link_libraries(hanami_config PUBLIC
    # Remove: L../../hanami_common/src
    # Remove: L../../hanami_common/src/debug
    # Remove: L../../hanami_common/src/release
    # Remove: L../../hanami_ini/src
    # Remove: L../../hanami_ini/src/debug
    # Remove: L../../hanami_ini/src/release
    # Remove: qt
    hanami_common
    hanami_ini
)

install(TARGETS hanami_config
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
