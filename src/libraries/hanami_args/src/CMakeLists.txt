qt_add_library(hanami_args
    ../include/hanami_args/arg_parser.h
    arg_parser.cpp
)
target_include_directories(hanami_args PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../../third-party-libs/json/include
    ../../hanami_common/include
    ../include
)

target_link_libraries(hanami_args PUBLIC
    # Remove: L../../hanami_common/src
    # Remove: L../../hanami_common/src/debug
    # Remove: L../../hanami_common/src/release
    # Remove: qt
    hanami_common
)

install(TARGETS hanami_args
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
