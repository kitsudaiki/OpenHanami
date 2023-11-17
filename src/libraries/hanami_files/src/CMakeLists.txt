qt_add_library(hanami_files
    ../include/hanami_files/data_set_files/data_set_file.h
    ../include/hanami_files/data_set_files/data_set_functions.h
    ../include/hanami_files/data_set_files/image_data_set_file.h
    ../include/hanami_files/data_set_files/table_data_set_file.h
    data_set_files/data_set_file.cpp
    data_set_files/data_set_functions.cpp
    data_set_files/image_data_set_file.cpp
    data_set_files/table_data_set_file.cpp
)
target_include_directories(hanami_files PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../../third-party-libs/json/include
    ../../hanami_common/include
    ../include
)

target_link_libraries(hanami_files PUBLIC
    # Remove: L../../hanami_common/src
    # Remove: L../../hanami_common/src/debug
    # Remove: L../../hanami_common/src/release
    # Remove: qt
    hanami_common
)

install(TARGETS hanami_files
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
