qt_add_library(hanami_obj
    ../include/hanami_obj/obj_item.h
    obj_creator.cpp obj_creator.h
    obj_item.cpp
    obj_parser.cpp obj_parser.h
)
target_include_directories(hanami_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../hanami_common/include
    ../include
)

target_link_libraries(hanami_obj PUBLIC
    # Remove: L../../hanami_common/src
    # Remove: L../../hanami_common/src/debug
    # Remove: L../../hanami_common/src/release
    # Remove: qt
    hanami_common
)

install(TARGETS hanami_obj
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
