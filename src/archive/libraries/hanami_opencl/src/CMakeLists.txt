qt_add_library(hanami_opencl
    ../include/hanami_opencl/gpu_data.h
    ../include/hanami_opencl/gpu_handler.h
    ../include/hanami_opencl/gpu_interface.h
    gpu_data.cpp
    gpu_handler.cpp
    gpu_interface.cpp
)
target_include_directories(hanami_opencl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../hanami_common/include
    ../include
)

target_link_libraries(hanami_opencl PUBLIC
    # Remove: L../../hanami_common/src
    # Remove: L../../hanami_common/src/debug
    # Remove: L../../hanami_common/src/release
    # Remove: qt
    OpenCL
    hanami_common
)

install(TARGETS hanami_opencl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
