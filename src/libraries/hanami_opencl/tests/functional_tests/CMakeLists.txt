qt_add_executable(functional_tests
    main.cpp
    simple_test.cpp simple_test.h
)
target_include_directories(functional_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../../hanami_common/include
    ../../include
    ../../src
)

target_link_libraries(functional_tests PRIVATE
    # Remove: L../../../hanami_common/src
    # Remove: L../../../hanami_common/src/debug
    # Remove: L../../../hanami_common/src/release
    # Remove: L../../src
    # Remove: qt
    OpenCL
    hanami_common
    hanami_opencl
)

install(TARGETS functional_tests
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

qt_generate_deploy_app_script(
    TARGET functional_tests
    FILENAME_VARIABLE deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
