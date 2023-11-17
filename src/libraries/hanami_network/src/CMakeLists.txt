qt_add_library(hanami_network
    ../include/hanami_network/session.h
    ../include/hanami_network/session_controller.h
    abstract_server.cpp abstract_server.h
    abstract_socket.cpp abstract_socket.h
    callbacks.h
    handler/message_blocker_handler.cpp handler/message_blocker_handler.h
    handler/reply_handler.cpp handler/reply_handler.h
    handler/session_handler.cpp handler/session_handler.h
    message_definitions.h
    messages_processing/error_processing.h
    messages_processing/heartbeat_processing.h
    messages_processing/multiblock_data_processing.h
    messages_processing/session_processing.h
    messages_processing/singleblock_data_processing.h
    messages_processing/stream_data_processing.h
    multiblock_io.cpp multiblock_io.h
    session.cpp
    session_controller.cpp
    tcp/tcp_server.cpp tcp/tcp_server.h
    tcp/tcp_socket.cpp tcp/tcp_socket.h
    template_server.h
    template_socket.h
    tls_tcp/tls_tcp_server.cpp tls_tcp/tls_tcp_server.h
    tls_tcp/tls_tcp_socket.cpp tls_tcp/tls_tcp_socket.h
    unix/unix_domain_server.cpp unix/unix_domain_server.h
    unix/unix_domain_socket.cpp unix/unix_domain_socket.h
)
target_include_directories(hanami_network PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../../hanami_common/include
    ../include
)

target_link_libraries(hanami_network PUBLIC
    # Remove: L../../hanami_common/src
    # Remove: L../../hanami_common/src/debug
    # Remove: L../../hanami_common/src/release
    # Remove: qt
    crypt
    hanami_common
    ssl
)

install(TARGETS hanami_network
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
