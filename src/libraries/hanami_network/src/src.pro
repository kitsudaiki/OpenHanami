QT       -= qt core gui

TARGET = hanami_network
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.8.4

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

LIBS +=  -lssl -lcrypt

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/hanami_network/session.h \
    ../include/hanami_network/session_controller.h \
    abstract_server.h \
    abstract_socket.h \
    callbacks.h \
    handler/message_blocker_handler.h \
    handler/reply_handler.h \
    handler/session_handler.h \
    message_definitions.h \
    messages_processing/error_processing.h \
    messages_processing/heartbeat_processing.h \
    messages_processing/multiblock_data_processing.h \
    messages_processing/session_processing.h \
    messages_processing/singleblock_data_processing.h \
    messages_processing/stream_data_processing.h \
    multiblock_io.h \
    tcp/tcp_server.h \
    tcp/tcp_socket.h \
    template_server.h \
    template_socket.h \
    tls_tcp/tls_tcp_server.h \
    tls_tcp/tls_tcp_socket.h \
    unix/unix_domain_server.h \
    unix/unix_domain_socket.h

SOURCES += \
    abstract_server.cpp \
    abstract_socket.cpp \
    handler/message_blocker_handler.cpp \
    handler/reply_handler.cpp \
    handler/session_handler.cpp \
    multiblock_io.cpp \
    session.cpp \
    session_controller.cpp \
    tcp/tcp_server.cpp \
    tcp/tcp_socket.cpp \
    tls_tcp/tls_tcp_server.cpp \
    tls_tcp/tls_tcp_socket.cpp \
    unix/unix_domain_server.cpp \
    unix/unix_domain_socket.cpp

