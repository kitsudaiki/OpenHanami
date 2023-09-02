QT       -= qt core gui

TARGET = KitsunemimiSakuraNetwork
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.8.4

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS +=  -lssl -lcrypt

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiSakuraNetwork/session.h \
    ../include/libKitsunemimiSakuraNetwork/session_controller.h \
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

