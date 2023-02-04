QT       -= qt core gui

TARGET = KitsunemimiNetwork
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.8.2

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS +=  -lssl

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiNetwork/abstract_server.h \
    ../include/libKitsunemimiNetwork/abstract_socket.h \
    ../include/libKitsunemimiNetwork/tcp/tcp_server.h \
    ../include/libKitsunemimiNetwork/message_ring_buffer.h \
    ../include/libKitsunemimiNetwork/template_server.h \
    ../include/libKitsunemimiNetwork/template_socket.h \
    ../include/libKitsunemimiNetwork/tls_tcp/tls_tcp_server.h \
    ../include/libKitsunemimiNetwork/tcp/tcp_socket.h \
    ../include/libKitsunemimiNetwork/tls_tcp/tls_tcp_socket.h \
    ../include/libKitsunemimiNetwork/unix/unix_domain_server.h \
    ../include/libKitsunemimiNetwork/unix/unix_domain_socket.h

SOURCES += \
    abstract_server.cpp \
    abstract_socket.cpp \
    tcp/tcp_server.cpp \
    tcp/tcp_socket.cpp \
    tls_tcp/tls_tcp_server.cpp \
    tls_tcp/tls_tcp_socket.cpp \
    unix/unix_domain_socket.cpp \
    unix/unix_domain_server.cpp
