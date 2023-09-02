include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lhanami_network
INCLUDEPATH += $$PWD

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

LIBS +=  -lssl -lcrypt

SOURCES += \
    main.cpp \
    session_test.cpp \
    tcp/tcp_test.cpp \
    unix/unix_domain_test.cpp

HEADERS += \
    session_test.h \
    cert_init.h \
    tcp/tcp_test.h \
    unix/unix_domain_test.h
