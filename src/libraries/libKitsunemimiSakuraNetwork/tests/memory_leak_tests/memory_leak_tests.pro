include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiSakuraNetwork
INCLUDEPATH += $$PWD

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

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
