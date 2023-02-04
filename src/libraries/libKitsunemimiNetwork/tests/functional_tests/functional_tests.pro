include(../../defaults.pri)

QT       -= qt core gui

CONFIG -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiNetwork
INCLUDEPATH += $$PWD

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

LIBS += -lssl

HEADERS += \
    cert_init.h \
    libKitsunemimiNetwork/tcp/tcp_test.h \
    libKitsunemimiNetwork/tls_tcp/tls_tcp_test.h \
    libKitsunemimiNetwork/unix/unix_domain_test.h

SOURCES += \
    libKitsunemimiNetwork/tcp/tcp_test.cpp \
    libKitsunemimiNetwork/tls_tcp/tls_tcp_test.cpp \
    libKitsunemimiNetwork/unix/unix_domain_test.cpp \
    main.cpp
