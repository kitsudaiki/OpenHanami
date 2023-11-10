include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

LIBS += -L../../src -lhanami_crypto

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

LIBS +=  -lssl -lcrypto -lcryptopp

INCLUDEPATH += ../../../../third-party-libs/json/include

INCLUDEPATH += $$PWD

SOURCES += \
    common_test.cpp \
    hashes_test.cpp \
    main.cpp  \
    symmetric_encryption_test.cpp

HEADERS += \
    common_test.h \
    hashes_test.h \
    symmetric_encryption_test.h
