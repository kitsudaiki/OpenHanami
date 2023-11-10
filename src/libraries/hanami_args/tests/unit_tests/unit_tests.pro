include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

LIBS += -L../../src -lhanami_args
INCLUDEPATH += $$PWD

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

INCLUDEPATH += ../../../../third-party-libs/json/include

SOURCES += \
    main.cpp \
    arg_parser_test.cpp

HEADERS += \
    arg_parser_test.h
