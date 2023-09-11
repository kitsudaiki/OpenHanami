QT -= qt core gui

TARGET = hanami_args
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.4.0

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

INCLUDEPATH += ../../../third-party-libs/json/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/hanami_args/arg_parser.h

SOURCES += \
    arg_parser.cpp

