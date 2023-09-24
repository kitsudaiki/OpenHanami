QT -= qt core gui

TARGET = hanami_config
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.4.0

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

LIBS += -L../../hanami_ini/src -lhanami_ini
LIBS += -L../../hanami_ini/src/debug -lhanami_ini
LIBS += -L../../hanami_ini/src/release -lhanami_ini
INCLUDEPATH += ../../hanami_ini/include

INCLUDEPATH += ../../../third-party-libs/json/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

SOURCES += \
    config_handler.cpp

HEADERS += \
    ../include/hanami_config/config_handler.h

