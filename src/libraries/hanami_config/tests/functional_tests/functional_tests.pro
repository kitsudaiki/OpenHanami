include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lhanami_config

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

LIBS += -L../../../hanami_ini/src -lhanami_ini
LIBS += -L../../../hanami_ini/src/debug -lhanami_ini
LIBS += -L../../../hanami_ini/src/release -lhanami_ini
INCLUDEPATH += ../../../hanami_ini/include

INCLUDEPATH += ../../../../third-party-libs/json/include

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp \
    config_handler_test.cpp

HEADERS += \
    config_handler_test.h
