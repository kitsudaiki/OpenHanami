include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

LIBS += -L../../src -lhanami_policies

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

LIBS += -L../../../hanami_args/src -lhanami_args
LIBS += -L../../../hanami_args/src/debug -lhanami_args
LIBS += -L../../../hanami_args/src/release -lhanami_args
INCLUDEPATH += ../../../hanami_args/include

LIBS += -L../../../hanami_ini/src -lhanami_ini
LIBS += -L../../../hanami_ini/src/debug -lhanami_ini
LIBS += -L../../../hanami_ini/src/release -lhanami_ini
INCLUDEPATH += ../../../hanami_ini/include

LIBS += -L../../../hanami_config/src -lhanami_config
LIBS += -L../../../hanami_config/src/debug -lhanami_config
LIBS += -L../../../hanami_config/src/release -lhanami_config
INCLUDEPATH += ../../../hanami_config/include

INCLUDEPATH += ../../../../third-party-libs/json/include

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp  \
    policy_test.cpp

HEADERS += \
    policy_test.h
