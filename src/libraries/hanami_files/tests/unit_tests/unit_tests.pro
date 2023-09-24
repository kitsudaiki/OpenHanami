include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lhanami_files

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

INCLUDEPATH += ../../../../third-party-libs/json/include

LIBS += -luuid

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp 
