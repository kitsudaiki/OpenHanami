include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lhanami_sqlite

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

LIBS += -L../../../hanami_json/src -lhanami_json
LIBS += -L../../../hanami_json/src/debug -lhanami_json
LIBS += -L../../../hanami_json/src/release -lhanami_json
INCLUDEPATH += ../../l../ibhanami_json/include


LIBS += -lsqlite3

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp  \
    sqlite_test.cpp

HEADERS += \
    sqlite_test.h
