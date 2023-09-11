include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lhanami_sqlite

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

LIBS += -lsqlite3

INCLUDEPATH += ../../../../third-party-libs/json/include

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp  \
    sqlite_test.cpp

HEADERS += \
    sqlite_test.h
