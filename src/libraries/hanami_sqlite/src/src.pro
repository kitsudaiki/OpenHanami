QT -= qt core gui

TARGET = hanami_sqlite
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.3.0

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

LIBS += -L../../hanami_json/src -lhanami_json
LIBS += -L../../hanami_json/src/debug -lhanami_json
LIBS += -L../../hanami_json/src/release -lhanami_json
INCLUDEPATH += ../../hanami_json/include


LIBS += -lsqlite3

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/hanami_sqlite/sqlite.h

SOURCES += \
    sqlite.cpp

