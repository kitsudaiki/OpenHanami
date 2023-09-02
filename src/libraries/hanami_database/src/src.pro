QT -= qt core gui

TARGET = hanami_database
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.5.0

LIBS += -L../../hanami_sqlite/src -lhanami_sqlite
LIBS += -L../../hanami_sqlite/src/debug -lhanami_sqlite
LIBS += -L../../hanami_sqlite/src/release -lhanami_sqlite
INCLUDEPATH += ../../hanami_sqlite/include

LIBS += -L../../hanami_json/src -lhanami_json
LIBS += -L../../hanami_json/src/debug -lhanami_json
LIBS += -L../../hanami_json/src/release -lhanami_json
INCLUDEPATH += ../../hanami_json/include

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

LIBS += -lsqlite3

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/hanami_database/sql_table.h \
    ../include/hanami_database/sql_database.h

SOURCES += \
    sql_database.cpp \
    sql_table.cpp

