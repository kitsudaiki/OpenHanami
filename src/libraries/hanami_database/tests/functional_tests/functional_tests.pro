include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lhanami_database

LIBS += -L../../../hanami_sqlite/src -lhanami_sqlite
LIBS += -L../../../hanami_sqlite/src/debug -lhanami_sqlite
LIBS += -L../../../hanami_sqlite/src/release -lhanami_sqlite
INCLUDEPATH += ../../../hanami_sqlite/include

LIBS += -L../../../hanami_json/src -lhanami_json
LIBS += -L../../../hanami_json/src/debug -lhanami_json
LIBS += -L../../../hanami_json/src/release -lhanami_json
INCLUDEPATH += ../../../hanami_json/include

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

LIBS += -lsqlite3

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp  \
    sql_table_test.cpp \
    test_table.cpp

HEADERS += \
    sql_table_test.h \
    test_table.h
