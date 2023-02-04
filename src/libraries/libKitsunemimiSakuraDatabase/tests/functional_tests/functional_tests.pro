include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiSakuraDatabase

LIBS += -L../../../libKitsunemimiSqlite/src -lKitsunemimiSqlite
LIBS += -L../../../libKitsunemimiSqlite/src/debug -lKitsunemimiSqlite
LIBS += -L../../../libKitsunemimiSqlite/src/release -lKitsunemimiSqlite
INCLUDEPATH += ../../../libKitsunemimiSqlite/include

LIBS += -L../../../libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../../libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../../libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../../libKitsunemimiJson/include

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

LIBS += -lsqlite3

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp  \
    sql_table_test.cpp \
    test_table.cpp

HEADERS += \
    sql_table_test.h \
    test_table.h
