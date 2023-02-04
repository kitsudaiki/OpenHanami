QT -= qt core gui

TARGET = KitsunemimiSakuraDatabase
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.5.0

LIBS += -L../../libKitsunemimiSqlite/src -lKitsunemimiSqlite
LIBS += -L../../libKitsunemimiSqlite/src/debug -lKitsunemimiSqlite
LIBS += -L../../libKitsunemimiSqlite/src/release -lKitsunemimiSqlite
INCLUDEPATH += ../../libKitsunemimiSqlite/include

LIBS += -L../../libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../libKitsunemimiJson/include

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS += -lsqlite3

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiSakuraDatabase/sql_table.h \
    ../include/libKitsunemimiSakuraDatabase/sql_database.h

SOURCES += \
    sql_database.cpp \
    sql_table.cpp

