QT -= qt core gui

TARGET = KitsunemimiHanamiDatabase
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.3.0

LIBS += -L../../libKitsunemimiHanamiCommon/src -lKitsunemimiHanamiCommon
LIBS += -L../../libKitsunemimiHanamiCommon/src/debug -lKitsunemimiHanamiCommon
LIBS += -L../../libKitsunemimiHanamiCommon/src/release -lKitsunemimiHanamiCommon
INCLUDEPATH += ../../libKitsunemimiHanamiCommon/include

LIBS += -L../../libKitsunemimiSakuraDatabase/src -lKitsunemimiSakuraDatabase
LIBS += -L../../libKitsunemimiSakuraDatabase/src/debug -lKitsunemimiSakuraDatabase
LIBS += -L../../libKitsunemimiSakuraDatabase/src/release -lKitsunemimiSakuraDatabase
INCLUDEPATH += ../../libKitsunemimiSakuraDatabase/include

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

LIBS += -lsqlite3 -luuid

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiHanamiDatabase/hanami_sql_admin_table.h \
    ../include/libKitsunemimiHanamiDatabase/hanami_sql_log_table.h \
    ../include/libKitsunemimiHanamiDatabase/hanami_sql_table.h

SOURCES += \
    hanami_sql_admin_table.cpp \
    hanami_sql_log_table.cpp \
    hanami_sql_table.cpp

