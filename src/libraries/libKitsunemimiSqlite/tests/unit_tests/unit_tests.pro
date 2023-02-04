include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiSqlite

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

LIBS += -L../../../libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../../libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../../libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../l../ibKitsunemimiJson/include


LIBS += -lsqlite3

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp  \
    sqlite_test.cpp

HEADERS += \
    sqlite_test.h
