include(../../defaults.pri)

QT -= qt core gui

CONFIG -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiHanamiClusterParser
INCLUDEPATH += $$PWD

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

SOURCES += \
    cluster_parsestring_test.cpp \
    main.cpp

HEADERS += \
    cluster_parsestring_test.h
