include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiHanamiHardware

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

LIBS += -luuid

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp 
