include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiSakuraHardware

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

LIBS += -L../../../libKitsunemimiCpu/src -lKitsunemimiCpu
LIBS += -L../../../libKitsunemimiCpu/src/debug -lKitsunemimiCpu
LIBS += -L../../../libKitsunemimiCpu/src/release -lKitsunemimiCpu
INCLUDEPATH += ../../../libKitsunemimiCpu/include

LIBS += -L../../../libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../../libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../../libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../../libKitsunemimiJson/include

INCLUDEPATH += $$PWD

SOURCES += \
    main.cpp 
