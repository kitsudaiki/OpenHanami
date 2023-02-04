include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include


LIBS +=  -lOpenCL

INCLUDEPATH += $$PWD

LIBS += -L../../src -lKitsunemimiOpencl

SOURCES += \
    main.cpp \
    simple_test.cpp

HEADERS += \
    simple_test.h

