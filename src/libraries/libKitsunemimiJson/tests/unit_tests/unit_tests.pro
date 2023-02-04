include(../../defaults.pri)

QT -= qt core gui

CONFIG -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiJson
INCLUDEPATH += $$PWD

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

SOURCES += \
    main.cpp \
    libKitsunemimiJson/json_item_parseString_test.cpp \
    libKitsunemimiJson/json_item_test.cpp

HEADERS += \
    libKitsunemimiJson/json_item_parseString_test.h \
    libKitsunemimiJson/json_item_test.h

