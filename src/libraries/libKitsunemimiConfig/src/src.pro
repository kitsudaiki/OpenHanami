QT -= qt core gui

TARGET = KitsunemimiConfig
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.4.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS += -L../../libKitsunemimiIni/src -lKitsunemimiIni
LIBS += -L../../libKitsunemimiIni/src/debug -lKitsunemimiIni
LIBS += -L../../libKitsunemimiIni/src/release -lKitsunemimiIni
INCLUDEPATH += ../../libKitsunemimiIni/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

SOURCES += \
    config_handler.cpp

HEADERS += \
    ../include/libKitsunemimiConfig/config_handler.h

