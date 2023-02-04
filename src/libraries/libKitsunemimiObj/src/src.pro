QT -= qt core gui

TARGET = KitsunemimiObj
CONFIG += c++17
TEMPLATE = lib
VERSION = 0.2.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

INCLUDEPATH += $$PWD \
            $$PWD/../include

SOURCES += \
    obj_item.cpp \
    obj_parser.cpp \
    obj_creator.cpp

HEADERS += \
    ../include/libKitsunemimiObj/obj_item.h \
    obj_parser.h \
    obj_creator.h
