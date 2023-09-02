QT -= qt core gui

TARGET = hanami_obj
CONFIG += c++17
TEMPLATE = lib
VERSION = 0.2.0

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

INCLUDEPATH += $$PWD \
            $$PWD/../include

SOURCES += \
    obj_item.cpp \
    obj_parser.cpp \
    obj_creator.cpp

HEADERS += \
    ../include/hanami_obj/obj_item.h \
    obj_parser.h \
    obj_creator.h
