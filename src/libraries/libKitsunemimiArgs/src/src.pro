QT -= qt core gui

TARGET = KitsunemimiArgs
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.4.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiArgs/arg_parser.h

SOURCES += \
    arg_parser.cpp

