QT -= qt core gui

TARGET = KitsunemimiCpu
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.3.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiCpu/cpu.h \
    ../include/libKitsunemimiCpu/memory.h \
    ../include/libKitsunemimiCpu/rapl.h

SOURCES += \
    cpu.cpp \
    memory.cpp \
    rapl.cpp

