QT -= qt core gui

TARGET = KitsunemimiOpencl
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.4.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS +=  -lOpenCL


INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiOpencl/gpu_interface.h \
    ../include/libKitsunemimiOpencl/gpu_handler.h \
    ../include/libKitsunemimiOpencl/gpu_data.h

SOURCES += \
    gpu_interface.cpp \
    gpu_handler.cpp \
    gpu_data.cpp
