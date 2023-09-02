QT -= qt core gui

TARGET = hanami_opencl
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.4.0

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

LIBS +=  -lOpenCL


INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/hanami_opencl/gpu_interface.h \
    ../include/hanami_opencl/gpu_handler.h \
    ../include/hanami_opencl/gpu_data.h

SOURCES += \
    gpu_interface.cpp \
    gpu_handler.cpp \
    gpu_data.cpp
