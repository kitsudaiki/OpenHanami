include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++14 console

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

LIBS +=  -lOpenCL

INCLUDEPATH += $$PWD

LIBS += -L../../src -lhanami_opencl

SOURCES += \
    main.cpp \
    simple_test.cpp

HEADERS += \
    simple_test.h

