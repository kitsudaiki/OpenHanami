QT -= qt core gui

TARGET = hanami_hardware
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.1.0

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

LIBS += -L../../hanami_json/src -lhanami_json
LIBS += -L../../hanami_json/src/debug -lhanami_json
LIBS += -L../../hanami_json/src/release -lhanami_json
INCLUDEPATH += ../../hanami_json/include

LIBS += -L../../hanami_cpu/src -lhanami_cpu
LIBS += -L../../hanami_cpu/src/debug -lhanami_cpu
LIBS += -L../../hanami_cpu/src/release -lhanami_cpu
INCLUDEPATH += ../../hanami_cpu/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/hanami_hardware/cpu_core.h \
    ../include/hanami_hardware/cpu_package.h \
    ../include/hanami_hardware/cpu_thread.h \
    ../include/hanami_hardware/host.h \
    ../include/hanami_hardware/power_measuring.h \
    ../include/hanami_hardware/speed_measuring.h \
    ../include/hanami_hardware/temperature_measuring.h \
    ../include/hanami_hardware/value_container.h

SOURCES += \
    cpu_core.cpp \
    cpu_package.cpp \
    cpu_thread.cpp \
    host.cpp \
    power_measuring.cpp \
    speed_measuring.cpp \
    temperature_measuring.cpp \
    value_container.cpp

