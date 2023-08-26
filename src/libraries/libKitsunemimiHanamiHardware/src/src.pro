QT -= qt core gui

TARGET = KitsunemimiHanamiHardware
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.1.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS += -L../../libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../libKitsunemimiJson/include

LIBS += -L../../libKitsunemimiCpu/src -lKitsunemimiCpu
LIBS += -L../../libKitsunemimiCpu/src/debug -lKitsunemimiCpu
LIBS += -L../../libKitsunemimiCpu/src/release -lKitsunemimiCpu
INCLUDEPATH += ../../libKitsunemimiCpu/include

LIBS += -L../../libKitsunemimiSakuraHardware/src -lKitsunemimiSakuraHardware
LIBS += -L../../libKitsunemimiSakuraHardware/src/debug -lKitsunemimiSakuraHardware
LIBS += -L../../libKitsunemimiSakuraHardware/src/release -lKitsunemimiSakuraHardware
INCLUDEPATH += ../../libKitsunemimiSakuraHardware/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiHanamiHardware/power_measuring.h \
    ../include/libKitsunemimiHanamiHardware/speed_measuring.h \
    ../include/libKitsunemimiHanamiHardware/temperature_measuring.h \
    ../include/libKitsunemimiHanamiHardware/value_container.h

SOURCES += \
    power_measuring.cpp \
    speed_measuring.cpp \
    temperature_measuring.cpp \
    value_container.cpp
