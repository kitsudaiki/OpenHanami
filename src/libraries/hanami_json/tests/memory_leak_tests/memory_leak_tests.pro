include(../../defaults.pri)

QT -= qt core gui

CONFIG -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lhanami_json
INCLUDEPATH += $$PWD

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

SOURCES += \
    main.cpp \
    hanami_json/json_item_parseString_test.cpp \
    hanami_json/json_item_test.cpp

HEADERS += \
    hanami_json/json_item_parseString_test.h \
    hanami_json/json_item_test.h

