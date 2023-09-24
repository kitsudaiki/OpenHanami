include(../../defaults.pri)

QT -= qt core gui

CONFIG -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lhanami_cluster_parser
INCLUDEPATH += $$PWD

LIBS += -L../../../hanami_common/src -lhanami_common
LIBS += -L../../../hanami_common/src/debug -lhanami_common
LIBS += -L../../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../hanami_common/include

INCLUDEPATH += ../../../../third-party-libs/json/include

SOURCES += \
    cluster_parsestring_test.cpp \
    main.cpp

HEADERS += \
    cluster_parsestring_test.h
