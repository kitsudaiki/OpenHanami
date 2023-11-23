QT -= qt core gui

TARGET = hanami_files
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.1.0

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

INCLUDEPATH += ../../../third-party-libs/json/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

SOURCES += \
    dataset_files/dataset_file.cpp \
    dataset_files/dataset_functions.cpp \
    dataset_files/image_dataset_file.cpp \
    dataset_files/table_dataset_file.cpp

HEADERS += \
    ../include/hanami_files/dataset_files/dataset_file.h \
    ../include/hanami_files/dataset_files/dataset_functions.h \
    ../include/hanami_files/dataset_files/image_dataset_file.h \
    ../include/hanami_files/dataset_files/table_dataset_file.h

