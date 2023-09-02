QT -= qt core gui

TARGET = hanami_files
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

INCLUDEPATH += $$PWD \
               $$PWD/../include

SOURCES += \
    data_set_files/data_set_file.cpp \
    data_set_files/data_set_functions.cpp \
    data_set_files/image_data_set_file.cpp \
    data_set_files/table_data_set_file.cpp

HEADERS += \
    ../include/hanami_files/data_set_files/data_set_file.h \
    ../include/hanami_files/data_set_files/data_set_functions.h \
    ../include/hanami_files/data_set_files/image_data_set_file.h \
    ../include/hanami_files/data_set_files/table_data_set_file.h

