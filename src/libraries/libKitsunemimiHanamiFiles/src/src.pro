QT -= qt core gui

TARGET = KitsunemimiHanamiFiles
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

INCLUDEPATH += $$PWD \
               $$PWD/../include

SOURCES += \
    data_set_files/data_set_file.cpp \
    data_set_files/data_set_functions.cpp \
    data_set_files/image_data_set_file.cpp \
    data_set_files/table_data_set_file.cpp

HEADERS += \
    ../include/libKitsunemimiHanamiFiles/data_set_files/data_set_file.h \
    ../include/libKitsunemimiHanamiFiles/data_set_files/data_set_functions.h \
    ../include/libKitsunemimiHanamiFiles/data_set_files/image_data_set_file.h \
    ../include/libKitsunemimiHanamiFiles/data_set_files/table_data_set_file.h

