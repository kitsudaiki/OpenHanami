QT -= qt core gui

TARGET = KitsunemimiHanamiFiles
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.1.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS += -L../../libKitsunemimiHanamiCommon/src -lKitsunemimiHanamiCommon
LIBS += -L../../libKitsunemimiHanamiCommon/src/debug -lKitsunemimiHanamiCommon
LIBS += -L../../libKitsunemimiHanamiCommon/src/release -lKitsunemimiHanamiCommon
INCLUDEPATH += ../../libKitsunemimiHanamiCommon/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiHanamiFiles/data_set_files/data_set_file.h \
    ../include/libKitsunemimiHanamiFiles/data_set_files/image_data_set_file.h \
    ../include/libKitsunemimiHanamiFiles/data_set_files/table_data_set_file.h \
    ../include/libKitsunemimiHanamiFiles/structs.h

SOURCES += \
    data_set_files/data_set_file.cpp \
    data_set_files/image_data_set_file.cpp \
    data_set_files/table_data_set_file.cpp


