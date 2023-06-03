QT -= qt core gui

TARGET = KitsunemimiHanamiCommon
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.2.0

LIBS += -L../../libKitsunemimiArgs/src -lKitsunemimiArgs
LIBS += -L../../libKitsunemimiArgs/src/debug -lKitsunemimiArgs
LIBS += -L../../libKitsunemimiArgs/src/release -lKitsunemimiArgs
INCLUDEPATH += ../../libKitsunemimiArgs/include

LIBS += -L../../libKitsunemimiConfig/src -lKitsunemimiConfig
LIBS += -L../../libKitsunemimiConfig/src/debug -lKitsunemimiConfig
LIBS += -L../../libKitsunemimiConfig/src/release -lKitsunemimiConfig
INCLUDEPATH += ../../libKitsunemimiConfig/include

LIBS += -L../../libKitsunemimiIni/src -lKitsunemimiIni
LIBS += -L../../libKitsunemimiIni/src/debug -lKitsunemimiIni
LIBS += -L../../libKitsunemimiIni/src/release -lKitsunemimiIni
INCLUDEPATH += ../../libKitsunemimiIni/include

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS += -luuid

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiHanamiCommon/args.h \
    ../include/libKitsunemimiHanamiCommon/defines.h \
    ../include/libKitsunemimiHanamiCommon/config.h \
    ../include/libKitsunemimiHanamiCommon/uuid.h \
    ../include/libKitsunemimiHanamiCommon/structs.h \
    ../include/libKitsunemimiHanamiCommon/enums.h \
    ../include/libKitsunemimiHanamiCommon/generic_main.h \
    ../include/libKitsunemimiHanamiCommon/functions.h

SOURCES += \
    config.cpp

