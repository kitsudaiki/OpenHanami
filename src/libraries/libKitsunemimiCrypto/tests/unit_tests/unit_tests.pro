include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiCrypto

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

LIBS +=  -lssl -lcrypto -lcryptopp

INCLUDEPATH += $$PWD

SOURCES += \
    common_test.cpp \
    hashes_test.cpp \
    main.cpp  \
    symmetric_encryption_test.cpp

HEADERS += \
    common_test.h \
    hashes_test.h \
    symmetric_encryption_test.h
