include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiJwt

LIBS += -L../../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../libKitsunemimiCommon/include

LIBS += -L../../../libKitsunemimiCrypto/src -lKitsunemimiCrypto
LIBS += -L../../../libKitsunemimiCrypto/src/debug -lKitsunemimiCrypto
LIBS += -L../../../libKitsunemimiCrypto/src/release -lKitsunemimiCrypto
INCLUDEPATH += ../../../libKitsunemimiCrypto/include

LIBS += -L../../../libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../../libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../../libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../../libKitsunemimiJson/include

LIBS +=  -lssl -lcrypto -lcryptopp


INCLUDEPATH += $$PWD

SOURCES += \
    jwt_test.cpp \
    main.cpp 

HEADERS += \
    jwt_test.h
