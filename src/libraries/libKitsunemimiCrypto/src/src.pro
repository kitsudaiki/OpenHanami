QT -= qt core gui

TARGET = KitsunemimiCrypto
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.2.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS +=  -lssl -lcrypto -lcryptopp

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libKitsunemimiCrypto/hashes.h \
    ../include/libKitsunemimiCrypto/common.h \
    ../include/libKitsunemimiCrypto/symmetric_encryption.h \
    ../include/libKitsunemimiCrypto/signing.h

SOURCES += \
    common.cpp \
    hashes.cpp \
    symmetric_encryption.cpp

