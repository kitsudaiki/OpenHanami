QT -= qt core gui

TARGET = hanami_crypto
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.2.0

LIBS += -L../../hanami_common/src -lhanami_common
LIBS += -L../../hanami_common/src/debug -lhanami_common
LIBS += -L../../hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../hanami_common/include

LIBS +=  -lssl -lcrypto -lcryptopp

INCLUDEPATH += ../../../third-party-libs/json/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/hanami_crypto/hashes.h \
    ../include/hanami_crypto/common.h \
    ../include/hanami_crypto/symmetric_encryption.h \
    ../include/hanami_crypto/signing.h

SOURCES += \
    common.cpp \
    hashes.cpp \
    symmetric_encryption.cpp

