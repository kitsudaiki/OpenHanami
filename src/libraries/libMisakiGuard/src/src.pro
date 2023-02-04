QT -= qt core gui

TARGET = MisakiGuard
CONFIG += c++17
TEMPLATE = lib
VERSION = 0.1.0

LIBS += -L../../libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libKitsunemimiCommon/include

LIBS += -L../../libKitsunemimiJwt/src -lKitsunemimiJwt
LIBS += -L../../libKitsunemimiJwt/src/debug -lKitsunemimiJwt
LIBS += -L../../libKitsunemimiJwt/src/release -lKitsunemimiJwt
INCLUDEPATH += ../../libKitsunemimiJwt/include

LIBS += -L../../libKitsunemimiCrypto/src -lKitsunemimiCrypto
LIBS += -L../../libKitsunemimiCrypto/src/debug -lKitsunemimiCrypto
LIBS += -L../../libKitsunemimiCrypto/src/release -lKitsunemimiCrypto
INCLUDEPATH += ../../libKitsunemimiCrypto/include

LIBS += -L../../libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../libKitsunemimiJson/include

LIBS += -L../../libKitsunemimiIni/src -lKitsunemimiIni
LIBS += -L../../libKitsunemimiIni/src/debug -lKitsunemimiIni
LIBS += -L../../libKitsunemimiIni/src/release -lKitsunemimiIni
INCLUDEPATH += ../../libKitsunemimiIni/include

LIBS += -L../../libKitsunemimiConfig/src -lKitsunemimiConfig
LIBS += -L../../libKitsunemimiConfig/src/debug -lKitsunemimiConfig
LIBS += -L../../libKitsunemimiConfig/src/release -lKitsunemimiConfig
INCLUDEPATH += ../../libKitsunemimiConfig/include

LIBS += -L../../libKitsunemimiNetwork/src -lKitsunemimiNetwork
LIBS += -L../../libKitsunemimiNetwork/src/debug -lKitsunemimiNetwork
LIBS += -L../../libKitsunemimiNetwork/src/release -lKitsunemimiNetwork
INCLUDEPATH += ../../libKitsunemimiNetwork/include

LIBS += -L../../libKitsunemimiSakuraNetwork/src -lKitsunemimiSakuraNetwork
LIBS += -L../../libKitsunemimiSakuraNetwork/src/debug -lKitsunemimiSakuraNetwork
LIBS += -L../../libKitsunemimiSakuraNetwork/src/release -lKitsunemimiSakuraNetwork
INCLUDEPATH += ../../libKitsunemimiSakuraNetwork/include

LIBS += -L../../libKitsunemimiHanamiCommon/src -lKitsunemimiHanamiCommon
LIBS += -L../../libKitsunemimiHanamiCommon/src/debug -lKitsunemimiHanamiCommon
LIBS += -L../../libKitsunemimiHanamiCommon/src/release -lKitsunemimiHanamiCommon
INCLUDEPATH += ../../libKitsunemimiHanamiCommon/include

LIBS += -L../../libKitsunemimiHanamiNetwork/src -lKitsunemimiHanamiNetwork
LIBS += -L../../libKitsunemimiHanamiNetwork/src/debug -lKitsunemimiHanamiNetwork
LIBS += -L../../libKitsunemimiHanamiNetwork/src/release -lKitsunemimiHanamiNetwork
INCLUDEPATH += ../../libKitsunemimiHanamiNetwork/include

LIBS += -lssl -lcryptopp -lcrypto

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libMisakiGuard/misaki_input.h \
    generate_api_docu.h \
    md_docu_generation.h \
    rst_docu_generation.h

SOURCES += \
    generate_api_docu.cpp \
    md_docu_generation.cpp \
    misaki_input.cpp \
    rst_docu_generation.cpp

DISTFILES +=
