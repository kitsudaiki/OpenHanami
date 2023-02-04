QT -= qt core gui

TARGET = AzukiHeart
CONFIG += console
CONFIG += c++17

LIBS += -L../../libraries/../libraries/libAzukiHeart/src -lAzukiHeart
LIBS += -L../../libraries/../libraries/libAzukiHeart/src/debug -lAzukiHeart
LIBS += -L../../libraries/../libraries/libAzukiHeart/src/release -lAzukiHeart
INCLUDEPATH += ../../libraries/libAzukiHeart/include

LIBS += -L../../libraries/../libraries/libMisakiGuard/src -lMisakiGuard
LIBS += -L../../libraries/../libraries/libMisakiGuard/src/debug -lMisakiGuard
LIBS += -L../../libraries/../libraries/libMisakiGuard/src/release -lMisakiGuard
INCLUDEPATH += ../../libraries/libMisakiGuard/include

LIBS += -L../../libraries/../libraries/libKitsunemimiHanamiNetwork/src -lKitsunemimiHanamiNetwork
LIBS += -L../../libraries/../libraries/libKitsunemimiHanamiNetwork/src/debug -lKitsunemimiHanamiNetwork
LIBS += -L../../libraries/../libraries/libKitsunemimiHanamiNetwork/src/release -lKitsunemimiHanamiNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiNetwork/include

LIBS += -L../../libraries/../libraries/libKitsunemimiHanamiCommon/src -lKitsunemimiHanamiCommon
LIBS += -L../../libraries/../libraries/libKitsunemimiHanamiCommon/src/debug -lKitsunemimiHanamiCommon
LIBS += -L../../libraries/../libraries/libKitsunemimiHanamiCommon/src/release -lKitsunemimiHanamiCommon
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiCommon/include

LIBS += -L../../libraries/../libraries/libKitsunemimiSakuraHardware/src -lKitsunemimiSakuraHardware
LIBS += -L../../libraries/../libraries/libKitsunemimiSakuraHardware/src/debug -lKitsunemimiSakuraHardware
LIBS += -L../../libraries/../libraries/libKitsunemimiSakuraHardware/src/release -lKitsunemimiSakuraHardware
INCLUDEPATH += ../../libraries/libKitsunemimiSakuraHardware/include

LIBS += -L../../libraries/../libraries/libKitsunemimiArgs/src -lKitsunemimiArgs
LIBS += -L../../libraries/../libraries/libKitsunemimiArgs/src/debug -lKitsunemimiArgs
LIBS += -L../../libraries/../libraries/libKitsunemimiArgs/src/release -lKitsunemimiArgs
INCLUDEPATH += ../../libraries/libKitsunemimiArgs/include

LIBS += -L../../libraries/../libraries/libKitsunemimiConfig/src -lKitsunemimiConfig
LIBS += -L../../libraries/../libraries/libKitsunemimiConfig/src/debug -lKitsunemimiConfig
LIBS += -L../../libraries/../libraries/libKitsunemimiConfig/src/release -lKitsunemimiConfig
INCLUDEPATH += ../../libraries/libKitsunemimiConfig/include

LIBS += -L../../libraries/../libraries/libKitsunemimiSakuraNetwork/src -lKitsunemimiSakuraNetwork
LIBS += -L../../libraries/../libraries/libKitsunemimiSakuraNetwork/src/debug -lKitsunemimiSakuraNetwork
LIBS += -L../../libraries/../libraries/libKitsunemimiSakuraNetwork/src/release -lKitsunemimiSakuraNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiSakuraNetwork/include

LIBS += -L../../libraries/../libraries/libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libraries/../libraries/libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libraries/../libraries/libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libraries/libKitsunemimiCommon/include

LIBS += -L../../libraries/../libraries/libKitsunemimiNetwork/src -lKitsunemimiNetwork
LIBS += -L../../libraries/../libraries/libKitsunemimiNetwork/src/debug -lKitsunemimiNetwork
LIBS += -L../../libraries/../libraries/libKitsunemimiNetwork/src/release -lKitsunemimiNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiNetwork/include

LIBS += -L../../libraries/../libraries/libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../libraries/../libraries/libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../libraries/../libraries/libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../libraries/libKitsunemimiJson/include

LIBS += -L../../libraries/../libraries/libKitsunemimiIni/src -lKitsunemimiIni
LIBS += -L../../libraries/../libraries/libKitsunemimiIni/src/debug -lKitsunemimiIni
LIBS += -L../../libraries/../libraries/libKitsunemimiIni/src/release -lKitsunemimiIni
INCLUDEPATH += ../../libraries/libKitsunemimiIni/include

LIBS += -L../../libraries/../libraries/libKitsunemimiJwt/src -lKitsunemimiJwt
LIBS += -L../../libraries/../libraries/libKitsunemimiJwt/src/debug -lKitsunemimiJwt
LIBS += -L../../libraries/../libraries/libKitsunemimiJwti/src/release -lKitsunemimiJwt
INCLUDEPATH += ../../libraries/libKitsunemimiJwt/include

LIBS += -L../../libraries/../libraries/libKitsunemimiCrypto/src -lKitsunemimiCrypto
LIBS += -L../../libraries/../libraries/libKitsunemimiCrypto/src/debug -lKitsunemimiCrypto
LIBS += -L../../libraries/../libraries/libKitsunemimiCrypto/src/release -lKitsunemimiCrypto
INCLUDEPATH += ../../libraries/libKitsunemimiCrypto/include

LIBS += -L../../libraries/../libraries/libKitsunemimiCpu/src -lKitsunemimiCpu
LIBS += -L../../libraries/../libraries/libKitsunemimiCpu/src/debug -lKitsunemimiCpu
LIBS += -L../../libraries/../libraries/libKitsunemimiCpu/src/release -lKitsunemimiCpu
INCLUDEPATH += ../../libraries/libKitsunemimiCpu/include

LIBS += -lcryptopp -lssl -luuid -lcrypto -lprotobuf -lpthread

INCLUDEPATH += $$PWD \
               src

SOURCES += src/main.cpp \
    src/api/v1/measurements/power_consumption.cpp \
    src/api/v1/measurements/speed.cpp \
    src/api/v1/measurements/temperature_production.cpp \
    src/api/v1/system_info/get_system_info.cpp \
    src/api/v1/threading/get_thread_mapping.cpp \
    src/azuki_root.cpp \
    src/core/power_measuring.cpp \
    src/core/speed_measuring.cpp \
    src/core/temperature_measuring.cpp \
    src/core/thread_binder.cpp \
    src/core/value_container.cpp

HEADERS += \
    src/api/blossom_initializing.h \
    src/api/v1/measurements/power_consumption.h \
    src/api/v1/measurements/speed.h \
    src/api/v1/measurements/temperature_production.h \
    src/api/v1/system_info/get_system_info.h \
    src/api/v1/threading/get_thread_mapping.h \
    src/args.h \
    src/azuki_root.h \
    src/callbacks.h \
    src/config.h \
    src/core/power_measuring.h \
    src/core/speed_measuring.h \
    src/core/temperature_measuring.h \
    src/core/thread_binder.h \
    src/core/value_container.h

AZUKI_PROTO_BUFFER = ../../libraries/libKitsunemimiHanamiMessages/protobuffers/azuki_messages.proto3

OTHER_FILES += $$AZUKI_PROTO_BUFFER

protobuf_decl.name = protobuf headers
protobuf_decl.name = protobuf headers
protobuf_decl.input = KYOUKO_PROTO_BUFFER
protobuf_decl.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.h
protobuf_decl.commands = protoc --cpp_out=${QMAKE_FILE_IN_PATH} --proto_path=${QMAKE_FILE_IN_PATH} ${QMAKE_FILE_NAME}
protobuf_decl.variable_out = HEADERS
QMAKE_EXTRA_COMPILERS += protobuf_decl

protobuf_impl.name = protobuf sources
protobuf_impl.input = KYOUKO_PROTO_BUFFER
protobuf_impl.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.cc
protobuf_impl.depends = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.h
protobuf_impl.commands = $$escape_expand(\n)
protobuf_impl.variable_out = SOURCES
QMAKE_EXTRA_COMPILERS += protobuf_impl
