QT -= qt core gui

TARGET = ToriiGateway
CONFIG += console c++17
CONFIG -= app_bundle

LIBS += -L../../libraries/libShioriArchive/src -lShioriArchive
LIBS += -L../../libraries/libShioriArchive/src/debug -lShioriArchive
LIBS += -L../../libraries/libShioriArchive/src/release -lShioriArchive
INCLUDEPATH += ../../libraries/libShioriArchive/include

LIBS += -L../../libraries/libAzukiHeart/src -lAzukiHeart
LIBS += -L../../libraries/libAzukiHeart/src/debug -lAzukiHeart
LIBS += -L../../libraries/libAzukiHeart/src/release -lAzukiHeart
INCLUDEPATH += ../../libraries/libAzukiHeart/include

LIBS += -L../../libraries/libKitsunemimiHanamiNetwork/src -lKitsunemimiHanamiNetwork
LIBS += -L../../libraries/libKitsunemimiHanamiNetwork/src/debug -lKitsunemimiHanamiNetwork
LIBS += -L../../libraries/libKitsunemimiHanamiNetwork/src/release -lKitsunemimiHanamiNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiNetwork/include

LIBS += -L../../libraries/libKitsunemimiArgs/src -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/debug -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/release -lKitsunemimiArgs
INCLUDEPATH += ../../libraries/libKitsunemimiArgs/include

LIBS += -L../../libraries/libKitsunemimiConfig/src -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/debug -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/release -lKitsunemimiConfig
INCLUDEPATH += ../../libraries/libKitsunemimiConfig/include

LIBS += -L../../libraries/libKitsunemimiSakuraNetwork/src -lKitsunemimiSakuraNetwork
LIBS += -L../../libraries/libKitsunemimiSakuraNetwork/src/debug -lKitsunemimiSakuraNetwork
LIBS += -L../../libraries/libKitsunemimiSakuraNetwork/src/release -lKitsunemimiSakuraNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiSakuraNetwork/include

LIBS += -L../../libraries/libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libraries/libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libraries/libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libraries/libKitsunemimiCommon/include

LIBS += -L../../libraries/libKitsunemimiNetwork/src -lKitsunemimiNetwork
LIBS += -L../../libraries/libKitsunemimiNetwork/src/debug -lKitsunemimiNetwork
LIBS += -L../../libraries/libKitsunemimiNetwork/src/release -lKitsunemimiNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiNetwork/include

LIBS += -L../../libraries/libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../libraries/libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../libraries/libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../libraries/libKitsunemimiJson/include

LIBS += -L../../libraries/libKitsunemimiIni/src -lKitsunemimiIni
LIBS += -L../../libraries/libKitsunemimiIni/src/debug -lKitsunemimiIni
LIBS += -L../../libraries/libKitsunemimiIni/src/release -lKitsunemimiIni
INCLUDEPATH += ../../libraries/libKitsunemimiIni/include

LIBS += -L../../libraries/libKitsunemimiHanamiCommon/src -lKitsunemimiHanamiCommon
LIBS += -L../../libraries/libKitsunemimiHanamiCommon/src/debug -lKitsunemimiHanamiCommon
LIBS += -L../../libraries/libKitsunemimiHanamiCommon/src/release -lKitsunemimiHanamiCommon
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiCommon/include

LIBS += -L../../libraries/libKitsunemimiJwt/src -lKitsunemimiJwt
LIBS += -L../../libraries/libKitsunemimiJwt/src/debug -lKitsunemimiJwt
LIBS += -L../../libraries/libKitsunemimiJwti/src/release -lKitsunemimiJwt
INCLUDEPATH += ../../libraries/libKitsunemimiJwt/include

LIBS += -L../../libraries/libKitsunemimiCrypto/src -lKitsunemimiCrypto
LIBS += -L../../libraries/libKitsunemimiCrypto/src/debug -lKitsunemimiCrypto
LIBS += -L../../libraries/libKitsunemimiCrypto/src/release -lKitsunemimiCrypto
INCLUDEPATH += ../../libraries/libKitsunemimiCrypto/include

LIBS += -lcryptopp -lssl -lcrypto -luuid -pthread -lprotobuf

INCLUDEPATH += $$PWD \
               src

SOURCES += \
        src/callbacks.cpp \
        src/core/http_processing/file_send.cpp \
        src/core/http_processing/http_processing.cpp \
        src/core/http_websocket_thread.cpp \
        src/main.cpp \
        src/torii_root.cpp \
        src/core/http_server.cpp

HEADERS += \
        src/args.h \
        src/core/http_processing/file_send.h \
        src/core/http_processing/http_processing.h \
        src/core/http_processing/response_builds.h \
        src/core/http_processing/string_functions.h \
        src/core/http_websocket_thread.h \
        src/torii_root.h \
        src/core/http_server.h \
        src/config.h \
        src/callbacks.h

SHIORI_PROTO_BUFFER = ../../libraries/libKitsunemimiHanamiMessages/protobuffers/shiori_messages.proto3

OTHER_FILES += $$SHIORI_PROTO_BUFFER

protobuf_decl.name = protobuf headers
protobuf_decl.input = SHIORI_PROTO_BUFFER
protobuf_decl.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.h
protobuf_decl.commands = protoc --cpp_out=${QMAKE_FILE_IN_PATH} --proto_path=${QMAKE_FILE_IN_PATH} ${QMAKE_FILE_NAME}
protobuf_decl.variable_out = HEADERS
QMAKE_EXTRA_COMPILERS += protobuf_decl

protobuf_impl.name = protobuf sources
protobuf_impl.input = SHIORI_PROTO_BUFFER
protobuf_impl.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.cc
protobuf_impl.depends = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.h
protobuf_impl.commands = $$escape_expand(\n)
protobuf_impl.variable_out = SOURCES
QMAKE_EXTRA_COMPILERS += protobuf_impl
