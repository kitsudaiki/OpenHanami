QT -= qt core gui

TARGET = HanamiAiSdk
CONFIG += c++17
TEMPLATE = lib
VERSION = 0.3.1

LIBS += -L../../../../libraries/libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../../../libraries/libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../../../libraries/libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../../../libraries/libKitsunemimiCommon/include

LIBS += -L../../../../libraries/libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../../../libraries/libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../../../libraries/libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../../../libraries/libKitsunemimiJson/include

LIBS += -L../../../../libraries/libKitsunemimiCrypto/src -lKitsunemimiCrypto
LIBS += -L../../../../libraries/libKitsunemimiCrypto/src/debug -lKitsunemimiCrypto
LIBS += -L../../../../libraries/libKitsunemimiCrypto/src/release -lKitsunemimiCrypto
INCLUDEPATH += ../../../../libraries/libKitsunemimiCrypto/include

LIBS += -L../../../../libraries/libKitsunemimiHanamiCommon/src -lKitsunemimiHanamiCommon
LIBS += -L../../../../libraries/libKitsunemimiHanamiCommon/src/debug -lKitsunemimiHanamiCommon
LIBS += -L../../../../libraries/libKitsunemimiHanamiCommon/src/release -lKitsunemimiHanamiCommon
INCLUDEPATH += ../../../../libraries/libKitsunemimiHanamiCommon/include

LIBS += -lssl -lcryptopp -lcrypt

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/libHanamiAiSdk/cluster.h \
    ../include/libHanamiAiSdk/data_set.h \
    ../include/libHanamiAiSdk/init.h \
    ../include/libHanamiAiSdk/project.h \
    ../include/libHanamiAiSdk/request_result.h \
    ../include/libHanamiAiSdk/task.h \
    ../include/libHanamiAiSdk/template.h \
    ../include/libHanamiAiSdk/user.h \
    ../include/libHanamiAiSdk/snapshot.h \
    ../include/libHanamiAiSdk/io.h \
    common/http_client.h \
    ../include/libHanamiAiSdk/common/websocket_client.h

SOURCES += \
    cluster.cpp \
    data_set.cpp \
    init.cpp \
    io.cpp \
    project.cpp \
    request_result.cpp \
    task.cpp \
    template.cpp \
    user.cpp \
    snapshot.cpp \
    common/http_client.cpp \
    common/websocket_client.cpp


HANAMI_PROTO_BUFFER = ../../../../libraries/libKitsunemimiHanamiMessages/protobuffers/hanami_messages.proto3

OTHER_FILES += $$HANAMI_PROTO_BUFFER

protobuf_decl_hanami.name = protobuf headers
protobuf_decl_hanami.name = protobuf headers
protobuf_decl_hanami.input = HANAMI_PROTO_BUFFER
protobuf_decl_hanami.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.h
protobuf_decl_hanami.commands = protoc --cpp_out=${QMAKE_FILE_IN_PATH} --proto_path=${QMAKE_FILE_IN_PATH} ${QMAKE_FILE_NAME}
protobuf_decl_hanami.variable_out = HEADERS
QMAKE_EXTRA_COMPILERS += protobuf_decl_hanami

protobuf_impl_hanami.name = protobuf sources
protobuf_impl_hanami.input = HANAMI_PROTO_BUFFER
protobuf_impl_hanami.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.cc
protobuf_impl_hanami.depends = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.h
protobuf_impl_hanami.commands = $$escape_expand(\n)
protobuf_impl_hanami.variable_out = SOURCES
QMAKE_EXTRA_COMPILERS += protobuf_impl_hanami
