QT -= qt core gui

TARGET = hanami_sdk
CONFIG += c++17
TEMPLATE = lib
VERSION = 0.3.1

LIBS += -L../../../../libraries/hanami_common/src -lhanami_common
LIBS += -L../../../../libraries/hanami_common/src/debug -lhanami_common
LIBS += -L../../../../libraries/hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../../../libraries/hanami_common/include

LIBS += -L../../../../libraries/hanami_crypto/src -lhanami_crypto
LIBS += -L../../../../libraries/hanami_crypto/src/debug -lhanami_crypto
LIBS += -L../../../../libraries/hanami_crypto/src/release -lhanami_crypto
INCLUDEPATH += ../../../../libraries/hanami_crypto/include

LIBS += -lssl -lcryptopp -lcrypt

INCLUDEPATH += ../../../../third-party-libs/json/include

INCLUDEPATH += $$PWD \
               $$PWD/../include

HEADERS += \
    ../include/hanami_sdk/cluster.h \
    ../include/hanami_sdk/common/websocket_client.h \
    ../include/hanami_sdk/data_set.h \
    ../include/hanami_sdk/init.h \
    ../include/hanami_sdk/io.h \
    ../include/hanami_sdk/project.h \
    ../include/hanami_sdk/request_result.h \
    ../include/hanami_sdk/checkpoint.h \
    ../include/hanami_sdk/task.h \
    ../include/hanami_sdk/user.h \
    common/http_client.h

SOURCES += \
    cluster.cpp \
    common/http_client.cpp \
    common/websocket_client.cpp \
    data_set.cpp \
    init.cpp \
    io.cpp \
    project.cpp \
    request_result.cpp \
    checkpoint.cpp \
    task.cpp \
    user.cpp


HANAMI_PROTO_BUFFER = ../../../../libraries/hanami_messages/protobuffers/hanami_messages.proto3

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
