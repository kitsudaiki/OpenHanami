QT -= qt core gui

TARGET = MisakiGuard
CONFIG += console c++17
CONFIG -= app_bundle

LIBS += -L../../libraries/libAzukiHeart/src -lAzukiHeart
LIBS += -L../../libraries/libAzukiHeart/src/debug -lAzukiHeart
LIBS += -L../../libraries/libAzukiHeart/src/release -lAzukiHeart
INCLUDEPATH += ../../libraries/libAzukiHeart/include

LIBS += -L../../libraries/libMisakiGuard/src -lMisakiGuard
LIBS += -L../../libraries/libMisakiGuard/src/debug -lMisakiGuard
LIBS += -L../../libraries/libMisakiGuard/src/release -lMisakiGuard
INCLUDEPATH += ../../libraries/libMisakiGuard/include

LIBS += -L../../libraries/libKitsunemimiHanamiNetwork/src -lKitsunemimiHanamiNetwork
LIBS += -L../../libraries/libKitsunemimiHanamiNetwork/src/debug -lKitsunemimiHanamiNetwork
LIBS += -L../../libraries/libKitsunemimiHanamiNetwork/src/release -lKitsunemimiHanamiNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiNetwork/include

LIBS += -L../../libraries/libKitsunemimiHanamiDatabase/src -lKitsunemimiHanamiDatabase
LIBS += -L../../libraries/libKitsunemimiHanamiDatabase/src/debug -lKitsunemimiHanamiDatabase
LIBS += -L../../libraries/libKitsunemimiHanamiDatabase/src/release -lKitsunemimiHanamiDatabase
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiDatabase/include

LIBS += -L../../libraries/libKitsunemimiHanamiPolicies/src -lKitsunemimiHanamiPolicies
LIBS += -L../../libraries/libKitsunemimiHanamiPolicies/src/debug -lKitsunemimiHanamiPolicies
LIBS += -L../../libraries/libKitsunemimiHanamiPolicies/src/release -lKitsunemimiHanamiPolicies
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiPolicies/include

LIBS += -L../../libraries/libKitsunemimiHanamiCommon/src -lKitsunemimiHanamiCommon
LIBS += -L../../libraries/libKitsunemimiHanamiCommon/src/debug -lKitsunemimiHanamiCommon
LIBS += -L../../libraries/libKitsunemimiHanamiCommon/src/release -lKitsunemimiHanamiCommon
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiCommon/include

LIBS += -L../../libraries/libKitsunemimiSakuraDatabase/src -lKitsunemimiSakuraDatabase
LIBS += -L../../libraries/libKitsunemimiSakuraDatabase/src/debug -lKitsunemimiSakuraDatabase
LIBS += -L../../libraries/libKitsunemimiSakuraDatabase/src/release -lKitsunemimiSakuraDatabase
INCLUDEPATH += ../../libraries/libKitsunemimiSakuraDatabase/include

LIBS += -L../../libraries/libKitsunemimiArgs/src -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/debug -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/release -lKitsunemimiArgs
INCLUDEPATH += ../../libraries/libKitsunemimiArgs/include

LIBS += -L../../libraries/libKitsunemimiConfig/src -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/debug -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/release -lKitsunemimiConfig
INCLUDEPATH += ../../libraries/libKitsunemimiConfig/include

LIBS += -L../../libraries/libKitsunemimiSqlite/src -lKitsunemimiSqlite
LIBS += -L../../libraries/libKitsunemimiSqlite/src/debug -lKitsunemimiSqlite
LIBS += -L../../libraries/libKitsunemimiSqlite/src/release -lKitsunemimiSqlite
INCLUDEPATH += ../../libraries/libKitsunemimiSqlite/include

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

LIBS += -L../../libraries/libKitsunemimiJwt/src -lKitsunemimiJwt
LIBS += -L../../libraries/libKitsunemimiJwt/src/debug -lKitsunemimiJwt
LIBS += -L../../libraries/libKitsunemimiJwti/src/release -lKitsunemimiJwt
INCLUDEPATH += ../../libraries/libKitsunemimiJwt/include

LIBS += -L../../libraries/libKitsunemimiCrypto/src -lKitsunemimiCrypto
LIBS += -L../../libraries/libKitsunemimiCrypto/src/debug -lKitsunemimiCrypto
LIBS += -L../../libraries/libKitsunemimiCrypto/src/release -lKitsunemimiCrypto
INCLUDEPATH += ../../libraries/libKitsunemimiCrypto/include


LIBS += -lcryptopp -lssl -lsqlite3 -luuid -lcrypto -pthread -lprotobuf

INCLUDEPATH += $$PWD \
               src

SOURCES += src/main.cpp \
    src/api/v1/auth/create_internal_token.cpp \
    src/api/v1/auth/create_token.cpp \
    src/api/v1/auth/list_user_projects.cpp \
    src/api/v1/auth/renew_token.cpp \
    src/api/v1/auth/validate_access.cpp \
    src/api/v1/project/create_project.cpp \
    src/api/v1/project/delete_project.cpp \
    src/api/v1/project/get_project.cpp \
    src/api/v1/project/list_projects.cpp \
    src/api/v1/user/add_project_to_user.cpp \
    src/api/v1/user/create_user.cpp \
    src/api/v1/user/delete_user.cpp \
    src/api/v1/user/get_user.cpp \
    src/api/v1/user/list_users.cpp \
    src/api/v1/documentation/generate_rest_api_docu.cpp \
    src/api/v1/user/remove_project_from_user.cpp \
    src/database/projects_table.cpp \
    src/misaki_root.cpp \
    src/database/users_table.cpp

HEADERS += \
    src/api/v1/auth/create_internal_token.h \
    src/api/v1/auth/create_token.h \
    src/api/v1/auth/list_user_projects.h \
    src/api/v1/auth/renew_token.h \
    src/api/v1/auth/validate_access.h \
    src/api/blossom_initializing.h \
    src/api/v1/project/create_project.h \
    src/api/v1/project/delete_project.h \
    src/api/v1/project/get_project.h \
    src/api/v1/project/list_projects.h \
    src/api/v1/user/add_project_to_user.h \
    src/api/v1/user/create_user.h \
    src/api/v1/user/delete_user.h \
    src/api/v1/user/get_user.h \
    src/api/v1/user/list_users.h \
    src/api/v1/user/remove_project_from_user.h \
    src/args.h \
    src/callbacks.h \
    src/config.h \
    src/api/v1/documentation/generate_rest_api_docu.h \
    src/database/projects_table.h \
    src/misaki_root.h \
    src/database/users_table.h

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
