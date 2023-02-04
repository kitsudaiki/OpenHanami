QT -= qt core gui

TARGET = ShioriArchive
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

LIBS += -L../../libraries/libKitsunemimiSakuraNetwork/src -lKitsunemimiSakuraNetwork
LIBS += -L../../libraries/libKitsunemimiSakuraNetwork/src/debug -lKitsunemimiSakuraNetwork
LIBS += -L../../libraries/libKitsunemimiSakuraNetwork/src/release -lKitsunemimiSakuraNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiSakuraNetwork/include

LIBS += -L../../libraries/libKitsunemimiSqlite/src -lKitsunemimiSqlite
LIBS += -L../../libraries/libKitsunemimiSqlite/src/debug -lKitsunemimiSqlite
LIBS += -L../../libraries/libKitsunemimiSqlite/src/release -lKitsunemimiSqlite
INCLUDEPATH += ../../libraries/libKitsunemimiSqlite/include

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

LIBS += -lcryptopp -lssl -lsqlite3 -luuid -lcrypto -pthread -lprotobuf -lpthread

INCLUDEPATH += $$PWD \
               src

SOURCES += src/main.cpp \
    src/api/v1/cluster_snapshot/create_cluster_snapshot.cpp \
    src/api/v1/cluster_snapshot/delete_cluster_snapshot.cpp \
    src/api/v1/cluster_snapshot/finish_cluster_snapshot.cpp \
    src/api/v1/cluster_snapshot/get_cluster_snapshot.cpp \
    src/api/v1/cluster_snapshot/list_cluster_snapshot.cpp \
    src/api/v1/data_files/check_data_set.cpp \
    src/api/v1/data_files/csv/create_csv_data_set.cpp \
    src/api/v1/data_files/csv/finalize_csv_data_set.cpp \
    src/api/v1/data_files/delete_data_set.cpp \
    src/api/v1/data_files/get_data_set.cpp \
    src/api/v1/data_files/get_progress_data_set.cpp \
    src/api/v1/data_files/list_data_set.cpp \
    src/api/v1/data_files/mnist/create_mnist_data_set.cpp \
    src/api/v1/data_files/mnist/finalize_mnist_data_set.cpp \
    src/api/v1/logs/get_audit_log.cpp \
    src/api/v1/logs/get_error_log.cpp \
    src/api/v1/request_results/delete_request_result.cpp \
    src/api/v1/request_results/get_request_result.cpp \
    src/api/v1/request_results/list_request_result.cpp \
    src/core/data_set_files/data_set_file.cpp \
    src/core/data_set_files/image_data_set_file.cpp \
    src/core/data_set_files/table_data_set_file.cpp \
    src/core/temp_file_handler.cpp \
    src/database/audit_log_table.cpp \
    src/database/cluster_snapshot_table.cpp \
    src/database/data_set_table.cpp \
    src/database/error_log_table.cpp \
    src/database/request_result_table.cpp \
    src/shiori_root.cpp

HEADERS += \
    src/api/blossom_initializing.h \
    src/api/v1/cluster_snapshot/create_cluster_snapshot.h \
    src/api/v1/cluster_snapshot/delete_cluster_snapshot.h \
    src/api/v1/cluster_snapshot/finish_cluster_snapshot.h \
    src/api/v1/cluster_snapshot/get_cluster_snapshot.h \
    src/api/v1/cluster_snapshot/list_cluster_snapshot.h \
    src/api/v1/data_files/check_data_set.h \
    src/api/v1/data_files/csv/create_csv_data_set.h \
    src/api/v1/data_files/csv/finalize_csv_data_set.h \
    src/api/v1/data_files/delete_data_set.h \
    src/api/v1/data_files/get_data_set.h \
    src/api/v1/data_files/get_progress_data_set.h \
    src/api/v1/data_files/list_data_set.h \
    src/api/v1/data_files/mnist/create_mnist_data_set.h \
    src/api/v1/data_files/mnist/finalize_mnist_data_set.h \
    src/api/v1/logs/get_audit_log.h \
    src/api/v1/logs/get_error_log.h \
    src/api/v1/request_results/delete_request_result.h \
    src/api/v1/request_results/get_request_result.h \
    src/api/v1/request_results/list_request_result.h \
    src/args.h \
    src/callbacks.h \
    src/config.h \
    src/core/data_set_files/data_set_file.h \
    src/core/data_set_files/image_data_set_file.h \
    src/core/data_set_files/table_data_set_file.h \
    src/core/temp_file_handler.h \
    src/database/audit_log_table.h \
    src/database/cluster_snapshot_table.h \
    src/database/data_set_table.h \
    src/database/error_log_table.h \
    src/database/request_result_table.h \
    src/shiori_root.h \
    ../../libraries/libKitsunemimiHanamiMessages/hanami_messages/shiori_messages.h

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
