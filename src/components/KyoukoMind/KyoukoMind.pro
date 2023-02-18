QT -= qt core gui

TARGET = KyoukoMind
CONFIG += console
CONFIG += c++17

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

LIBS += -L../../libraries/libKitsunemimiHanamiSegmentParser/src -lKitsunemimiHanamiSegmentParser
LIBS += -L../../libraries/libKitsunemimiHanamiSegmentParser/src/debug -lKitsunemimiHanamiSegmentParser
LIBS += -L../../libraries/libKitsunemimiHanamiSegmentParser/src/release -lKitsunemimiHanamiSegmentParser
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiSegmentParser/include

LIBS += -L../../libraries/libKitsunemimiHanamiClusterParser/src -lKitsunemimiHanamiClusterParser
LIBS += -L../../libraries/libKitsunemimiHanamiClusterParser/src/debug -lKitsunemimiHanamiClusterParser
LIBS += -L../../libraries/libKitsunemimiHanamiClusterParser/src/release -lKitsunemimiHanamiClusterParser
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiClusterParser/include

LIBS += -L../../libraries/libKitsunemimiArgs/src -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/debug -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/release -lKitsunemimiArgs
INCLUDEPATH += ../../libraries/libKitsunemimiArgs/include

LIBS += -L../../libraries/libKitsunemimiConfig/src -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/debug -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/release -lKitsunemimiConfig
INCLUDEPATH += ../../libraries/libKitsunemimiConfig/include

LIBS += -L../../libraries/libKitsunemimiSakuraDatabase/src -lKitsunemimiSakuraDatabase
LIBS += -L../../libraries/libKitsunemimiSakuraDatabase/src/debug -lKitsunemimiSakuraDatabase
LIBS += -L../../libraries/libKitsunemimiSakuraDatabase/src/release -lKitsunemimiSakuraDatabase
INCLUDEPATH += ../../libraries/libKitsunemimiSakuraDatabase/include

LIBS += -L../../libraries/libKitsunemimiSakuraNetwork/src -lKitsunemimiSakuraNetwork
LIBS += -L../../libraries/libKitsunemimiSakuraNetwork/src/debug -lKitsunemimiSakuraNetwork
LIBS += -L../../libraries/libKitsunemimiSakuraNetwork/src/release -lKitsunemimiSakuraNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiSakuraNetwork/include

LIBS += -L../../libraries/libKitsunemimiNetwork/src -lKitsunemimiNetwork
LIBS += -L../../libraries/libKitsunemimiNetwork/src/debug -lKitsunemimiNetwork
LIBS += -L../../libraries/libKitsunemimiNetwork/src/release -lKitsunemimiNetwork
INCLUDEPATH += ../../libraries/libKitsunemimiNetwork/include

LIBS += -L../../libraries/libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libraries/libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libraries/libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libraries/libKitsunemimiCommon/include

LIBS += -L../../libraries/libKitsunemimiSqlite/src -lKitsunemimiSqlite
LIBS += -L../../libraries/libKitsunemimiSqlite/src/debug -lKitsunemimiSqlite
LIBS += -L../../libraries/libKitsunemimiSqlite/src/release -lKitsunemimiSqlite
INCLUDEPATH += ../../libraries/libKitsunemimiSqlite/include

LIBS += -L../../libraries/libKitsunemimiIni/src -lKitsunemimiIni
LIBS += -L../../libraries/libKitsunemimiIni/src/debug -lKitsunemimiIni
LIBS += -L../../libraries/libKitsunemimiIni/src/release -lKitsunemimiIni
INCLUDEPATH += ../../libraries/libKitsunemimiIni/include

LIBS += -L../../libraries/libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../libraries/libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../libraries/libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../libraries/libKitsunemimiJson/include

LIBS += -L../../libraries/libKitsunemimiJwt/src -lKitsunemimiJwt
LIBS += -L../../libraries/libKitsunemimiJwt/src/debug -lKitsunemimiJwt
LIBS += -L../../libraries/libKitsunemimiJwti/src/release -lKitsunemimiJwt
INCLUDEPATH += ../../libraries/libKitsunemimiJwt/include

LIBS += -L../../libraries/libKitsunemimiCrypto/src -lKitsunemimiCrypto
LIBS += -L../../libraries/libKitsunemimiCrypto/src/debug -lKitsunemimiCrypto
LIBS += -L../../libraries/libKitsunemimiCrypto/src/release -lKitsunemimiCrypto
INCLUDEPATH += ../../libraries/libKitsunemimiCrypto/include

LIBS += -L../../libraries/libKitsunemimiOpencl/src -lKitsunemimiOpencl
LIBS += -L../../libraries/libKitsunemimiOpencl/src/debug -lKitsunemimiOpencl
LIBS += -L../../libraries/libKitsunemimiOpencl/src/release -lKitsunemimiOpencl
INCLUDEPATH += ../../libraries/libKitsunemimiOpencl/include

LIBS += -lcryptopp -lssl -lsqlite3 -luuid -lcrypto -pthread -lprotobuf -lOpenCL

INCLUDEPATH += $$PWD \
               src

HEADERS += \
    src/api/blossom_initializing.h \
    src/api/v1/cluster/create_cluster.h \
    src/api/v1/cluster/delete_cluster.h \
    src/api/v1/cluster/list_cluster.h \
    src/api/v1/cluster/load_cluster.h \
    src/api/v1/cluster/save_cluster.h \
    src/api/v1/cluster/set_cluster_mode.h \
    src/api/v1/cluster/show_cluster.h \
    src/api/v1/task/create_task.h \
    src/api/v1/task/delete_task.h \
    src/api/v1/task/list_task.h \
    src/api/v1/task/show_task.h \
    src/api/v1/template/delete_template.h \
    src/api/v1/template/list_templates.h \
    src/api/v1/template/show_template.h \
    src/api/v1/template/upload_template.h \
    src/args.h \
    src/callbacks.h \
    src/common.h \
    src/common/defines.h \
    src/common/enums.h \
    src/common/functions.h \
    src/common/includes.h \
    src/common/structs.h \
    src/common/typedefs.h \
    src/config.h \
    src/core/callbacks.h \
    src/core/cluster/cluster.h \
    src/core/cluster/cluster_handler.h \
    src/core/cluster/cluster_init.h \
    src/core/cluster/statemachine_init.h \
    src/core/cluster/states/cycle_finish_state.h \
    src/core/cluster/states/tables/table_interpolation_state.h \
    src/core/cluster/states/tables/table_learn_forward_state.h \
    src/core/cluster/states/images/image_identify_state.h \
    src/core/cluster/states/images/image_learn_forward_state.h \
    src/core/cluster/states/snapshots/restore_cluster_state.h \
    src/core/cluster/states/snapshots/save_cluster_state.h \
    src/core/cluster/states/task_handle_state.h \
    src/core/cluster/task.h \
    src/core/processing/cpu_processing_unit.h \
    src/core/processing/processing_unit_handler.h \
    src/core/processing/segment_queue.h \
    src/core/routing_functions.h \
    src/core/segments/abstract_segment.h \
    src/core/segments/brick.h \
    src/core/segments/core_segment/backpropagation.h \
    src/core/segments/core_segment/core_segment.h \
    src/core/segments/core_segment/objects.h \
    src/core/segments/core_segment/processing.h \
    src/core/segments/core_segment/reduction.h \
    src/core/segments/core_segment/section_update.h \
    src/core/segments/input_segment/input_segment.h \
    src/core/segments/input_segment/objects.h \
    src/core/segments/input_segment/processing.h \
    src/core/segments/output_segment/backpropagation.h \
    src/core/segments/output_segment/objects.h \
    src/core/segments/output_segment/output_segment.h \
    src/core/segments/output_segment/processing.h \
    src/core/segments/segment_meta.h \
    src/core/struct_validation.h \
    src/database/cluster_table.h \
    src/database/template_table.h \
    src/io/protobuf_messages.h \
    src/kyouko_root.h

SOURCES += \
    src/api/v1/cluster/create_cluster.cpp \
    src/api/v1/cluster/delete_cluster.cpp \
    src/api/v1/cluster/list_cluster.cpp \
    src/api/v1/cluster/load_cluster.cpp \
    src/api/v1/cluster/save_cluster.cpp \
    src/api/v1/cluster/set_cluster_mode.cpp \
    src/api/v1/cluster/show_cluster.cpp \
    src/api/v1/task/create_task.cpp \
    src/api/v1/task/delete_task.cpp \
    src/api/v1/task/list_task.cpp \
    src/api/v1/task/show_task.cpp \
    src/api/v1/template/delete_template.cpp \
    src/api/v1/template/list_templates.cpp \
    src/api/v1/template/show_template.cpp \
    src/api/v1/template/upload_template.cpp \
    src/callbacks.cpp \
    src/core/cluster/cluster.cpp \
    src/core/cluster/cluster_handler.cpp \
    src/core/cluster/cluster_init.cpp \
    src/core/cluster/statemachine_init.cpp \
    src/core/cluster/states/cycle_finish_state.cpp \
    src/core/cluster/states/tables/table_interpolation_state.cpp \
    src/core/cluster/states/tables/table_learn_forward_state.cpp \
    src/core/cluster/states/images/image_identify_state.cpp \
    src/core/cluster/states/images/image_learn_forward_state.cpp \
    src/core/cluster/states/snapshots/restore_cluster_state.cpp \
    src/core/cluster/states/snapshots/save_cluster_state.cpp \
    src/core/cluster/states/task_handle_state.cpp \
    src/core/processing/cpu_processing_unit.cpp \
    src/core/processing/processing_unit_handler.cpp \
    src/core/processing/segment_queue.cpp \
    src/core/segments/abstract_segment.cpp \
    src/core/segments/core_segment/core_segment.cpp \
    src/core/segments/input_segment/input_segment.cpp \
    src/core/segments/output_segment/output_segment.cpp \
    src/database/cluster_table.cpp \
    src/database/template_table.cpp \
    src/io/protobuf_messages.cpp \
    src/kyouko_root.cpp \
    src/main.cpp

KYOUKO_PROTO_BUFFER = ../../libraries/libKitsunemimiHanamiMessages/protobuffers/kyouko_messages.proto3
# GPU_KERNEL = src/core/segments/core_segment/gpu_kernel.cl

OTHER_FILES += $$KYOUKO_PROTO_BUFFER # \
               # $$GPU_KERNEL

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

# gpu_processing.input = GPU_KERNEL
# gpu_processing.output = ${QMAKE_FILE_BASE}.h
# gpu_processing.commands = xxd -i ${QMAKE_FILE_IN} \
#    | sed 's/_________Hanami_AI_KyoukoMind_src_core_segments_dynamic_segment_//g' \
#    | sed 's/________KyoukoMind_src_core_segments_dynamic_segment_//g' \
#    | sed 's/_______KyoukoMind_src_core_segments_dynamic_segment_//g' \
#    | sed 's/______KyoukoMind_src_core_segments_dynamic_segment_//g' \
#    | sed 's/_____KyoukoMind_src_core_segments_dynamic_segment_//g' \
#    | sed 's/____KyoukoMind_src_core_segments_dynamic_segment_//g' \
#    | sed 's/___KyoukoMind_src_core_segments_dynamic_segment_//g' > ${QMAKE_FILE_BASE}.h
# gpu_processing.variable_out = HEADERS
# gpu_processing.CONFIG += target_predeps no_link

# QMAKE_EXTRA_COMPILERS += gpu_processing

