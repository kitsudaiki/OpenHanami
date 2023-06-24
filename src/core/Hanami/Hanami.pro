QT -= qt core gui

TARGET = Hanami
CONFIG += console
CONFIG += c++17

INCLUDEPATH += ../../libraries/libKitsunemimiHanamiMessages/protobuffers

LIBS += -L../../libraries/libKitsunemimiHanamiSegmentParser/src -lKitsunemimiHanamiSegmentParser
LIBS += -L../../libraries/libKitsunemimiHanamiSegmentParser/src/debug -lKitsunemimiHanamiSegmentParser
LIBS += -L../../libraries/libKitsunemimiHanamiSegmentParser/src/release -lKitsunemimiHanamiSegmentParser
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiSegmentParser/include

LIBS += -L../../libraries/libKitsunemimiHanamiClusterParser/src -lKitsunemimiHanamiClusterParser
LIBS += -L../../libraries/libKitsunemimiHanamiClusterParser/src/debug -lKitsunemimiHanamiClusterParser
LIBS += -L../../libraries/libKitsunemimiHanamiClusterParser/src/release -lKitsunemimiHanamiClusterParser
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiClusterParser/include

LIBS += -L../../libraries/libKitsunemimiHanamiPolicies/src -lKitsunemimiHanamiPolicies
LIBS += -L../../libraries/libKitsunemimiHanamiPolicies/src/debug -lKitsunemimiHanamiPolicies
LIBS += -L../../libraries/libKitsunemimiHanamiPolicies/src/release -lKitsunemimiHanamiPolicies
INCLUDEPATH += ../../libraries/libKitsunemimiHanamiPolicies/include

LIBS += -L../../libraries/libKitsunemimiArgs/src -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/debug -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/release -lKitsunemimiArgs
INCLUDEPATH += ../../libraries/libKitsunemimiArgs/include

LIBS += -L../../libraries/libKitsunemimiConfig/src -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/debug -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/release -lKitsunemimiConfig
INCLUDEPATH += ../../libraries/libKitsunemimiConfig/include

LIBS += -L../../libraries/../libraries/libKitsunemimiSakuraHardware/src -lKitsunemimiSakuraHardware
LIBS += -L../../libraries/../libraries/libKitsunemimiSakuraHardware/src/debug -lKitsunemimiSakuraHardware
LIBS += -L../../libraries/../libraries/libKitsunemimiSakuraHardware/src/release -lKitsunemimiSakuraHardware
INCLUDEPATH += ../../libraries/libKitsunemimiSakuraHardware/include

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

LIBS += -L../../libraries/../libraries/libKitsunemimiCpu/src -lKitsunemimiCpu
LIBS += -L../../libraries/../libraries/libKitsunemimiCpu/src/debug -lKitsunemimiCpu
LIBS += -L../../libraries/../libraries/libKitsunemimiCpu/src/release -lKitsunemimiCpu
INCLUDEPATH += ../../libraries/libKitsunemimiCpu/include

LIBS += -L../../sdk/cpp/libHanamiAiSdk/src -lHanamiAiSdk
LIBS += -L../../sdk/cpp/libHanamiAiSdk/src/debug -lHanamiAiSdk
LIBS += -L../../sdk/cpp/libHanamiAiSdk/src/release -lHanamiAiSdk
INCLUDEPATH += ../../sdk/cpp/libHanamiAiSdk/include

LIBS += -lcryptopp -lssl -lsqlite3 -luuid -lcrypto -pthread -lprotobuf -lOpenCL
LIBS +=  -L"/usr/local/cuda-12.1/targets/x86_64-linux/lib" -lcuda -lcudart -lcublas

INCLUDEPATH += $$PWD \
               src

HANAMI_PROTO_BUFFER = ../../libraries/libKitsunemimiHanamiMessages/protobuffers/hanami_messages.proto3
GPU_KERNEL = src/core/segments/core_segment/gpu_kernel.cl
CUDA_SOURCES = src/core/segments/core_segment/gpu_kernel.cu

OTHER_FILES += \
    $$CUDA_SOURCES

cudaKernel.input = CUDA_SOURCES
cudaKernel.output = ${QMAKE_FILE_BASE}.o
cudaKernel.commands = /usr/local/cuda-12.1/bin/nvcc -O3 -c  -o ${QMAKE_FILE_BASE}.o ${QMAKE_FILE_IN} || nvcc -O3 -c  -o ${QMAKE_FILE_BASE}.o ${QMAKE_FILE_IN}
cudaKernel.CONFIG += target_predeps
QMAKE_EXTRA_COMPILERS += cudaKernel

OTHER_FILES += $$HANAMI_PROTO_BUFFER \
               $$GPU_KERNEL

protobuf_decl.name = protobuf headers
protobuf_decl.name = protobuf headers
protobuf_decl.input = HANAMI_PROTO_BUFFER
protobuf_decl.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.h
protobuf_decl.commands = protoc --cpp_out=${QMAKE_FILE_IN_PATH} --proto_path=${QMAKE_FILE_IN_PATH} ${QMAKE_FILE_NAME}
protobuf_decl.variable_out = HEADERS
QMAKE_EXTRA_COMPILERS += protobuf_decl

protobuf_impl.name = protobuf sources
protobuf_impl.input = HANAMI_PROTO_BUFFER
protobuf_impl.output = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.cc
protobuf_impl.depends = ${QMAKE_FILE_IN_PATH}/${QMAKE_FILE_BASE}.proto3.pb.h
protobuf_impl.commands = $$escape_expand(\n)
protobuf_impl.variable_out = SOURCES
QMAKE_EXTRA_COMPILERS += protobuf_impl

gpu_processing.input = GPU_KERNEL
gpu_processing.output = ${QMAKE_FILE_BASE}.h
gpu_processing.commands = xxd -i ${QMAKE_FILE_IN} \
   | sed -E \'s/unsigned char.*\\\[\\\]/unsigned char gpu_kernel_cl\\\[\\\]/g\' \
   | sed -E \'s/unsigned int .* =/unsigned int gpu_kernel_cl_len =/g\' > ${QMAKE_FILE_BASE}.h
gpu_processing.variable_out = HEADERS
gpu_processing.CONFIG += no_link

QMAKE_EXTRA_COMPILERS += gpu_processing

DISTFILES += \
    src/core/segments/core_segment/gpu_kernel.cu

HEADERS += \
    src/api/endpoint_processing/blossom.h \
    src/api/endpoint_processing/http_processing/file_send.h \
    src/api/endpoint_processing/http_processing/http_processing.h \
    src/api/endpoint_processing/http_processing/response_builds.h \
    src/api/endpoint_processing/http_processing/string_functions.h \
    src/api/endpoint_processing/http_server.h \
    src/api/endpoint_processing/http_websocket_thread.h \
    src/api/endpoint_processing/items/item_methods.h \
    src/api/endpoint_processing/items/sakura_items.h \
    src/api/endpoint_processing/items/value_item_map.h \
    src/api/endpoint_processing/items/value_items.h \
    src/api/endpoint_processing/runtime_validation.h \
    src/api/v1/auth/create_token.h \
    src/api/v1/auth/list_user_projects.h \
    src/api/v1/auth/renew_token.h \
    src/api/v1/auth/validate_access.h \
    src/api/v1/blossom_initializing.h \
    src/api/v1/cluster/create_cluster.h \
    src/api/v1/cluster/delete_cluster.h \
    src/api/v1/cluster/list_cluster.h \
    src/api/v1/cluster/load_cluster.h \
    src/api/v1/cluster/save_cluster.h \
    src/api/v1/cluster/set_cluster_mode.h \
    src/api/v1/cluster/show_cluster.h \
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
    src/api/v1/documentation/generate_rest_api_docu.h \
    src/api/v1/logs/get_audit_log.h \
    src/api/v1/logs/get_error_log.h \
    src/api/v1/measurements/power_consumption.h \
    src/api/v1/measurements/speed.h \
    src/api/v1/measurements/temperature_production.h \
    src/api/v1/project/create_project.h \
    src/api/v1/project/delete_project.h \
    src/api/v1/project/get_project.h \
    src/api/v1/project/list_projects.h \
    src/api/v1/request_results/delete_request_result.h \
    src/api/v1/request_results/get_request_result.h \
    src/api/v1/request_results/list_request_result.h \
    src/api/v1/system_info/get_system_info.h \
    src/api/v1/task/create_task.h \
    src/api/v1/task/delete_task.h \
    src/api/v1/task/list_task.h \
    src/api/v1/task/show_task.h \
    src/api/v1/template/delete_template.h \
    src/api/v1/template/list_templates.h \
    src/api/v1/template/show_template.h \
    src/api/v1/template/upload_template.h \
    src/api/v1/threading/get_thread_mapping.h \
    src/api/v1/user/add_project_to_user.h \
    src/api/v1/user/create_user.h \
    src/api/v1/user/delete_user.h \
    src/api/v1/user/get_user.h \
    src/api/v1/user/list_users.h \
    src/api/v1/user/remove_project_from_user.h \
    src/args.h \
    src/callbacks.h \
    src/common.h \
    src/common/defines.h \
    src/common/enums.h \
    src/common/functions.h \
    src/common/structs.h \
    src/common/typedefs.h \
    src/config.h \
    src/core/callbacks.h \
    src/core/cluster/cluster.h \
    src/core/cluster/cluster_handler.h \
    src/core/cluster/cluster_init.h \
    src/core/cluster/statemachine_init.h \
    src/core/cluster/states/cycle_finish_state.h \
    src/core/cluster/states/graphs/graph_interpolation_state.h \
    src/core/cluster/states/graphs/graph_learn_forward_state.h \
    src/core/cluster/states/images/image_identify_state.h \
    src/core/cluster/states/images/image_learn_forward_state.h \
    src/core/cluster/states/snapshots/restore_cluster_state.h \
    src/core/cluster/states/snapshots/save_cluster_state.h \
    src/core/cluster/states/tables/table_interpolation_state.h \
    src/core/cluster/states/tables/table_learn_forward_state.h \
    src/core/cluster/states/task_handle_state.h \
    src/core/cluster/task.h \
    src/core/data_set_files/data_set_file.h \
    src/core/data_set_files/data_set_functions.h \
    src/core/data_set_files/image_data_set_file.h \
    src/core/data_set_files/table_data_set_file.h \
    src/core/power_measuring.h \
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
    src/core/speed_measuring.h \
    src/core/struct_validation.h \
    src/core/temp_file_handler.h \
    src/core/temperature_measuring.h \
    src/core/thread_binder.h \
    src/core/value_container.h \
    src/database/audit_log_table.h \
    src/database/cluster_snapshot_table.h \
    src/database/cluster_table.h \
    src/database/data_set_table.h \
    src/database/error_log_table.h \
    src/database/generic_tables/hanami_sql_admin_table.h \
    src/database/generic_tables/hanami_sql_log_table.h \
    src/database/generic_tables/hanami_sql_table.h \
    src/database/projects_table.h \
    src/database/request_result_table.h \
    src/database/template_table.h \
    src/database/users_table.h \
    src/hanami_root.h \
    src/io/protobuf_messages.h

SOURCES += \
    src/api/endpoint_processing/blossom.cpp \
    src/api/endpoint_processing/http_processing/file_send.cpp \
    src/api/endpoint_processing/http_processing/http_processing.cpp \
    src/api/endpoint_processing/http_server.cpp \
    src/api/endpoint_processing/http_websocket_thread.cpp \
    src/api/endpoint_processing/items/item_methods.cpp \
    src/api/endpoint_processing/items/sakura_items.cpp \
    src/api/endpoint_processing/items/value_item_map.cpp \
    src/api/endpoint_processing/runtime_validation.cpp \
    src/api/v1/auth/create_token.cpp \
    src/api/v1/auth/list_user_projects.cpp \
    src/api/v1/auth/renew_token.cpp \
    src/api/v1/auth/validate_access.cpp \
    src/api/v1/cluster/create_cluster.cpp \
    src/api/v1/cluster/delete_cluster.cpp \
    src/api/v1/cluster/list_cluster.cpp \
    src/api/v1/cluster/load_cluster.cpp \
    src/api/v1/cluster/save_cluster.cpp \
    src/api/v1/cluster/set_cluster_mode.cpp \
    src/api/v1/cluster/show_cluster.cpp \
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
    src/api/v1/documentation/generate_rest_api_docu.cpp \
    src/api/v1/logs/get_audit_log.cpp \
    src/api/v1/logs/get_error_log.cpp \
    src/api/v1/measurements/power_consumption.cpp \
    src/api/v1/measurements/speed.cpp \
    src/api/v1/measurements/temperature_production.cpp \
    src/api/v1/project/create_project.cpp \
    src/api/v1/project/delete_project.cpp \
    src/api/v1/project/get_project.cpp \
    src/api/v1/project/list_projects.cpp \
    src/api/v1/request_results/delete_request_result.cpp \
    src/api/v1/request_results/get_request_result.cpp \
    src/api/v1/request_results/list_request_result.cpp \
    src/api/v1/system_info/get_system_info.cpp \
    src/api/v1/task/create_task.cpp \
    src/api/v1/task/delete_task.cpp \
    src/api/v1/task/list_task.cpp \
    src/api/v1/task/show_task.cpp \
    src/api/v1/template/delete_template.cpp \
    src/api/v1/template/list_templates.cpp \
    src/api/v1/template/show_template.cpp \
    src/api/v1/template/upload_template.cpp \
    src/api/v1/threading/get_thread_mapping.cpp \
    src/api/v1/user/add_project_to_user.cpp \
    src/api/v1/user/create_user.cpp \
    src/api/v1/user/delete_user.cpp \
    src/api/v1/user/get_user.cpp \
    src/api/v1/user/list_users.cpp \
    src/api/v1/user/remove_project_from_user.cpp \
    src/callbacks.cpp \
    src/core/cluster/cluster.cpp \
    src/core/cluster/cluster_handler.cpp \
    src/core/cluster/cluster_init.cpp \
    src/core/cluster/statemachine_init.cpp \
    src/core/cluster/states/cycle_finish_state.cpp \
    src/core/cluster/states/graphs/graph_interpolation_state.cpp \
    src/core/cluster/states/graphs/graph_learn_forward_state.cpp \
    src/core/cluster/states/images/image_identify_state.cpp \
    src/core/cluster/states/images/image_learn_forward_state.cpp \
    src/core/cluster/states/snapshots/restore_cluster_state.cpp \
    src/core/cluster/states/snapshots/save_cluster_state.cpp \
    src/core/cluster/states/tables/table_interpolation_state.cpp \
    src/core/cluster/states/tables/table_learn_forward_state.cpp \
    src/core/cluster/states/task_handle_state.cpp \
    src/core/data_set_files/data_set_file.cpp \
    src/core/data_set_files/data_set_functions.cpp \
    src/core/data_set_files/image_data_set_file.cpp \
    src/core/data_set_files/table_data_set_file.cpp \
    src/core/power_measuring.cpp \
    src/core/processing/cpu_processing_unit.cpp \
    src/core/processing/processing_unit_handler.cpp \
    src/core/processing/segment_queue.cpp \
    src/core/segments/abstract_segment.cpp \
    src/core/segments/core_segment/core_segment.cpp \
    src/core/segments/input_segment/input_segment.cpp \
    src/core/segments/output_segment/output_segment.cpp \
    src/core/speed_measuring.cpp \
    src/core/temp_file_handler.cpp \
    src/core/temperature_measuring.cpp \
    src/core/thread_binder.cpp \
    src/core/value_container.cpp \
    src/database/audit_log_table.cpp \
    src/database/cluster_snapshot_table.cpp \
    src/database/cluster_table.cpp \
    src/database/data_set_table.cpp \
    src/database/error_log_table.cpp \
    src/database/generic_tables/hanami_sql_admin_table.cpp \
    src/database/generic_tables/hanami_sql_log_table.cpp \
    src/database/generic_tables/hanami_sql_table.cpp \
    src/database/projects_table.cpp \
    src/database/request_result_table.cpp \
    src/database/template_table.cpp \
    src/database/users_table.cpp \
    src/hanami_root.cpp \
    src/io/protobuf_messages.cpp \
    src/main.cpp
