QT -= qt core gui

TARGET = Hanami
CONFIG += console
CONFIG += c++17

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

INCLUDEPATH += ../libraries/hanami_messages/protobuffers

LIBS += -L../libraries/hanami_hardware/src -lhanami_hardware
LIBS += -L../libraries/hanami_hardware/src/debug -lhanami_hardware
LIBS += -L../libraries/hanami_hardware/src/release -lhanami_hardware
INCLUDEPATH += ../libraries/hanami_hardware/include

LIBS += -L../libraries/hanami_files/src -lhanami_files
LIBS += -L../libraries/hanami_files/src/debug -lhanami_files
LIBS += -L../libraries/hanami_files/src/release -lhanami_files
INCLUDEPATH += ../libraries/hanami_files/include

LIBS += -L../libraries/hanami_cluster_parser/src -lhanami_cluster_parser
LIBS += -L../libraries/hanami_cluster_parser/src/debug -lhanami_cluster_parser
LIBS += -L../libraries/hanami_cluster_parser/src/release -lhanami_cluster_parser
INCLUDEPATH += ../libraries/hanami_cluster_parser/include

LIBS += -L../libraries/hanami_policies/src -lhanami_policies
LIBS += -L../libraries/hanami_policies/src/debug -lhanami_policies
LIBS += -L../libraries/hanami_policies/src/release -lhanami_policies
INCLUDEPATH += ../libraries/hanami_policies/include

LIBS += -L../libraries/hanami_args/src -lhanami_args
LIBS += -L../libraries/hanami_args/src/debug -lhanami_args
LIBS += -L../libraries/hanami_args/src/release -lhanami_args
INCLUDEPATH += ../libraries/hanami_args/include

LIBS += -L../libraries/hanami_config/src -lhanami_config
LIBS += -L../libraries/hanami_config/src/debug -lhanami_config
LIBS += -L../libraries/hanami_config/src/release -lhanami_config
INCLUDEPATH += ../libraries/hanami_config/include

LIBS += -L../libraries/hanami_database/src -lhanami_database
LIBS += -L../libraries/hanami_database/src/debug -lhanami_database
LIBS += -L../libraries/hanami_database/src/release -lhanami_database
INCLUDEPATH += ../libraries/hanami_database/include

LIBS += -L../libraries/hanami_common/src -lhanami_common
LIBS += -L../libraries/hanami_common/src/debug -lhanami_common
LIBS += -L../libraries/hanami_common/src/release -lhanami_common
INCLUDEPATH += ../libraries/hanami_common/include

LIBS += -L../libraries/hanami_sqlite/src -lhanami_sqlite
LIBS += -L../libraries/hanami_sqlite/src/debug -lhanami_sqlite
LIBS += -L../libraries/hanami_sqlite/src/release -lhanami_sqlite
INCLUDEPATH += ../libraries/hanami_sqlite/include

LIBS += -L../libraries/hanami_ini/src -lhanami_ini
LIBS += -L../libraries/hanami_ini/src/debug -lhanami_ini
LIBS += -L../libraries/hanami_ini/src/release -lhanami_ini
INCLUDEPATH += ../libraries/hanami_ini/include

LIBS += -L../libraries/hanami_crypto/src -lhanami_crypto
LIBS += -L../libraries/hanami_crypto/src/debug -lhanami_crypto
LIBS += -L../libraries/hanami_crypto/src/release -lhanami_crypto
INCLUDEPATH += ../libraries/hanami_crypto/include

LIBS += -L../libraries/../libraries/hanami_cpu/src -lhanami_cpu
LIBS += -L../libraries/../libraries/hanami_cpu/src/debug -lhanami_cpu
LIBS += -L../libraries/../libraries/hanami_cpu/src/release -lhanami_cpu
INCLUDEPATH += ../libraries/hanami_cpu/include

INCLUDEPATH += ../third-party-libs/jwt-cpp/include
INCLUDEPATH += ../third-party-libs/json/include

LIBS += -lcryptopp -lcrypto -lssl -lsqlite3 -luuid -pthread -lprotobuf
# LIBS += -lOpenCL
LIBS +=  -L"/usr/local/cuda-12.1/targets/x86_64-linux/lib"  -L"/usr/local/cuda-12.2/targets/x86_64-linux/lib" -lcuda -lcudart
INCLUDEPATH += $$PWD \
               src

HANAMI_PROTO_BUFFER = ../libraries/hanami_messages/protobuffers/hanami_messages.proto3
# GPU_KERNEL = src/core/processing/opencl/gpu_kernel.cl
CUDA_SOURCES = src/core/processing/cuda/gpu_kernel.cu

OTHER_FILES += \
    $$CUDA_SOURCES

cudaKernel.input = CUDA_SOURCES
cudaKernel.output = ${QMAKE_FILE_BASE}.o
cudaKernel.commands = /usr/local/cuda-12.1/bin/nvcc -O3 -c -I$$PWD/../libraries/hanami_common/include -o ${QMAKE_FILE_BASE}.o ${QMAKE_FILE_IN} \
                      || /usr/local/cuda-12.2/bin/nvcc -O3 -c -I$$PWD/../libraries/hanami_common/include -o ${QMAKE_FILE_BASE}.o ${QMAKE_FILE_IN} \
                      || nvcc -O3 -c -I$$PWD/../libraries/hanami_common/include -o ${QMAKE_FILE_BASE}.o ${QMAKE_FILE_IN}
cudaKernel.CONFIG += target_predeps
QMAKE_EXTRA_COMPILERS += cudaKernel

OTHER_FILES += $$HANAMI_PROTO_BUFFER \
               $$CUDA_SOURCES

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

# gpu_processing.input = GPU_KERNEL
# gpu_processing.output = ${QMAKE_FILE_BASE}.h
# gpu_processing.commands = xxd -i ${QMAKE_FILE_IN} \
#    | sed -E \'s/unsigned char.*\\\[\\\]/unsigned char gpu_kernel_cl\\\[\\\]/g\' \
#    | sed -E \'s/unsigned int .* =/unsigned int gpu_kernel_cl_len =/g\' > ${QMAKE_FILE_BASE}.h
# gpu_processing.variable_out = HEADERS
# gpu_processing.CONFIG += no_link
# QMAKE_EXTRA_COMPILERS += gpu_processing

HEADERS += \
    src/api/endpoint_processing/blossom.h \
    src/api/endpoint_processing/http_processing/file_send.h \
    src/api/endpoint_processing/http_processing/http_processing.h \
    src/api/endpoint_processing/http_processing/response_builds.h \
    src/api/endpoint_processing/http_processing/string_functions.h \
    src/api/endpoint_processing/http_server.h \
    src/api/endpoint_processing/http_websocket_thread.h \
    src/api/endpoint_processing/runtime_validation.h \
    src/api/http/v1/auth/create_token.h \
    src/api/http/v1/auth/list_user_projects.h \
    src/api/http/v1/auth/renew_token.h \
    src/api/http/v1/auth/validate_access.h \
    src/api/http/v1/blossom_initializing.h \
    src/api/http/v1/cluster/create_cluster.h \
    src/api/http/v1/cluster/delete_cluster.h \
    src/api/http/v1/cluster/list_cluster.h \
    src/api/http/v1/cluster/load_cluster.h \
    src/api/http/v1/cluster/save_cluster.h \
    src/api/http/v1/cluster/set_cluster_mode.h \
    src/api/http/v1/cluster/show_cluster.h \
    src/api/http/v1/checkpoint/delete_checkpoint.h \
    src/api/http/v1/checkpoint/get_checkpoint.h \
    src/api/http/v1/checkpoint/list_checkpoint.h \
    src/api/http/v1/data_files/check_data_set.h \
    src/api/http/v1/data_files/csv/create_csv_data_set.h \
    src/api/http/v1/data_files/csv/finalize_csv_data_set.h \
    src/api/http/v1/data_files/delete_data_set.h \
    src/api/http/v1/data_files/get_data_set.h \
    src/api/http/v1/data_files/get_progress_data_set.h \
    src/api/http/v1/data_files/list_data_set.h \
    src/api/http/v1/data_files/mnist/create_mnist_data_set.h \
    src/api/http/v1/data_files/mnist/finalize_mnist_data_set.h \
    src/common/using.h \
    src/database/tempfile_table.h \
    src/documentation/generate_rest_api_docu.h \
    src/api/http/v1/logs/get_audit_log.h \
    src/api/http/v1/logs/get_error_log.h \
    src/api/http/v1/measurements/power_consumption.h \
    src/api/http/v1/measurements/speed.h \
    src/api/http/v1/measurements/temperature_production.h \
    src/api/http/v1/project/create_project.h \
    src/api/http/v1/project/delete_project.h \
    src/api/http/v1/project/get_project.h \
    src/api/http/v1/project/list_projects.h \
    src/api/http/v1/request_results/delete_request_result.h \
    src/api/http/v1/request_results/get_request_result.h \
    src/api/http/v1/request_results/list_request_result.h \
    src/api/http/v1/system_info/get_system_info.h \
    src/api/http/v1/task/create_task.h \
    src/api/http/v1/task/delete_task.h \
    src/api/http/v1/task/list_task.h \
    src/api/http/v1/task/show_task.h \
    src/api/http/v1/threading/get_thread_mapping.h \
    src/api/http/v1/user/add_project_to_user.h \
    src/api/http/v1/user/create_user.h \
    src/api/http/v1/user/delete_user.h \
    src/api/http/v1/user/get_user.h \
    src/api/http/v1/user/list_users.h \
    src/api/http/v1/user/remove_project_from_user.h \
    src/api/websocket/cluster_io.h \
    src/api/websocket/file_upload.h \
    src/args.h \
    src/common.h \
    src/common/defines.h \
    src/common/enums.h \
    src/common/functions.h \
    src/common/structs.h \
    src/common/typedefs.h \
    src/config.h \
    src/core/cluster/add_tasks.h \
    src/core/cluster/cluster.h \
    src/core/cluster/cluster_handler.h \
    src/core/cluster/cluster_init.h \
    src/core/cluster/statemachine_init.h \
    src/core/cluster/states/cycle_finish_state.h \
    src/core/cluster/states/graphs/graph_interpolation_state.h \
    src/core/cluster/states/graphs/graph_train_forward_state.h \
    src/core/cluster/states/images/image_identify_state.h \
    src/core/cluster/states/images/image_train_forward_state.h \
    src/core/cluster/states/checkpoints/restore_cluster_state.h \
    src/core/cluster/states/checkpoints/save_cluster_state.h \
    src/core/cluster/states/tables/table_interpolation_state.h \
    src/core/cluster/states/tables/table_train_forward_state.h \
    src/core/cluster/states/task_handle_state.h \
    src/core/cluster/task.h \
    src/core/processing/cluster_queue.h \
    src/core/processing/cluster_io_functions.h \
    src/core/processing/cpu/backpropagation.h \
    src/core/processing/cpu/processing.h \
    src/core/processing/cpu/reduction.h \
    src/core/processing/cpu_processing_unit.h \
    src/core/processing/objects.h \
    src/core/processing/processing_unit_handler.h \
    src/core/processing/section_update.h \
    src/core/routing_functions.h \
    src/core/temp_file_handler.h \
    src/core/thread_binder.h \
    src/database/audit_log_table.h \
    src/database/checkpoint_table.h \
    src/database/cluster_table.h \
    src/database/data_set_table.h \
    src/database/error_log_table.h \
    src/database/generic_tables/hanami_sql_admin_table.h \
    src/database/generic_tables/hanami_sql_log_table.h \
    src/database/generic_tables/hanami_sql_table.h \
    src/database/projects_table.h \
    src/database/request_result_table.h \
    src/database/users_table.h \
    src/hanami_root.h

SOURCES += \
    src/api/endpoint_processing/blossom.cpp \
    src/api/endpoint_processing/http_processing/file_send.cpp \
    src/api/endpoint_processing/http_processing/http_processing.cpp \
    src/api/endpoint_processing/http_server.cpp \
    src/api/endpoint_processing/http_websocket_thread.cpp \
    src/api/endpoint_processing/runtime_validation.cpp \
    src/api/http/v1/auth/create_token.cpp \
    src/api/http/v1/auth/list_user_projects.cpp \
    src/api/http/v1/auth/renew_token.cpp \
    src/api/http/v1/auth/validate_access.cpp \
    src/api/http/v1/cluster/create_cluster.cpp \
    src/api/http/v1/cluster/delete_cluster.cpp \
    src/api/http/v1/cluster/list_cluster.cpp \
    src/api/http/v1/cluster/load_cluster.cpp \
    src/api/http/v1/cluster/save_cluster.cpp \
    src/api/http/v1/cluster/set_cluster_mode.cpp \
    src/api/http/v1/cluster/show_cluster.cpp \
    src/api/http/v1/checkpoint/delete_checkpoint.cpp \
    src/api/http/v1/checkpoint/get_checkpoint.cpp \
    src/api/http/v1/checkpoint/list_checkpoint.cpp \
    src/api/http/v1/data_files/check_data_set.cpp \
    src/api/http/v1/data_files/csv/create_csv_data_set.cpp \
    src/api/http/v1/data_files/csv/finalize_csv_data_set.cpp \
    src/api/http/v1/data_files/delete_data_set.cpp \
    src/api/http/v1/data_files/get_data_set.cpp \
    src/api/http/v1/data_files/get_progress_data_set.cpp \
    src/api/http/v1/data_files/list_data_set.cpp \
    src/api/http/v1/data_files/mnist/create_mnist_data_set.cpp \
    src/api/http/v1/data_files/mnist/finalize_mnist_data_set.cpp \
    src/database/tempfile_table.cpp \
    src/documentation/generate_rest_api_docu.cpp \
    src/api/http/v1/logs/get_audit_log.cpp \
    src/api/http/v1/logs/get_error_log.cpp \
    src/api/http/v1/measurements/power_consumption.cpp \
    src/api/http/v1/measurements/speed.cpp \
    src/api/http/v1/measurements/temperature_production.cpp \
    src/api/http/v1/project/create_project.cpp \
    src/api/http/v1/project/delete_project.cpp \
    src/api/http/v1/project/get_project.cpp \
    src/api/http/v1/project/list_projects.cpp \
    src/api/http/v1/request_results/delete_request_result.cpp \
    src/api/http/v1/request_results/get_request_result.cpp \
    src/api/http/v1/request_results/list_request_result.cpp \
    src/api/http/v1/system_info/get_system_info.cpp \
    src/api/http/v1/task/create_task.cpp \
    src/api/http/v1/task/delete_task.cpp \
    src/api/http/v1/task/list_task.cpp \
    src/api/http/v1/task/show_task.cpp \
    src/api/http/v1/threading/get_thread_mapping.cpp \
    src/api/http/v1/user/add_project_to_user.cpp \
    src/api/http/v1/user/create_user.cpp \
    src/api/http/v1/user/delete_user.cpp \
    src/api/http/v1/user/get_user.cpp \
    src/api/http/v1/user/list_users.cpp \
    src/api/http/v1/user/remove_project_from_user.cpp \
    src/api/websocket/cluster_io.cpp \
    src/api/websocket/file_upload.cpp \
    src/core/cluster/add_tasks.cpp \
    src/core/cluster/cluster.cpp \
    src/core/cluster/cluster_handler.cpp \
    src/core/cluster/cluster_init.cpp \
    src/core/cluster/statemachine_init.cpp \
    src/core/cluster/states/cycle_finish_state.cpp \
    src/core/cluster/states/graphs/graph_interpolation_state.cpp \
    src/core/cluster/states/graphs/graph_train_forward_state.cpp \
    src/core/cluster/states/images/image_identify_state.cpp \
    src/core/cluster/states/images/image_train_forward_state.cpp \
    src/core/cluster/states/checkpoints/restore_cluster_state.cpp \
    src/core/cluster/states/checkpoints/save_cluster_state.cpp \
    src/core/cluster/states/tables/table_interpolation_state.cpp \
    src/core/cluster/states/tables/table_train_forward_state.cpp \
    src/core/cluster/states/task_handle_state.cpp \
    src/core/processing/cluster_queue.cpp \
    src/core/processing/cpu_processing_unit.cpp \
    src/core/processing/processing_unit_handler.cpp \
    src/core/temp_file_handler.cpp \
    src/core/thread_binder.cpp \
    src/database/audit_log_table.cpp \
    src/database/checkpoint_table.cpp \
    src/database/cluster_table.cpp \
    src/database/data_set_table.cpp \
    src/database/error_log_table.cpp \
    src/database/generic_tables/hanami_sql_admin_table.cpp \
    src/database/generic_tables/hanami_sql_log_table.cpp \
    src/database/generic_tables/hanami_sql_table.cpp \
    src/database/projects_table.cpp \
    src/database/request_result_table.cpp \
    src/database/users_table.cpp \
    src/hanami_root.cpp \
    src/main.cpp
