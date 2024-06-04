cmake_minimum_required(VERSION 3.16)
project(Hanami CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Setting the compiler
set(CMAKE_CXX_COMPILER "clang++-15")

# Adding source files
set(SOURCES
    src/api/endpoint_processing/blossom.h
    src/api/endpoint_processing/http_processing/file_send.h
    src/api/endpoint_processing/http_processing/http_processing.h
    src/api/endpoint_processing/http_processing/response_builds.h
    src/api/endpoint_processing/http_processing/string_functions.h
    src/api/endpoint_processing/http_server.h
    src/api/endpoint_processing/http_websocket_thread.h
    src/api/endpoint_processing/runtime_validation.h
    src/api/http/v1/auth/create_token.h
    src/api/http/v1/auth/renew_token.h
    src/api/http/v1/auth/validate_access.h
    src/api/http/v1/blossom_initializing.h
    src/api/http/v1/checkpoint/delete_checkpoint.h
    src/api/http/v1/checkpoint/get_checkpoint.h
    src/api/http/v1/checkpoint/list_checkpoint.h
    src/api/http/v1/cluster/create_cluster.h
    src/api/http/v1/cluster/delete_cluster.h
    src/api/http/v1/cluster/list_cluster.h
    src/api/http/v1/cluster/load_cluster.h
    src/api/http/v1/cluster/save_cluster.h
    src/api/http/v1/cluster/set_cluster_mode.h
    src/api/http/v1/cluster/show_cluster.h
    src/api/http/v1/cluster/switch_hosts.h
    src/api/http/v1/data_files/check_dataset.h
    src/api/http/v1/data_files/csv/create_csv_dataset.h
    src/api/http/v1/data_files/csv/finalize_csv_dataset.h
    src/api/http/v1/data_files/delete_dataset.h
    src/api/http/v1/data_files/get_dataset.h
    src/api/http/v1/data_files/get_progress_dataset.h
    src/api/http/v1/data_files/list_dataset.h
    src/api/http/v1/data_files/mnist/create_mnist_dataset.h
    src/api/http/v1/data_files/mnist/finalize_mnist_dataset.h
    src/api/http/v1/hosts/list_hosts.h
    src/api/http/v1/logs/get_audit_log.h
    src/api/http/v1/logs/get_error_log.h
    src/api/http/v1/measurements/power_consumption.h
    src/api/http/v1/measurements/speed.h
    src/api/http/v1/measurements/temperature_production.h
    src/api/http/v1/project/create_project.h
    src/api/http/v1/project/delete_project.h
    src/api/http/v1/project/get_project.h
    src/api/http/v1/project/list_projects.h
    src/api/http/v1/request_results/delete_request_result.h
    src/api/http/v1/request_results/get_request_result.h
    src/api/http/v1/request_results/list_request_result.h
    src/api/http/v1/system_info/get_system_info.h
    src/api/http/v1/task/create_task.h
    src/api/http/v1/task/delete_task.h
    src/api/http/v1/task/list_task.h
    src/api/http/v1/task/show_task.h
    src/api/http/v1/threading/get_thread_mapping.h
    src/api/http/v1/user/add_project_to_user.h
    src/api/http/v1/user/create_user.h
    src/api/http/v1/user/delete_user.h
    src/api/http/v1/user/get_user.h
    src/api/http/v1/user/list_users.h
    src/api/http/v1/user/list_user_projects.h
    src/api/http/v1/user/remove_project_from_user.h
    src/api/websocket/cluster_io.h
    src/api/websocket/file_upload.h
    src/args.h
    src/config.h
    src/core/cluster/add_tasks.h
    src/core/cluster/cluster.h
    src/core/cluster/cluster_handler.h
    src/core/cluster/cluster_init.h
    src/core/cluster/objects.h
    src/core/cluster/statemachine_init.h
    src/core/cluster/states/checkpoints/restore_cluster_state.h
    src/core/cluster/states/checkpoints/save_cluster_state.h
    src/core/cluster/states/cycle_finish_state.h
    src/core/cluster/states/graphs/graph_interpolation_state.h
    src/core/cluster/states/graphs/graph_train_forward_state.h
    src/core/cluster/states/images/image_identify_state.h
    src/core/cluster/states/images/image_train_forward_state.h
    src/core/cluster/states/tables/table_interpolation_state.h
    src/core/cluster/states/tables/table_train_forward_state.h
    src/core/cluster/states/task_handle_state.h
    src/core/cluster/task.h
    src/core/cuda_functions.h
    src/core/io/buffer/buffer_io.h
    src/core/io/disc/checkpoint_io.h
    src/core/io/io_interface.h
    src/core/processing/cluster_io_functions.h
    src/core/processing/cluster_resize.h
    src/core/processing/cpu/backpropagation.h
    src/core/processing/cpu/cpu_host.h
    src/core/processing/cpu/processing.h
    src/core/processing/cpu/reduction.h
    src/core/processing/cpu/worker_thread.h
    src/core/processing/cuda/cuda_host.h
    src/core/processing/logical_host.h
    src/core/processing/physical_host.h
    src/core/routing_functions.h
    src/core/temp_file_handler.h
    src/core/thread_binder.h
    src/database/audit_log_table.h
    src/database/checkpoint_table.h
    src/database/cluster_table.h
    src/database/dataset_table.h
    src/database/error_log_table.h
    src/database/generic_tables/hanami_sql_admin_table.h
    src/database/generic_tables/hanami_sql_log_table.h
    src/database/generic_tables/hanami_sql_table.h
    src/database/projects_table.h
    src/database/request_result_table.h
    src/database/tempfile_table.h
    src/database/users_table.h
    src/documentation/generate_rest_api_docu.h
    src/hanami_root.h
    src/api/endpoint_processing/blossom.cpp
    src/api/endpoint_processing/http_processing/file_send.cpp
    src/api/endpoint_processing/http_processing/http_processing.cpp
    src/api/endpoint_processing/http_server.cpp
    src/api/endpoint_processing/http_websocket_thread.cpp
    src/api/endpoint_processing/runtime_validation.cpp
    src/api/http/v1/auth/create_token.cpp
    src/api/http/v1/auth/renew_token.cpp
    src/api/http/v1/auth/validate_access.cpp
    src/api/http/v1/checkpoint/delete_checkpoint.cpp
    src/api/http/v1/checkpoint/get_checkpoint.cpp
    src/api/http/v1/checkpoint/list_checkpoint.cpp
    src/api/http/v1/cluster/create_cluster.cpp
    src/api/http/v1/cluster/delete_cluster.cpp
    src/api/http/v1/cluster/list_cluster.cpp
    src/api/http/v1/cluster/load_cluster.cpp
    src/api/http/v1/cluster/save_cluster.cpp
    src/api/http/v1/cluster/set_cluster_mode.cpp
    src/api/http/v1/cluster/show_cluster.cpp
    src/api/http/v1/cluster/switch_hosts.cpp
    src/api/http/v1/data_files/check_dataset.cpp
    src/api/http/v1/data_files/csv/create_csv_dataset.cpp
    src/api/http/v1/data_files/csv/finalize_csv_dataset.cpp
    src/api/http/v1/data_files/delete_dataset.cpp
    src/api/http/v1/data_files/get_dataset.cpp
    src/api/http/v1/data_files/get_progress_dataset.cpp
    src/api/http/v1/data_files/list_dataset.cpp
    src/api/http/v1/data_files/mnist/create_mnist_dataset.cpp
    src/api/http/v1/data_files/mnist/finalize_mnist_dataset.cpp
    src/api/http/v1/hosts/list_hosts.cpp
    src/api/http/v1/logs/get_audit_log.cpp
    src/api/http/v1/logs/get_error_log.cpp
    src/api/http/v1/measurements/power_consumption.cpp
    src/api/http/v1/measurements/speed.cpp
    src/api/http/v1/measurements/temperature_production.cpp
    src/api/http/v1/project/create_project.cpp
    src/api/http/v1/project/delete_project.cpp
    src/api/http/v1/project/get_project.cpp
    src/api/http/v1/project/list_projects.cpp
    src/api/http/v1/request_results/delete_request_result.cpp
    src/api/http/v1/request_results/get_request_result.cpp
    src/api/http/v1/request_results/list_request_result.cpp
    src/api/http/v1/system_info/get_system_info.cpp
    src/api/http/v1/task/create_task.cpp
    src/api/http/v1/task/delete_task.cpp
    src/api/http/v1/task/list_task.cpp
    src/api/http/v1/task/show_task.cpp
    src/api/http/v1/threading/get_thread_mapping.cpp
    src/api/http/v1/user/add_project_to_user.cpp
    src/api/http/v1/user/create_user.cpp
    src/api/http/v1/user/delete_user.cpp
    src/api/http/v1/user/get_user.cpp
    src/api/http/v1/user/list_users.cpp
    src/api/http/v1/user/list_user_projects.cpp
    src/api/http/v1/user/remove_project_from_user.cpp
    src/api/websocket/cluster_io.cpp
    src/api/websocket/file_upload.cpp
    src/core/cluster/add_tasks.cpp
    src/core/cluster/cluster.cpp
    src/core/cluster/cluster_handler.cpp
    src/core/cluster/cluster_init.cpp
    src/core/cluster/statemachine_init.cpp
    src/core/cluster/states/checkpoints/restore_cluster_state.cpp
    src/core/cluster/states/checkpoints/save_cluster_state.cpp
    src/core/cluster/states/cycle_finish_state.cpp
    src/core/cluster/states/graphs/graph_interpolation_state.cpp
    src/core/cluster/states/graphs/graph_train_forward_state.cpp
    src/core/cluster/states/images/image_identify_state.cpp
    src/core/cluster/states/images/image_train_forward_state.cpp
    src/core/cluster/states/tables/table_interpolation_state.cpp
    src/core/cluster/states/tables/table_train_forward_state.cpp
    src/core/cluster/states/task_handle_state.cpp
    src/core/io/buffer/buffer_io.cpp
    src/core/io/disc/checkpoint_io.cpp
    src/core/io/io_interface.cpp
    src/core/processing/cpu/cpu_host.cpp
    src/core/processing/cpu/worker_thread.cpp
    src/core/processing/cuda/cuda_host.cpp
    src/core/processing/logical_host.cpp
    src/core/processing/physical_host.cpp
    src/core/temp_file_handler.cpp
    src/core/thread_binder.cpp
    src/database/audit_log_table.cpp
    src/database/checkpoint_table.cpp
    src/database/cluster_table.cpp
    src/database/dataset_table.cpp
    src/database/error_log_table.cpp
    src/database/generic_tables/hanami_sql_admin_table.cpp
    src/database/generic_tables/hanami_sql_log_table.cpp
    src/database/generic_tables/hanami_sql_table.cpp
    src/database/projects_table.cpp
    src/database/request_result_table.cpp
    src/database/tempfile_table.cpp
    src/database/users_table.cpp
    src/documentation/generate_rest_api_docu.cpp
    src/hanami_root.cpp
)

# Adding the executable
add_executable(Hanami
    ${SOURCES}
    src/main.cpp
    # ${CUDA_OBJECTS}
    )
add_executable(Hanami_unittests
    ${SOURCES}
    tests/unit_tests/database/cluster_table_test.h
    tests/unit_tests/database/cluster_table_test.cpp
    tests/unit_tests/database/audit_log_table_test.h
    tests/unit_tests/database/audit_log_table_test.cpp
    tests/unit_tests/database/checkpoint_table_test.h
    tests/unit_tests/database/checkpoint_table_test.cpp
    tests/unit_tests/database/dataset_table_test.h
    tests/unit_tests/database/dataset_table_test.cpp
    tests/unit_tests/database/error_log_table_test.h
    tests/unit_tests/database/error_log_table_test.cpp
    tests/unit_tests/database/projects_table_test.h
    tests/unit_tests/database/projects_table_test.cpp
    tests/unit_tests/database/request_result_table_test.h
    tests/unit_tests/database/request_result_table_test.cpp
    tests/unit_tests/database/tempfile_table_test.h
    tests/unit_tests/database/tempfile_table_test.cpp
    tests/unit_tests/database/users_table_test.h
    tests/unit_tests/database/users_table_test.cpp
    tests/unit_tests/core/cluster_test.h
    tests/unit_tests/core/cluster_test.cpp
    tests/unit_tests/main.cpp
    # ${CUDA_OBJECTS}
    )

# Including directories
include_directories(
    src/
    ../libraries/hanami_messages/protobuffers
    ../libraries/hanami_cpu/include
    ../libraries/hanami_hardware/include
    ../libraries/hanami_common/include
    ../libraries/hanami_ini/include
    ../libraries/hanami_config/include
    ../libraries/hanami_sqlite/include
    ../libraries/hanami_database/include
    ../libraries/hanami_files/include
    ../libraries/hanami_crypto/include
    ../libraries/hanami_args/include
    ../libraries/hanami_policies/include
    ../libraries/hanami_cluster_parser/include
    ../third-party-libs/jwt-cpp/include
    ../third-party-libs/json/include
)

# Finding required packages or libraries
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
find_library(CUDA_LIB cuda)
find_library(CUDART_LIB cudart)
find_library(SSL_LIB ssl)
find_library(CRYPTO_LIB crypto)
find_library(SQLITE3_LIB sqlite3)
find_library(UUID_LIB uuid)
find_library(PROTOBUF_LIB protobuf)

# Linking libraries to the executable
target_link_libraries(Hanami
    hanami_cpu
    hanami_hardware
    hanami_common
    hanami_ini
    hanami_config
    hanami_sqlite
    hanami_database
    hanami_files
    hanami_crypto
    hanami_args
    hanami_policies
    hanami_cluster_parser
    ${SSL_LIB}
    ${CRYPTO_LIB}
    ${SQLITE3_LIB}
    ${UUID_LIB}
    ${CUDA_LIB}
    ${CUDART_LIB}
    ${PROTOBUF_LIB}
    Threads::Threads
    OpenSSL::SSL
)
target_link_libraries(Hanami_unittests
    hanami_cpu
    hanami_hardware
    hanami_common
    hanami_ini
    hanami_config
    hanami_sqlite
    hanami_database
    hanami_files
    hanami_crypto
    hanami_args
    hanami_policies
    hanami_cluster_parser
    ${SSL_LIB}
    ${CRYPTO_LIB}
    ${SQLITE3_LIB}
    ${UUID_LIB}
    ${CUDA_LIB}
    ${CUDART_LIB}
    ${PROTOBUF_LIB}
    Threads::Threads
    OpenSSL::SSL
)

# Set the path to your .proto file
set(PROTO_FILE ../libraries/hanami_messages/protobuffers)

# Set the output directory for generated files
set(PROTO_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROTO_OUTPUT_HEADER ${PROTO_OUTPUT_DIR}/${PROTO_FILE}/hanami_messages.proto3.pb.h)
set(PROTO_OUTPUT_SOURCE ${PROTO_OUTPUT_DIR}/${PROTO_FILE}/hanami_messages.proto3.pb.cc)
# set(CUDA_SOURCES src/core/processing/cuda/gpu_kernel.cu)
# get_filename_component(CUDA_SRC_NAME ${CUDA_SOURCES} NAME_WE)
# set(CUDA_OUTPUT_OBJ ${CMAKE_CURRENT_BINARY_DIR}/${CUDA_SRC_NAME}.o)

# Generate C++ headers and sources from the .proto file
add_custom_command(
    OUTPUT
        ${PROTO_OUTPUT_HEADER}
        ${PROTO_OUTPUT_SOURCE}
    COMMAND protoc
        --cpp_out=${PROTO_OUTPUT_DIR}/${PROTO_FILE}
        --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}/hanami_messages.proto3
    # COMMAND nvcc
    #     -O3
    #     -c
    #     -I${CMAKE_CURRENT_SOURCE_DIR}/../libraries/hanami_common/include
    #     -o ${CUDA_OUTPUT_OBJ}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCES}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE}/hanami_messages.proto3
)

# Add dependencies on the targets representing the generated files
add_custom_target(generate_protobuf_files DEPENDS
    ${PROTO_OUTPUT_HEADER}
    ${PROTO_OUTPUT_SOURCE}
    # ${CUDA_SOURCES}
)
add_dependencies(Hanami generate_protobuf_files)
target_sources(Hanami PRIVATE
    ${PROTO_OUTPUT_HEADER}
    ${PROTO_OUTPUT_SOURCE}
    # ${CUDA_SOURCES}
)
add_dependencies(Hanami_unittests generate_protobuf_files)
target_sources(Hanami_unittests PRIVATE
    ${PROTO_OUTPUT_HEADER}
    ${PROTO_OUTPUT_SOURCE}
    # ${CUDA_SOURCES}
)

