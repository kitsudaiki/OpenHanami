QT -= qt core gui

TARGET = SDK_API_Testing
CONFIG += console c++17
CONFIG -= app_bundle

LIBS += -L../../sdk/cpp/libHanamiAiSdk/src -lHanamiAiSdk
LIBS += -L../../sdk/cpp/libHanamiAiSdk/src/debug -lHanamiAiSdk
LIBS += -L../../sdk/cpp/libHanamiAiSdk/src/release -lHanamiAiSdk
INCLUDEPATH += ../../sdk/cpp/libHanamiAiSdk/include

LIBS += -L../../libraries/hanami_config/src -lhanami_config
LIBS += -L../../libraries/hanami_config/src/debug -lhanami_config
LIBS += -L../../libraries/hanami_config/src/release -lhanami_config
INCLUDEPATH += ../../libraries/hanami_config/include

LIBS += -L../../libraries/hanami_json/src -lhanami_json
LIBS += -L../../libraries/hanami_json/src/debug -lhanami_json
LIBS += -L../../libraries/hanami_json/src/release -lhanami_json
INCLUDEPATH += ../../libraries/hanami_json/include

LIBS += -L../../libraries/hanami_ini/src -lhanami_ini
LIBS += -L../../libraries/hanami_ini/src/debug -lhanami_ini
LIBS += -L../../libraries/hanami_ini/src/release -lhanami_ini
INCLUDEPATH += ../../libraries/hanami_ini/include

LIBS += -L../../libraries/hanami_args/src -lhanami_args
LIBS += -L../../libraries/hanami_args/src/debug -lhanami_args
LIBS += -L../../libraries/hanami_args/src/release -lhanami_args
INCLUDEPATH += ../../libraries/hanami_args/include

LIBS += -L../../libraries/hanami_crypto/src -lhanami_crypto
LIBS += -L../../libraries/hanami_crypto/src/debug -lhanami_crypto
LIBS += -L../../libraries/hanami_crypto/src/release -lhanami_crypto
INCLUDEPATH += ../../libraries/hanami_crypto/include

LIBS += -L../../libraries/hanami_common/src -lhanami_common
LIBS += -L../../libraries/hanami_common/src/debug -lhanami_common
LIBS += -L../../libraries/hanami_common/src/release -lhanami_common
INCLUDEPATH += ../../libraries/hanami_common/include

LIBS += -lcryptopp -lssl -luuid -lcrypto -pthread -lprotobuf

INCLUDEPATH += $$PWD \
               src

SOURCES += src/main.cpp \
    src/common/test_step.cpp \
    src/common/test_thread.cpp \
    src/rest_api_tests/kyouko/cluster/cluster_create_test.cpp \
    src/rest_api_tests/kyouko/cluster/cluster_delete_test.cpp \
    src/rest_api_tests/kyouko/cluster/cluster_get_test.cpp \
    src/rest_api_tests/kyouko/cluster/cluster_list_test.cpp \
    src/rest_api_tests/kyouko/cluster/cluster_load_test.cpp \
    src/rest_api_tests/kyouko/cluster/cluster_save_test.cpp \
    src/rest_api_tests/kyouko/cluster/cluster_switch_to_direct_test.cpp \
    src/rest_api_tests/kyouko/cluster/cluster_switch_to_task_test.cpp \
    src/rest_api_tests/kyouko/io/direct_io_test.cpp \
    src/rest_api_tests/kyouko/task/image_request_task_test.cpp \
    src/rest_api_tests/kyouko/task/image_train_task_test.cpp \
    src/rest_api_tests/kyouko/task/table_request_task_test.cpp \
    src/rest_api_tests/kyouko/task/table_train_task_test.cpp \
    src/rest_api_tests/misaki/project/project_create_test.cpp \
    src/rest_api_tests/misaki/project/project_delete_test.cpp \
    src/rest_api_tests/misaki/project/project_get_test.cpp \
    src/rest_api_tests/misaki/project/project_list_test.cpp \
    src/rest_api_tests/misaki/user/user_create_test.cpp \
    src/rest_api_tests/misaki/user/user_delete_test.cpp \
    src/rest_api_tests/misaki/user/user_get_test.cpp \
    src/rest_api_tests/misaki/user/user_list_test.cpp \
    src/rest_api_tests/rest_api_tests.cpp \
    src/rest_api_tests/shiori/datasets/dataset_check_test.cpp \
    src/rest_api_tests/shiori/datasets/dataset_create_csv_test.cpp \
    src/rest_api_tests/shiori/datasets/dataset_create_mnist_test.cpp \
    src/rest_api_tests/shiori/datasets/dataset_delete_test.cpp \
    src/rest_api_tests/shiori/datasets/dataset_get_test.cpp \
    src/rest_api_tests/shiori/datasets/dataset_list_test.cpp \
    src/rest_api_tests/shiori/request_results/request_result_delete_test.cpp \
    src/rest_api_tests/shiori/request_results/request_result_get_test.cpp \
    src/rest_api_tests/shiori/request_results/request_result_list_test.cpp \
    src/rest_api_tests/shiori/checkpoints/checkpoint_delete_test.cpp \
    src/rest_api_tests/shiori/checkpoints/checkpoint_get_test.cpp \
    src/rest_api_tests/shiori/checkpoints/checkpoint_list_test.cpp

HEADERS += \
    src/args.h \
    src/common/test_step.h \
    src/common/test_thread.h \
    src/config.h \
    src/rest_api_tests/kyouko/cluster/cluster_create_test.h \
    src/rest_api_tests/kyouko/cluster/cluster_delete_test.h \
    src/rest_api_tests/kyouko/cluster/cluster_get_test.h \
    src/rest_api_tests/kyouko/cluster/cluster_list_test.h \
    src/rest_api_tests/kyouko/cluster/cluster_load_test.h \
    src/rest_api_tests/kyouko/cluster/cluster_save_test.h \
    src/rest_api_tests/kyouko/cluster/cluster_switch_to_direct_test.h \
    src/rest_api_tests/kyouko/cluster/cluster_switch_to_task_test.h \
    src/rest_api_tests/kyouko/io/direct_io_test.h \
    src/rest_api_tests/kyouko/task/image_request_task_test.h \
    src/rest_api_tests/kyouko/task/image_train_task_test.h \
    src/rest_api_tests/kyouko/task/table_request_task_test.h \
    src/rest_api_tests/kyouko/task/table_train_task_test.h \
    src/rest_api_tests/misaki/project/project_create_test.h \
    src/rest_api_tests/misaki/project/project_delete_test.h \
    src/rest_api_tests/misaki/project/project_get_test.h \
    src/rest_api_tests/misaki/project/project_list_test.h \
    src/rest_api_tests/misaki/user/user_create_test.h \
    src/rest_api_tests/misaki/user/user_delete_test.h \
    src/rest_api_tests/misaki/user/user_get_test.h \
    src/rest_api_tests/misaki/user/user_list_test.h \
    src/rest_api_tests/rest_api_tests.h \
    src/rest_api_tests/shiori/datasets/dataset_check_test.h \
    src/rest_api_tests/shiori/datasets/dataset_create_csv_test.h \
    src/rest_api_tests/shiori/datasets/dataset_create_mnist_test.h \
    src/rest_api_tests/shiori/datasets/dataset_delete_test.h \
    src/rest_api_tests/shiori/datasets/dataset_get_test.h \
    src/rest_api_tests/shiori/datasets/dataset_list_test.h \
    src/rest_api_tests/shiori/request_results/request_result_delete_test.h \
    src/rest_api_tests/shiori/request_results/request_result_get_test.h \
    src/rest_api_tests/shiori/request_results/request_result_list_test.h \
    src/rest_api_tests/shiori/checkpoints/checkpoint_delete_test.h \
    src/rest_api_tests/shiori/checkpoints/checkpoint_get_test.h \
    src/rest_api_tests/shiori/checkpoints/checkpoint_list_test.h
