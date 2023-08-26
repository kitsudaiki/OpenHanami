QT -= qt core gui

TARGET = SDK_API_Testing
CONFIG += console c++17
CONFIG -= app_bundle

LIBS += -L../../sdk/cpp/libHanamiAiSdk/src -lHanamiAiSdk
LIBS += -L../../sdk/cpp/libHanamiAiSdk/src/debug -lHanamiAiSdk
LIBS += -L../../sdk/cpp/libHanamiAiSdk/src/release -lHanamiAiSdk
INCLUDEPATH += ../../sdk/cpp/libHanamiAiSdk/include

LIBS += -L../../libraries/libKitsunemimiConfig/src -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/debug -lKitsunemimiConfig
LIBS += -L../../libraries/libKitsunemimiConfig/src/release -lKitsunemimiConfig
INCLUDEPATH += ../../libraries/libKitsunemimiConfig/include

LIBS += -L../../libraries/libKitsunemimiJson/src -lKitsunemimiJson
LIBS += -L../../libraries/libKitsunemimiJson/src/debug -lKitsunemimiJson
LIBS += -L../../libraries/libKitsunemimiJson/src/release -lKitsunemimiJson
INCLUDEPATH += ../../libraries/libKitsunemimiJson/include

LIBS += -L../../libraries/libKitsunemimiIni/src -lKitsunemimiIni
LIBS += -L../../libraries/libKitsunemimiIni/src/debug -lKitsunemimiIni
LIBS += -L../../libraries/libKitsunemimiIni/src/release -lKitsunemimiIni
INCLUDEPATH += ../../libraries/libKitsunemimiIni/include

LIBS += -L../../libraries/libKitsunemimiArgs/src -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/debug -lKitsunemimiArgs
LIBS += -L../../libraries/libKitsunemimiArgs/src/release -lKitsunemimiArgs
INCLUDEPATH += ../../libraries/libKitsunemimiArgs/include

LIBS += -L../../libraries/libKitsunemimiCrypto/src -lKitsunemimiCrypto
LIBS += -L../../libraries/libKitsunemimiCrypto/src/debug -lKitsunemimiCrypto
LIBS += -L../../libraries/libKitsunemimiCrypto/src/release -lKitsunemimiCrypto
INCLUDEPATH += ../../libraries/libKitsunemimiCrypto/include

LIBS += -L../../libraries/libKitsunemimiCommon/src -lKitsunemimiCommon
LIBS += -L../../libraries/libKitsunemimiCommon/src/debug -lKitsunemimiCommon
LIBS += -L../../libraries/libKitsunemimiCommon/src/release -lKitsunemimiCommon
INCLUDEPATH += ../../libraries/libKitsunemimiCommon/include

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
