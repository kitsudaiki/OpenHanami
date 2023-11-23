QT       -= qt core gui

TARGET = hanami_common
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.26.1

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

INCLUDEPATH += ../../../third-party-libs/json/include

INCLUDEPATH += $$PWD \
            ../include

SOURCES += \
    buffer/item_buffer.cpp \
    buffer/stack_buffer_reserve.cpp \
    files/binary_file.cpp \
    files/binary_file_direct.cpp \
    files/text_file.cpp \
    items/table_item.cpp \
    logger.cpp \
    memory_counter.cpp \
    methods/file_methods.cpp \
    methods/string_methods.cpp \
    methods/vector_methods.cpp \
    process_execution.cpp \
    progress_bar.cpp \
    statemachine.cpp \
    test_helper/compare_test_helper.cpp \
    test_helper/memory_leak_test_helper.cpp \
    test_helper/speed_test_helper.cpp \
    threading/barrier.cpp \
    threading/cleanup_thread.cpp \
    threading/event.cpp \
    threading/event_queue.cpp \
    threading/thread.cpp \
    threading/thread_handler.cpp

HEADERS += \
    ../include/hanami_common/buffer/bit_buffer.h \
    ../include/hanami_common/buffer/data_buffer.h \
    ../include/hanami_common/buffer/item_buffer.h \
    ../include/hanami_common/buffer/ring_buffer.h \
    ../include/hanami_common/buffer/stack_buffer.h \
    ../include/hanami_common/buffer/stack_buffer_reserve.h \
    ../include/hanami_common/files/binary_file.h \
    ../include/hanami_common/files/binary_file_direct.h \
    ../include/hanami_common/files/text_file.h \
    ../include/hanami_common/items/table_item.h \
    ../include/hanami_common/logger.h \
    ../include/hanami_common/memory_counter.h \
    ../include/hanami_common/methods/file_methods.h \
    ../include/hanami_common/methods/string_methods.h \
    ../include/hanami_common/methods/vector_methods.h \
    ../include/hanami_common/process_execution.h \
    ../include/hanami_common/progress_bar.h \
    ../include/hanami_common/statemachine.h \
    ../include/hanami_common/structs.h \
    ../include/hanami_common/test_helper/compare_test_helper.h \
    ../include/hanami_common/test_helper/memory_leak_test_helper.h \
    ../include/hanami_common/test_helper/speed_test_helper.h \
    ../include/hanami_common/threading/barrier.h \
    ../include/hanami_common/threading/cleanup_thread.h \
    ../include/hanami_common/threading/event.h \
    ../include/hanami_common/threading/event_queue.h \
    ../include/hanami_common/threading/thread.h \
    ../include/hanami_common/threading/thread_handler.h \
    state.h
