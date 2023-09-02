QT       -= qt core gui

TARGET = hanami_common
TEMPLATE = lib
CONFIG += c++17
VERSION = 0.26.1

INCLUDEPATH += $$PWD \
            ../include

SOURCES += \
    files/binary_file_direct.cpp \
    methods/file_methods.cpp \
    files/binary_file.cpp \
    files/text_file.cpp \
    logger.cpp \
    progress_bar.cpp \
    threading/thread.cpp \
    statemachine.cpp \
    items/data_items.cpp \
    items/table_item.cpp \
    threading/barrier.cpp \
    process_execution.cpp \
    test_helper/compare_test_helper.cpp \
    test_helper/speed_test_helper.cpp \
    buffer/stack_buffer_reserve.cpp \
    methods/string_methods.cpp \
    methods/vector_methods.cpp \
    memory_counter.cpp \
    threading/event.cpp \
    threading/thread_handler.cpp \
    test_helper/memory_leak_test_helper.cpp \
    threading/cleanup_thread.cpp \
    buffer/item_buffer.cpp \
    threading/event_queue.cpp

HEADERS += \
    ../include/hanami_common/files/binary_file_direct.h \
    ../include/hanami_common/methods/file_methods.h \
    ../include/hanami_common/files/binary_file.h \
    ../include/hanami_common/files/text_file.h \
    ../include/hanami_common/logger.h \
    ../include/hanami_common/progress_bar.h \
    state.h \
    ../include/hanami_common/buffer/ring_buffer.h \
    ../include/hanami_common/methods/string_methods.h \
    ../include/hanami_common/methods/vector_methods.h \
    ../include/hanami_common/items/data_items.h \
    ../include/hanami_common/statemachine.h \
    ../include/hanami_common/threading/thread.h \
    ../include/hanami_common/items/table_item.h \
    ../include/hanami_common/threading/barrier.h \
    ../include/hanami_common/process_execution.h \
    ../include/hanami_common/test_helper/compare_test_helper.h \
    ../include/hanami_common/test_helper/speed_test_helper.h \
    ../include/hanami_common/buffer/data_buffer.h \
    ../include/hanami_common/buffer/stack_buffer.h \
    ../include/hanami_common/buffer/stack_buffer_reserve.h \
    ../include/hanami_common/memory_counter.h \
    ../include/hanami_common/threading/event.h \
    ../include/hanami_common/threading/thread_handler.h \
    ../include/hanami_common/test_helper/memory_leak_test_helper.h \
    ../include/hanami_common/threading/cleanup_thread.h \
    ../include/hanami_common/buffer/item_buffer.h \
    ../include/hanami_common/threading/event_queue.h \
    ../include/hanami_common/structs.h
