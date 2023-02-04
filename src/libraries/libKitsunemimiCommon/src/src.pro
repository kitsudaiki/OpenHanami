QT       -= qt core gui

TARGET = KitsunemimiCommon
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
    ../include/libKitsunemimiCommon/files/binary_file_direct.h \
    ../include/libKitsunemimiCommon/methods/file_methods.h \
    ../include/libKitsunemimiCommon/files/binary_file.h \
    ../include/libKitsunemimiCommon/files/text_file.h \
    ../include/libKitsunemimiCommon/logger.h \
    ../include/libKitsunemimiCommon/progress_bar.h \
    state.h \
    ../include/libKitsunemimiCommon/buffer/ring_buffer.h \
    ../include/libKitsunemimiCommon/methods/string_methods.h \
    ../include/libKitsunemimiCommon/methods/vector_methods.h \
    ../include/libKitsunemimiCommon/items/data_items.h \
    ../include/libKitsunemimiCommon/statemachine.h \
    ../include/libKitsunemimiCommon/threading/thread.h \
    ../include/libKitsunemimiCommon/items/table_item.h \
    ../include/libKitsunemimiCommon/threading/barrier.h \
    ../include/libKitsunemimiCommon/process_execution.h \
    ../include/libKitsunemimiCommon/test_helper/compare_test_helper.h \
    ../include/libKitsunemimiCommon/test_helper/speed_test_helper.h \
    ../include/libKitsunemimiCommon/buffer/data_buffer.h \
    ../include/libKitsunemimiCommon/buffer/stack_buffer.h \
    ../include/libKitsunemimiCommon/buffer/stack_buffer_reserve.h \
    ../include/libKitsunemimiCommon/memory_counter.h \
    ../include/libKitsunemimiCommon/threading/event.h \
    ../include/libKitsunemimiCommon/threading/thread_handler.h \
    ../include/libKitsunemimiCommon/test_helper/memory_leak_test_helper.h \
    ../include/libKitsunemimiCommon/threading/cleanup_thread.h \
    ../include/libKitsunemimiCommon/buffer/item_buffer.h \
    ../include/libKitsunemimiCommon/threading/event_queue.h
