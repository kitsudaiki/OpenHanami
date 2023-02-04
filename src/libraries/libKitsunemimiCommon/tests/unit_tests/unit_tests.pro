include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiCommon
INCLUDEPATH += $$PWD

HEADERS += \
    libKitsunemimiCommon/methods/file_methods_test.h \
    libKitsunemimiCommon/files/binary_file_with_directIO_test.h \
    libKitsunemimiCommon/files/binary_file_without_directIO_test.h \
    libKitsunemimiCommon/files/text_file_test.h \
    libKitsunemimiCommon/logger_test.h \
    libKitsunemimiCommon/progress_bar_test.h \
    libKitsunemimiCommon/state_test.h \
    libKitsunemimiCommon/statemachine_test.h \
    libKitsunemimiCommon/methods/string_methods_test.h \
    libKitsunemimiCommon/methods/vector_methods_test.h \
    libKitsunemimiCommon/items/data_items_DataArray_test.h \
    libKitsunemimiCommon/items/data_items_DataValue_test.h \
    libKitsunemimiCommon/items/table_item_test.h \
    libKitsunemimiCommon/items/data_items_DataMap_test.h \
    libKitsunemimiCommon/buffer/data_buffer_test.h \
    libKitsunemimiCommon/buffer/ring_buffer_test.h \
    libKitsunemimiCommon/buffer/stack_buffer_reserve_test.h \
    libKitsunemimiCommon/buffer/stack_buffer_test.h \
    libKitsunemimiCommon/buffer/item_buffer_test.h \
    libKitsunemimiCommon/threading/thread_handler_test.h

SOURCES += \
    libKitsunemimiCommon/methods/file_methods_test.cpp \
    libKitsunemimiCommon/files/binary_file_with_directIO_test.cpp \
    libKitsunemimiCommon/files/binary_file_without_directIO_test.cpp \
    libKitsunemimiCommon/files/text_file_test.cpp \
    libKitsunemimiCommon/logger_test.cpp \
    libKitsunemimiCommon/progress_bar_test.cpp \
    libKitsunemimiCommon/threading/thread_handler_test.cpp \
    main.cpp \
    libKitsunemimiCommon/state_test.cpp \
    libKitsunemimiCommon/statemachine_test.cpp \
    libKitsunemimiCommon/methods/string_methods_test.cpp \
    libKitsunemimiCommon/methods/vector_methods_test.cpp \
    libKitsunemimiCommon/items/data_items_DataArray_test.cpp \
    libKitsunemimiCommon/items/data_items_DataValue_test.cpp \
    libKitsunemimiCommon/items/table_item_test.cpp \
    libKitsunemimiCommon/items/data_items_DataMap_test.cpp \
    libKitsunemimiCommon/buffer/data_buffer_test.cpp \
    libKitsunemimiCommon/buffer/ring_buffer_test.cpp \
    libKitsunemimiCommon/buffer/stack_buffer_reserve_test.cpp \
    libKitsunemimiCommon/buffer/stack_buffer_test.cpp \
    libKitsunemimiCommon/buffer/item_buffer_test.cpp

