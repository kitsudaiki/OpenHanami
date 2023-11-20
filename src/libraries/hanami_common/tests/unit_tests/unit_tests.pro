include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

QMAKE_CXX = clang++-15
QMAKE_LINK = clang++-15

LIBS += -L../../src -lhanami_common
INCLUDEPATH += $$PWD
INCLUDEPATH += ../../../../third-party-libs/json/include

HEADERS += \
    hanami_common/buffer/bit_buffer_test.h \
    hanami_common/buffer/data_buffer_test.h \
    hanami_common/buffer/item_buffer_test.h \
    hanami_common/buffer/ring_buffer_test.h \
    hanami_common/buffer/stack_buffer_reserve_test.h \
    hanami_common/buffer/stack_buffer_test.h \
    hanami_common/files/binary_file_with_directIO_test.h \
    hanami_common/files/binary_file_without_directIO_test.h \
    hanami_common/files/text_file_test.h \
    hanami_common/items/table_item_test.h \
    hanami_common/logger_test.h \
    hanami_common/methods/file_methods_test.h \
    hanami_common/methods/string_methods_test.h \
    hanami_common/methods/vector_methods_test.h \
    hanami_common/progress_bar_test.h \
    hanami_common/state_test.h \
    hanami_common/statemachine_test.h \
    hanami_common/threading/thread_handler_test.h

SOURCES += \
    hanami_common/buffer/bit_buffer_test.cpp \
    hanami_common/buffer/data_buffer_test.cpp \
    hanami_common/buffer/item_buffer_test.cpp \
    hanami_common/buffer/ring_buffer_test.cpp \
    hanami_common/buffer/stack_buffer_reserve_test.cpp \
    hanami_common/buffer/stack_buffer_test.cpp \
    hanami_common/files/binary_file_with_directIO_test.cpp \
    hanami_common/files/binary_file_without_directIO_test.cpp \
    hanami_common/files/text_file_test.cpp \
    hanami_common/items/table_item_test.cpp \
    hanami_common/logger_test.cpp \
    hanami_common/methods/file_methods_test.cpp \
    hanami_common/methods/string_methods_test.cpp \
    hanami_common/methods/vector_methods_test.cpp \
    hanami_common/progress_bar_test.cpp \
    hanami_common/state_test.cpp \
    hanami_common/statemachine_test.cpp \
    hanami_common/threading/thread_handler_test.cpp \
    main.cpp

