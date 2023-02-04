include(../../defaults.pri)

QT -= qt core gui

CONFIG   -= app_bundle
CONFIG += c++17 console

LIBS += -L../../src -lKitsunemimiCommon
INCLUDEPATH += $$PWD

HEADERS += \
    libKitsunemimiCommon/state_test.h \
    libKitsunemimiCommon/statemachine_test.h \
    libKitsunemimiCommon/items/table_item_test.h \
    libKitsunemimiCommon/buffer/data_buffer_test.h \
    libKitsunemimiCommon/buffer/ring_buffer_test.h \
    libKitsunemimiCommon/buffer/stack_buffer_reserve_test.h \
    libKitsunemimiCommon/buffer/stack_buffer_test.h \
    libKitsunemimiCommon/items/data_items_test.h \
    libKitsunemimiCommon/threading/thread_test.h \
    libKitsunemimiCommon/threading/bogus_event.h \
    libKitsunemimiCommon/threading/bogus_thread.h \
    libKitsunemimiCommon/buffer/item_buffer_test.h

SOURCES += \
    main.cpp \
    libKitsunemimiCommon/state_test.cpp \
    libKitsunemimiCommon/statemachine_test.cpp \
    libKitsunemimiCommon/items/table_item_test.cpp \
    libKitsunemimiCommon/buffer/data_buffer_test.cpp \
    libKitsunemimiCommon/buffer/ring_buffer_test.cpp \
    libKitsunemimiCommon/buffer/stack_buffer_reserve_test.cpp \
    libKitsunemimiCommon/buffer/stack_buffer_test.cpp \
    libKitsunemimiCommon/items/data_items_test.cpp \
    libKitsunemimiCommon/threading/thread_test.cpp \
    libKitsunemimiCommon/threading/bogus_event.cpp \
    libKitsunemimiCommon/threading/bogus_thread.cpp \
    libKitsunemimiCommon/buffer/item_buffer_test.cpp
