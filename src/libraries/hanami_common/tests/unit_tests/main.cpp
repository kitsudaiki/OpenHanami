/**
 *  @file    main.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <iostream>

#include <hanami_common/buffer/data_buffer_test.h>
#include <hanami_common/buffer/item_buffer_test.h>
#include <hanami_common/buffer/ring_buffer_test.h>
#include <hanami_common/buffer/stack_buffer_reserve_test.h>
#include <hanami_common/buffer/stack_buffer_test.h>

#include <hanami_common/methods/string_methods_test.h>
#include <hanami_common/methods/vector_methods_test.h>
#include <hanami_common/methods/file_methods_test.h>

#include <hanami_common/state_test.h>
#include <hanami_common/statemachine_test.h>
#include <hanami_common/progress_bar_test.h>
#include <hanami_common/logger_test.h>

#include <hanami_common/threading/thread_handler_test.h>

#include <hanami_common/items/data_items_DataArray_test.h>
#include <hanami_common/items/data_items_DataMap_test.h>
#include <hanami_common/items/data_items_DataValue_test.h>
#include <hanami_common/items/table_item_test.h>

#include <hanami_common/files/text_file_test.h>
#include <hanami_common/files/binary_file_with_directIO_test.h>
#include <hanami_common/files/binary_file_without_directIO_test.h>

int main()
{
    Kitsunemimi::DataBuffer_Test();
    Kitsunemimi::ItemBuffer_Test();
    Kitsunemimi::RingBuffer_Test();
    Kitsunemimi::StackBufferReserve_Test();
    Kitsunemimi::StackBuffer_Test();

    Kitsunemimi::StringMethods_Test();
    Kitsunemimi::VectorMethods_Test();
    Kitsunemimi::FileMethods_Test();

    Kitsunemimi::State_Test();
    Kitsunemimi::Statemachine_Test();
    Kitsunemimi::ProgressBar_Test();
    Kitsunemimi::Logger_Test();

    Kitsunemimi::ThreadHandler_Test();

    Kitsunemimi::DataItems_DataValue_Test();
    Kitsunemimi::DataItems_DataArray_Test();
    Kitsunemimi::DataItems_DataMap_Test();

    Kitsunemimi::TextFile_Test();
    Kitsunemimi::BinaryFile_withDirectIO_Test();
    Kitsunemimi::BinaryFile_withoutDirectIO_Test();

    Kitsunemimi::TableItem_test();
}
