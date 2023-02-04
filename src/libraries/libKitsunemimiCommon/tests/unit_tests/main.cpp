/**
 *  @file    main.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <iostream>

#include <libKitsunemimiCommon/buffer/data_buffer_test.h>
#include <libKitsunemimiCommon/buffer/item_buffer_test.h>
#include <libKitsunemimiCommon/buffer/ring_buffer_test.h>
#include <libKitsunemimiCommon/buffer/stack_buffer_reserve_test.h>
#include <libKitsunemimiCommon/buffer/stack_buffer_test.h>

#include <libKitsunemimiCommon/methods/string_methods_test.h>
#include <libKitsunemimiCommon/methods/vector_methods_test.h>
#include <libKitsunemimiCommon/methods/file_methods_test.h>

#include <libKitsunemimiCommon/state_test.h>
#include <libKitsunemimiCommon/statemachine_test.h>
#include <libKitsunemimiCommon/progress_bar_test.h>
#include <libKitsunemimiCommon/logger_test.h>

#include <libKitsunemimiCommon/threading/thread_handler_test.h>

#include <libKitsunemimiCommon/items/data_items_DataArray_test.h>
#include <libKitsunemimiCommon/items/data_items_DataMap_test.h>
#include <libKitsunemimiCommon/items/data_items_DataValue_test.h>
#include <libKitsunemimiCommon/items/table_item_test.h>

#include <libKitsunemimiCommon/files/text_file_test.h>
#include <libKitsunemimiCommon/files/binary_file_with_directIO_test.h>
#include <libKitsunemimiCommon/files/binary_file_without_directIO_test.h>

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
