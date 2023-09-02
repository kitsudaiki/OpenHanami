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

#include <hanami_common/state_test.h>
#include <hanami_common/statemachine_test.h>

#include <hanami_common/items/data_items_test.h>
#include <hanami_common/items/table_item_test.h>

#include <hanami_common/threading/thread_test.h>

int main()
{
    Kitsunemimi::DataBuffer_Test();
    Kitsunemimi::ItemBuffer_Test();
    Kitsunemimi::RingBuffer_Test();
    Kitsunemimi::StackBufferReserve_Test();
    Kitsunemimi::StackBuffer_Test();

    Kitsunemimi::State_Test();
    Kitsunemimi::Statemachine_Test();

    Kitsunemimi::DataItems_Test();
    Kitsunemimi::TableItem_test();

    Kitsunemimi::Thread_Test();
}
