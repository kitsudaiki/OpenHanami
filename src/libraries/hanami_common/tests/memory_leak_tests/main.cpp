/**
 *  @file    main.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <hanami_common/buffer/data_buffer_test.h>
#include <hanami_common/buffer/item_buffer_test.h>
#include <hanami_common/buffer/ring_buffer_test.h>
#include <hanami_common/buffer/stack_buffer_reserve_test.h>
#include <hanami_common/buffer/stack_buffer_test.h>
#include <hanami_common/items/table_item_test.h>
#include <hanami_common/state_test.h>
#include <hanami_common/statemachine_test.h>
#include <hanami_common/threading/thread_test.h>

#include <iostream>

int
main()
{
    Hanami::DataBuffer_Test();
    Hanami::ItemBuffer_Test();
    Hanami::RingBuffer_Test();
    Hanami::StackBufferReserve_Test();
    Hanami::StackBuffer_Test();

    Hanami::State_Test();
    Hanami::Statemachine_Test();

    Hanami::TableItem_test();

    Hanami::Thread_Test();
}
