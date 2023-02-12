/**
 *  @file    data_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef DATA_BUFFER_TEST_H
#define DATA_BUFFER_TEST_H

#include <libKitsunemimiCommon/test_helper/memory_leak_test_helper.h>

namespace Kitsunemimi
{

class DataBuffer_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    DataBuffer_Test();

private:
    void create_delete_test();
    void fill_reset_test();
};

}

#endif // DATABUFFER_TEST_H
