/**
 *  @file    data_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef DATA_BUFFER_TEST_H
#define DATA_BUFFER_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{

class DataBuffer_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    DataBuffer_Test();

private:
    void constructor_test();
    void structSize_test();
    void copy_assingment_constructor_test();
    void copy_assingment_operator_test();
    void addObject_DataBuffer_test();
    void getBlock_DataBuffer_test();
    void reset_DataBuffer_test();

    void addData_DataBuffer_test();
    void allocateBlocks_DataBuffer_test();
};

}

#endif // DATABUFFER_TEST_H
