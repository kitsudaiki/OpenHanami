/**
 *  @file    data_buffer_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "data_buffer_test.h"

#include <hanami_common/buffer/data_buffer.h>

namespace Kitsunemimi
{

struct TestStruct
{
    uint8_t a = 0;
    uint8_t b = 0;
    uint64_t c = 0;
} __attribute__((packed));

DataBuffer_Test::DataBuffer_Test()
    : Kitsunemimi::CompareTestHelper("DataBuffer_Test")
{
    structSize_test();
    constructor_test();
    copy_assingment_constructor_test();
    copy_assingment_operator_test();
    addObject_DataBuffer_test();
    getBlock_DataBuffer_test();
    reset_DataBuffer_test();

    addData_DataBuffer_test();
    allocateBlocks_DataBuffer_test();
}

/**
 * structSize_test
 */
void
DataBuffer_Test::structSize_test()
{
    DataBuffer testBuffer(10);
    TEST_EQUAL(sizeof(DataBuffer) % 8, 0);
}

/**
 * constructor_test
 */
void
DataBuffer_Test::constructor_test()
{
    DataBuffer testBuffer(10);

    // check metadata of the buffer
    bool isNullptr = testBuffer.data == nullptr;
    TEST_EQUAL(isNullptr, false);
    TEST_EQUAL(testBuffer.numberOfBlocks, 10);
    TEST_EQUAL(testBuffer.usedBufferSize, 0);
    TEST_EQUAL(testBuffer.totalBufferSize, 10*testBuffer.blockSize);
}

/**
 * copy_assingment_constructor_test
 */
void
DataBuffer_Test::copy_assingment_constructor_test()
{
    // init
    DataBuffer testBuffer(10);
    TestStruct testStruct;
    testStruct.b = 42;

    // write data to buffer
    TEST_EQUAL(addObject_DataBuffer(testBuffer, &testStruct), true);

    // check metadata of the buffer
    TEST_EQUAL(testBuffer.numberOfBlocks, 10);
    TEST_EQUAL(testBuffer.usedBufferSize, 10);
    TEST_EQUAL(testBuffer.totalBufferSize, 10*testBuffer.blockSize);

    // use copy contstructor
    DataBuffer bufferCopy(testBuffer);

    // check metadata of the new buffer
    TEST_EQUAL(bufferCopy.numberOfBlocks, 10);
    TEST_EQUAL(bufferCopy.usedBufferSize, 10);
    TEST_EQUAL(bufferCopy.totalBufferSize, 10*bufferCopy.blockSize);

    // check content of the new buffer
    uint8_t* dataByte = static_cast<uint8_t*>(bufferCopy.data);
    TEST_EQUAL(static_cast<int>(dataByte[1]), 42);
}

/**
 * copy_assingment_operator_test
 */
void
DataBuffer_Test::copy_assingment_operator_test()
{
    // init
    DataBuffer testBuffer(10);
    TestStruct testStruct;
    testStruct.b = 42;

    // write data to buffer
    TEST_EQUAL(addObject_DataBuffer(testBuffer, &testStruct), true);

    // check metadata of the buffer
    TEST_EQUAL(testBuffer.numberOfBlocks, 10);
    TEST_EQUAL(testBuffer.usedBufferSize, 10);
    TEST_EQUAL(testBuffer.totalBufferSize, 10*testBuffer.blockSize);

    // use copy assignment
    DataBuffer bufferCopy(1);
    bufferCopy = testBuffer;

    // check metadata of the new buffer
    TEST_EQUAL(bufferCopy.numberOfBlocks, 10);
    TEST_EQUAL(bufferCopy.usedBufferSize, 10);
    TEST_EQUAL(bufferCopy.totalBufferSize, 10*bufferCopy.blockSize);

    // check content of the new buffer
    uint8_t* dataByte = static_cast<uint8_t*>(bufferCopy.data);
    TEST_EQUAL(static_cast<int>(dataByte[1]), 42);
}

/**
 * addObject_DataBuffer_test
 */
void
DataBuffer_Test::addObject_DataBuffer_test()
{
    // init
    DataBuffer testBuffer(10);
    TestStruct testStruct;
    testStruct.b = 42;

    // write data to buffer
    TEST_EQUAL(addObject_DataBuffer(testBuffer, &testStruct), true);

    // check metadata of the buffer
    TEST_EQUAL(testBuffer.numberOfBlocks, 10);
    TEST_EQUAL(testBuffer.usedBufferSize, 10);
    TEST_EQUAL(testBuffer.totalBufferSize, 10*testBuffer.blockSize);

    // check content of the buffer
    uint8_t* dataByte = static_cast<uint8_t*>(testBuffer.data);
    TEST_EQUAL(static_cast<int>(dataByte[1]), 42);
}

/**
 * getBlock_DataBuffer_test
 */
void
DataBuffer_Test::getBlock_DataBuffer_test()
{
    // init
    DataBuffer testBuffer(10);
    TestStruct testStruct;
    testStruct.b = 42;

    uint8_t* dataByte = static_cast<uint8_t*>(testBuffer.data);
    memcpy(&dataByte[4096], &testStruct, sizeof(TestStruct));

    // check content of the buffer with getBlock-method
    TEST_EQUAL(static_cast<int>(getBlock_DataBuffer(testBuffer, 1)[1]), 42);
}

/**
 * reset_DataBuffer_test
 */
void
DataBuffer_Test::reset_DataBuffer_test()
{
    // init
    DataBuffer testBuffer(10);
    TestStruct testStruct;
    testStruct.b = 42;

    // write data to buffer
    TEST_EQUAL(addObject_DataBuffer(testBuffer, &testStruct), true);

    // reset buffer
    TEST_EQUAL(reset_DataBuffer(testBuffer, 1), true);

    // check metadata of the buffer
    TEST_EQUAL(testBuffer.numberOfBlocks, 1);
    TEST_EQUAL(testBuffer.usedBufferSize, 0);
    TEST_EQUAL(testBuffer.totalBufferSize, 1*testBuffer.blockSize);

    // check content of the buffer
    uint8_t* dataByte = static_cast<uint8_t*>(testBuffer.data);
    TEST_EQUAL(static_cast<int>(dataByte[1]), 0);
}

/**
 * addData_DataBuffer_test
 */
void
DataBuffer_Test::addData_DataBuffer_test()
{
    // init
    DataBuffer testBuffer(10);
    TestStruct testStruct;
    testStruct.b = 42;

    // check metadata of the buffer
    TEST_EQUAL(testBuffer.usedBufferSize, 0);

    // add data to buffer
    void* testStructPtr = static_cast<void*>(&testStruct);
    TEST_EQUAL(addData_DataBuffer(testBuffer, testStructPtr, sizeof(TestStruct)), true);

    // check metadata of the buffer
    TEST_EQUAL(testBuffer.numberOfBlocks, 10);
    TEST_EQUAL(testBuffer.usedBufferSize, 10);
    TEST_EQUAL(testBuffer.totalBufferSize, 10*testBuffer.blockSize);

    // check content of the buffer
    uint8_t* dataByte = static_cast<uint8_t*>(testBuffer.data);
    TEST_EQUAL(static_cast<int>(dataByte[1]), 42);
}

/**
 * allocateBlocks_DataBuffer_test
 */
void
DataBuffer_Test::allocateBlocks_DataBuffer_test()
{
    // init
    DataBuffer testBuffer(10);
    TestStruct testStruct;
    testStruct.b = 42;

    // write data to buffer
    TEST_EQUAL(addObject_DataBuffer(testBuffer, &testStruct), true);

    // check metadata of the buffer
    TEST_EQUAL(testBuffer.numberOfBlocks, 10);
    TEST_EQUAL(testBuffer.usedBufferSize, 10);
    TEST_EQUAL(testBuffer.totalBufferSize, 10*testBuffer.blockSize);

    // check content of the buffer
    uint8_t* dataByte = static_cast<uint8_t*>(testBuffer.data);
    TEST_EQUAL(static_cast<int>(dataByte[1]), 42);

    // resize
    TEST_EQUAL(allocateBlocks_DataBuffer(testBuffer, 1), true);

    // check metadata of the buffer
    TEST_EQUAL(testBuffer.numberOfBlocks, 11);
    TEST_EQUAL(testBuffer.usedBufferSize, 10);
    TEST_EQUAL(testBuffer.totalBufferSize, 11*testBuffer.blockSize);

    // check content of the buffer
    dataByte = static_cast<uint8_t*>(testBuffer.data);
    TEST_EQUAL(static_cast<int>(dataByte[1]), 42);
}

}
