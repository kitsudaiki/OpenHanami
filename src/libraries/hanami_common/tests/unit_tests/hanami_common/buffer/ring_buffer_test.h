/**
 *  @file    ring_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef RING_BUFFER_TEST_H
#define RING_BUFFER_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class RingBuffer_Test
        : public Hanami::CompareTestHelper
{
public:
    RingBuffer_Test();

private:
    void addData_RingBuffer_test();
    void addObject_RingBuffer_test();
    void getWritePosition_RingBuffer_test();
    void getSpaceToEnd_RingBuffer_test();
    void getDataPointer_RingBuffer_test();
    void moveForward_RingBuffer_test();
    void getObject_RingBuffer_test();
};

}

#endif // RING_BUFFER_TEST_H
