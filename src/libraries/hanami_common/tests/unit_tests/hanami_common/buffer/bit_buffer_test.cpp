#include "bit_buffer_test.h"

#include <hanami_common/buffer/bit_buffer.h>

namespace Hanami
{

BitBuffer_Test::BitBuffer_Test() : Hanami::CompareTestHelper("BitBuffer_Test")
{
    set_get_test();
    complete_test();
}

void
BitBuffer_Test::set_get_test()
{
    BitBuffer<10> buffer;
    buffer.set(1, true);

    TEST_EQUAL(buffer.get(100), false);
    TEST_EQUAL(buffer.get(0), false);
    TEST_EQUAL(buffer.get(1), true);
    buffer.set(1, false);
    TEST_EQUAL(buffer.get(1), false);
}

void
BitBuffer_Test::complete_test()
{
    BitBuffer<10> buffer;

    TEST_EQUAL(buffer.isComplete(), false);
    buffer.set(0, true);
    TEST_EQUAL(buffer.isComplete(), false);
    buffer.set(1, true);
    buffer.set(2, true);
    buffer.set(3, true);
    buffer.set(4, true);
    buffer.set(5, true);
    buffer.set(6, true);
    buffer.set(7, true);
    buffer.set(8, true);
    buffer.set(9, true);
    TEST_EQUAL(buffer.isComplete(), true);
}

}  // namespace Hanami
