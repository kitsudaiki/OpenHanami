#ifndef BITBUFFER_TEST_H
#define BITBUFFER_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class BitBuffer_Test : public Hanami::CompareTestHelper
{
   public:
    BitBuffer_Test();

   private:
    void set_get_test();
    void complete_test();
};

}  // namespace Hanami

#endif  // BITBUFFER_TEST_H
