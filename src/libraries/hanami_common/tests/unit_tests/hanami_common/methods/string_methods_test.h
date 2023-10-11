/**
 *  @file    string_methods_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef STRING_METHODS_TEST_H
#define STRING_METHODS_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{

class StringMethods_Test : public Hanami::CompareTestHelper
{
   public:
    StringMethods_Test();

   private:
    void splitStringByDelimiter_test();
    void splitStringByLength_test();
    void replaceSubstring_test();
    void removeWhitespaces_test();

    void ltrim_test();
    void rtrim_test();
    void trim_test();

    void toUpperCase_test();
    void toLowerCase_test();
};

}  // namespace Hanami

#endif  // STRINGMETHODS_TEST_H
