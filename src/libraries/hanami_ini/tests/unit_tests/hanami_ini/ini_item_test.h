/**
 *  @file    ini_item_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef PARSERTEST_H
#define PARSERTEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

#include <algorithm>

namespace Hanami
{
class IniItem_Test : public Hanami::CompareTestHelper
{
   public:
    IniItem_Test();

   private:
    void parse_test();
    void get_test();
    void set_test();
    void removeGroup_test();
    void removeEntry_test();
    void print_test();

    const std::string getTestString();
};

}  // namespace Hanami

#endif  // PARSERTEST_H
