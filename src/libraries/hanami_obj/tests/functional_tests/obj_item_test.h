/**
 *  @file       obj_parser_test.h
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  MIT License
 */

#ifndef OBJPARSER_TEST_H
#define OBJPARSER_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

#include <string>
#include <utility>
#include <vector>

class ObjItem_Test : public Hanami::CompareTestHelper
{
   public:
    ObjItem_Test();

   private:
    void parse_test();
    void converter_test();

    const std::string getTestString();
    const std::string getCompareString();
};

#endif  // OBJPARSER_TEST_H
