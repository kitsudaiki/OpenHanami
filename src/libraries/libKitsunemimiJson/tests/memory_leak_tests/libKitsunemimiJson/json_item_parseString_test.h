/**
 *  @file    jsonItems_parseString_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef JSON_ITEM_PARSESTRING_TEST_H
#define JSON_ITEM_PARSESTRING_TEST_H

#include <libKitsunemimiCommon/test_helper/memory_leak_test_helper.h>

namespace Kitsunemimi
{
class JsonItem_ParseString_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    JsonItem_ParseString_Test();

private:
    void parseString_test();
};

}

#endif // JSON_ITEM_PARSESTRING_TEST_H
