/**
 *  @file    jsonItem_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef JSONITEM_TEST_H
#define JSONITEM_TEST_H

#include <assert.h>

#include <hanami_common/test_helper/memory_leak_test_helper.h>
#include <hanami_json/json_item.h>

namespace Kitsunemimi
{
class JsonItem_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    JsonItem_Test();

private:
    void constructor_test();
    void assigmentOperator_test();
    void insert_test();
    void append_test();
    void replaceItem_test();
    void deleteContent_test();

    void getItemContent_test();
    void get_test();

    JsonItem getTestItem();

};

}

#endif // JSONITEM_TEST_H
