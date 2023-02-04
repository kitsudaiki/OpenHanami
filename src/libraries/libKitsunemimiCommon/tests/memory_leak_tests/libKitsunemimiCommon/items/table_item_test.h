/**
 *  @file    table_item_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef TABLE_ITEM_TEST_H
#define TABLE_ITEM_TEST_H

#include <libKitsunemimiCommon/test_helper/memory_leak_test_helper.h>
#include <libKitsunemimiCommon/items/table_item.h>

namespace Kitsunemimi
{

class TableItem_test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    TableItem_test();

private:
    void create_delete_test();
    void add_delete_col_test();
    void add_delete_row_test();
};

}  // namespace Kitsunemimi

#endif // TABLE_ITEM_TEST_H
