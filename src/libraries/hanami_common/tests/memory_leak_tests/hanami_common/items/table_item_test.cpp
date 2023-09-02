/**
 *  @file    table_item_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "table_item_test.h"
#include <hanami_common/items/table_item.h>

namespace Kitsunemimi
{

TableItem_test::TableItem_test()
    : Kitsunemimi::MemoryLeakTestHelpter("TableItem_test")
{
    create_delete_test();
    add_delete_col_test();
    add_delete_row_test();
}

/**
 * @brief create_delete_test
 */
void
TableItem_test::create_delete_test()
{
    REINIT_TEST();

    TableItem* testItem = new TableItem();

    testItem->addColumn("asdf", "ASDF");
    testItem->addColumn("poipoipoi");
    testItem->addRow(std::vector<std::string>{"this is a test", "k"});
    testItem->addRow(std::vector<std::string>{"asdf", "qwert"});

    delete testItem;

    CHECK_MEMORY();
}

/**
 * @brief add_delete_col_test
 */
void
TableItem_test::add_delete_col_test()
{
    TableItem testItem;

    testItem.addColumn("asdf", "ASDF");
    testItem.deleteColumn("asdf");

    REINIT_TEST();

    testItem.addColumn("xyz", "ASDF");
    testItem.deleteColumn("xyz");

    CHECK_MEMORY();
}

/**
 * @brief add_delete_row_test
 */
void
TableItem_test::add_delete_row_test()
{
    TableItem testItem;

    testItem.addColumn("asdf", "ASDF");
    testItem.addColumn("poipoipoi");

    testItem.addRow(std::vector<std::string>{"this is a test", "k"});
    testItem.deleteRow(0);

    REINIT_TEST();

    testItem.addRow(std::vector<std::string>{"this is a test", "k"});
    testItem.deleteCell(0, 0);
    testItem.deleteRow(0);

    CHECK_MEMORY();
}

}
