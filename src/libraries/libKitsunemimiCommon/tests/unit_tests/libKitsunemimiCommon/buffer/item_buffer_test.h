/**
 *  @file    item_buffer_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef ITEM_BUFFER_TEST_H
#define ITEM_BUFFER_TEST_H

#include <libKitsunemimiCommon/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{

class ItemBuffer_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    ItemBuffer_Test();

private:
    void initBuffer_test();
    void deleteItem_test();
    void deleteAll_test();
    void addNewItem_test();
    void backup_restore_test();
};

}

#endif // ITEM_BUFFER_TEST_H
