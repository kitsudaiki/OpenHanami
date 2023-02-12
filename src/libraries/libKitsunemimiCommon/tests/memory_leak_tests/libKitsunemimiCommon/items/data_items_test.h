/**
 *  @file    data_items_DataArray_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef DATAITEMS_TEST_H
#define DATAITEMS_TEST_H

#include <libKitsunemimiCommon/test_helper/memory_leak_test_helper.h>

namespace Kitsunemimi
{
class DataArray;

class DataItems_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    DataItems_Test();

private:
    void map_create_delete_test();
    void map_insert_remove_test();

    void array_create_delete_test();
    void array_append_remove_test();
};

}

#endif // DATAITEMS_DATAARRAY_TEST_H
