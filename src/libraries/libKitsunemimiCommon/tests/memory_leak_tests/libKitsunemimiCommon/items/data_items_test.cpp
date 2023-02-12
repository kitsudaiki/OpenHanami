/**
 *  @file    data_items_DataArray_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "data_items_test.h"
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi
{

DataItems_Test::DataItems_Test()
    : Kitsunemimi::MemoryLeakTestHelpter("DataItems_Test")
{
    map_create_delete_test();
    map_insert_remove_test();

    array_create_delete_test();
    array_append_remove_test();
}

/**
 * @brief create_delete_test
 */
void
DataItems_Test::map_create_delete_test()
{
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);

    REINIT_TEST();

    // normal test
    DataMap* object = new DataMap();

    object->insert("poi", defaultValue.copy());
    object->insert("asdf", stringValue.copy());
    object->insert("hmm", intValue.copy());
    object->insert("xyz", floatValue.copy());
    object->insert("fail", nullptr);

    // copy-assignment
    DataMap* object2 = new DataMap();
    *object2 = *object;

    // copy-constructor
    DataMap* object3 = new DataMap(*object);

    delete object;
    delete object2;
    delete object3;

    CHECK_MEMORY();
}

/**
 * @brief map_insert_remove_test
 */
void
DataItems_Test::map_insert_remove_test()
{
    DataValue defaultValue;

    DataMap object;

    REINIT_TEST();

    object.insert("poi", defaultValue.copy());
    object.remove("poi");

    CHECK_MEMORY();
}

/**
 * @brief DataItems_Test::array_create_delete_test
 */
void
DataItems_Test::array_create_delete_test()
{
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);

    REINIT_TEST();

    // normal test
    DataArray* array = new DataArray();

    array->append(defaultValue.copy());
    array->append(stringValue.copy());
    array->append(intValue.copy());
    array->append(floatValue.copy());
    array->append(nullptr);

    // copy-assignment
    DataArray* array2 = new DataArray();
    *array2 = *array;

    // copy-constructor
    DataArray* array3 = new DataArray(*array);

    delete array;
    delete array2;
    delete array3;

    CHECK_MEMORY();
}

/**
 * @brief array_append_remove_test
 */
void
DataItems_Test::array_append_remove_test()
{
    DataValue defaultValue;

    DataArray array;

    array.append(defaultValue.copy());
    array.remove(0);

    REINIT_TEST();

    array.append(defaultValue.copy());
    array.remove(0);

    CHECK_MEMORY();
}

}
