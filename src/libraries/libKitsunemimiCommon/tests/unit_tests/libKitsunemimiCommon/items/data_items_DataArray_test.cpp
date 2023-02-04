/**
 *  @file    data_items_DataArray_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "data_items_DataArray_test.h"
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi
{

DataItems_DataArray_Test::DataItems_DataArray_Test()
    : Kitsunemimi::CompareTestHelper("DataItems_DataArray_Test")
{
    copy_assingment_constructor_test();
    copy_assingment_operator_test();
    operator_test();
    get_test();
    getSize_test();
    remove_test();
    clear_test();
    copy_test();
    toString_test();
    getType_test();
    isValue_isMap_isArray_test();
    toValue_toMap_toArray_test();
    getString_getInt_getFloat_test();

    // array-exclusive
    append_test();
}

/**
 * copy_assingment_constructor_test
 */
void
DataItems_DataArray_Test::copy_assingment_constructor_test()
{
    DataArray array = initTestArray();
    DataArray arrayCopy = array;
    TEST_EQUAL(arrayCopy.toString(), array.toString());
}

/**
 * copy_assingment_operator_test
 */
void
DataItems_DataArray_Test::copy_assingment_operator_test()
{
    DataArray array = initTestArray();
    DataArray arrayCopy;
    arrayCopy = array;
    TEST_EQUAL(arrayCopy.toString(), array.toString());
}

/**
 * append_test
 */
void
DataItems_DataArray_Test::append_test()
{
    DataArray array;
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);

    array.append(defaultValue.copy());
    array.append(stringValue.copy());
    array.append(intValue.copy());
    array.append(floatValue.copy());
    array.append(nullptr);

    TEST_EQUAL(array.size(), 5);
}

/**
 * operator_test
 */
void
DataItems_DataArray_Test::operator_test()
{
    DataArray array = initTestArray();

    TEST_EQUAL(array[1]->toString(), "test");

    // negative tests
    bool isNullptr = array[10] == nullptr;
    TEST_EQUAL(isNullptr, true);
    isNullptr = array["2"] == nullptr;
    TEST_EQUAL(isNullptr, true);
}

/**
 * get_test
 */
void
DataItems_DataArray_Test::get_test()
{
    DataArray array = initTestArray();

    TEST_EQUAL(array.get(1)->getString(), "test");

    // negative tests
    bool isNullptr = array.get(10) == nullptr;
    TEST_EQUAL(isNullptr, true);
    isNullptr = array.get("2") == nullptr;
    TEST_EQUAL(isNullptr, true);
}

/**
 * getSize_test
 */
void
DataItems_DataArray_Test::getSize_test()
{
    DataArray array = initTestArray();
    TEST_EQUAL(array.size(), 5);
}

/**
 * remove_test
 */
void
DataItems_DataArray_Test::remove_test()
{
    DataArray array = initTestArray();
    TEST_EQUAL(array.remove(1), true);
    TEST_EQUAL(array.remove("2"), true);

    TEST_EQUAL(array.get(1)->getInt(), 42);
    TEST_EQUAL(array.size(), 3);

    // negative tests
    TEST_EQUAL(array.remove(10), false);
}

/**
 * @brief clear_test
 */
void
DataItems_DataArray_Test::clear_test()
{
    DataArray array = initTestArray();
    array.clear();
    TEST_EQUAL(array.size(), 0);
}

/**
 * copy_test
 */
void
DataItems_DataArray_Test::copy_test()
{
    DataArray array = initTestArray();

    DataArray* arrayCopy = dynamic_cast<DataArray*>(array.copy());

    bool isNullptr = arrayCopy == nullptr;
    TEST_EQUAL(isNullptr, false);

    TEST_EQUAL(array.toString(), arrayCopy->toString());

    delete arrayCopy;
}

/**
 * toString_test
 */
void
DataItems_DataArray_Test::toString_test()
{
    DataArray array = initTestArray();

    std::string compare = "[\"\",\"test\",42,42.500000,null]";
    TEST_EQUAL(array.toString(), compare);

    compare = "[\n"
              "    \"\",\n"
              "    \"test\",\n"
              "    42,\n"
              "    42.500000,\n"
              "    null\n"
              "]";
    TEST_EQUAL(array.toString(true), compare);
}

/**
 * getType_test
 */
void
DataItems_DataArray_Test::getType_test()
{
    DataArray array = initTestArray();
    TEST_EQUAL(array.getType(), DataItem::ARRAY_TYPE);
}

/**
 * isValue_isMap_isArray_test
 */
void
DataItems_DataArray_Test::isValue_isMap_isArray_test()
{
    DataArray array = initTestArray();
    TEST_EQUAL(array.isValue(), false);
    TEST_EQUAL(array.isMap(), false);
    TEST_EQUAL(array.isArray(), true);
}

/**
 * toValue_toMap_toArray_test
 */
void
DataItems_DataArray_Test::toValue_toMap_toArray_test()
{
    DataArray array = initTestArray();

    bool isNullptr = array.toMap() == nullptr;
    TEST_EQUAL(isNullptr, true);

    isNullptr = array.toArray() == nullptr;
    TEST_EQUAL(isNullptr, false);

    isNullptr = array.toValue() == nullptr;
    TEST_EQUAL(isNullptr, true);
}

/**
 * getString_getInt_getFloat_test
 */
void
DataItems_DataArray_Test::getString_getInt_getFloat_test()
{
    DataArray array = initTestArray();
    TEST_EQUAL(array.getString(), "");
    TEST_EQUAL(array.getInt(), 0);
    TEST_EQUAL(array.getFloat(), 0.0f);
}

/**
 * create test data-array
 *
 * @return data-array for tests
 */
DataArray
DataItems_DataArray_Test::initTestArray()
{
    DataArray array;
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);

    array.append(defaultValue.copy());
    array.append(stringValue.copy());
    array.append(intValue.copy());
    array.append(floatValue.copy());
    array.append(nullptr);

    return array;
}

}  // namespace Kitsunemimi
