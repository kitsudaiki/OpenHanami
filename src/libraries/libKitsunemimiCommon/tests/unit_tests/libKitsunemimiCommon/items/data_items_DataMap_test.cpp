/**
 *  @file    data_items_DataMap_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "data_items_DataMap_test.h"
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi
{

DataItems_DataMap_Test::DataItems_DataMap_Test()
    : Kitsunemimi::CompareTestHelper("DataItems_DataMap_Test")
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

    // object-exclusive
    insert_test();
    getStringByKey_getIntByKey_getFloatByKey_test();
    getKeys_test();
    getValues_test();
    contains_test();
}

/**
 * copy_assingment_constructor_test
 */
void
DataItems_DataMap_Test::copy_assingment_constructor_test()
{
    DataMap map = initTestObject();
    DataMap mapCopy = map;
    TEST_EQUAL(mapCopy.toString(), map.toString());
}

/**
 * copy_assingment_operator_test
 */
void
DataItems_DataMap_Test::copy_assingment_operator_test()
{
    DataMap map = initTestObject();
    DataMap mapCopy;
    mapCopy = map;
    TEST_EQUAL(mapCopy.toString(), map.toString());
}

/**
 * operator_test
 */
void
DataItems_DataMap_Test::operator_test()
{
    DataMap object = initTestObject();

    TEST_EQUAL(object[0]->getString(), "test");
    TEST_EQUAL(object["hmm"]->getInt(), 42);

    // negative tests
    bool isNullptr = object[10] == nullptr;
    TEST_EQUAL(isNullptr, true);
    isNullptr = object["k"] == nullptr;
    TEST_EQUAL(isNullptr, true);
}

/**
 * get_test
 */
void
DataItems_DataMap_Test::get_test()
{
    DataMap object = initTestObject();

    TEST_EQUAL(object.get(0)->getString(), "test");
    TEST_EQUAL(object.get("hmm")->getInt(), 42);

    // negative tests
    bool isNullptr = object.get(10) == nullptr;
    TEST_EQUAL(isNullptr, true);
    isNullptr = object.get("k") == nullptr;
    TEST_EQUAL(isNullptr, true);
}

/**
 * getSize_test
 */
void
DataItems_DataMap_Test::getSize_test()
{
    DataMap object = initTestObject();
    TEST_EQUAL(object.size(), 5);
}

/**
 * remove_test
 */
void
DataItems_DataMap_Test::remove_test()
{
    DataMap object = initTestObject();
    TEST_EQUAL(object.remove(0), true);
    TEST_EQUAL(object.remove("hmm"), true);

    TEST_EQUAL(object.get(2)->toString(), "42.500000");
    TEST_EQUAL(object.size(), 3);

    // negative tests
    TEST_EQUAL(object.remove(10), false);
}

/**
 * @brief clear_test
 */
void
DataItems_DataMap_Test::clear_test()
{
    DataMap object = initTestObject();
    object.clear();
    TEST_EQUAL(object.size(), 0);
}

/**
 * copy_test
 */
void
DataItems_DataMap_Test::copy_test()
{
    DataMap object = initTestObject();

    DataMap* objectCopy = dynamic_cast<DataMap*>(object.copy());

    bool isNullptr = objectCopy == nullptr;
    TEST_EQUAL(isNullptr, false);

    TEST_EQUAL(object.toString(), objectCopy->toString());

    delete objectCopy;
}

/**
 * toString_test
 */
void
DataItems_DataMap_Test::toString_test()
{
    DataMap object = initTestObject();

    std::string compare = "{\"asdf\":\"test\","
                          "\"fail\":null,"
                          "\"hmm\":42,"
                          "\"poi\":\"\","
                          "\"xyz\":42.500000}";
    TEST_EQUAL(object.toString(), compare);

    compare = "{\n"
              "    \"asdf\": \"test\",\n"
              "    \"fail\": null,\n"
              "    \"hmm\": 42,\n"
              "    \"poi\": \"\",\n"
              "    \"xyz\": 42.500000\n"
              "}";
    TEST_EQUAL(object.toString(true), compare);
}

/**
 * getType_test
 */
void
DataItems_DataMap_Test::getType_test()
{
    DataMap object = initTestObject();
    TEST_EQUAL(object.getType(), DataItem::MAP_TYPE);
}

/**
 * isValue_isMap_isArray_test
 */
void
DataItems_DataMap_Test::isValue_isMap_isArray_test()
{
    DataMap object = initTestObject();
    TEST_EQUAL(object.isValue(), false);
    TEST_EQUAL(object.isMap(), true);
    TEST_EQUAL(object.isArray(), false);
}

/**
 * toValue_toMap_toArray_test
 */
void
DataItems_DataMap_Test::toValue_toMap_toArray_test()
{
    DataMap object = initTestObject();

    bool isNullptr = object.toMap() == nullptr;
    TEST_EQUAL(isNullptr, false);

    isNullptr = object.toArray() == nullptr;
    TEST_EQUAL(isNullptr, true);

    isNullptr = object.toValue() == nullptr;
    TEST_EQUAL(isNullptr, true);
}

/**
 * getString_getInt_getFloat_test
 */
void
DataItems_DataMap_Test::getString_getInt_getFloat_test()
{
    DataMap object = initTestObject();
    TEST_EQUAL(object.getString(), "");
    TEST_EQUAL(object.getInt(), 0);
    TEST_EQUAL(object.getFloat(), 0.0f);
}

/**
 * insert_test
 */
void
DataItems_DataMap_Test::insert_test()
{
    DataMap object;
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);

    TEST_EQUAL(object.insert("poi", defaultValue.copy()), true);
    TEST_EQUAL(object.insert("asdf", stringValue.copy()), true);
    TEST_EQUAL(object.insert("hmm", intValue.copy()), true);
    TEST_EQUAL(object.insert("xyz", floatValue.copy()), true);
    TEST_EQUAL(object.insert("fail", nullptr), true);
}

/**
 * getStringByKey_getIntByKey_getFloatByKey_test
 */
void
DataItems_DataMap_Test::getStringByKey_getIntByKey_getFloatByKey_test()
{
    DataMap object = initTestObject();

    TEST_EQUAL(object.getStringByKey("asdf"), "test");
    TEST_EQUAL(object.getIntByKey("hmm"), 42);
    TEST_EQUAL(object.getFloatByKey("xyz"), 42.5f);
}

/**
 * getKeys_test
 */
void
DataItems_DataMap_Test::getKeys_test()
{
    DataMap object = initTestObject();

    std::vector<std::string> keys = object.getKeys();
    TEST_EQUAL(keys.size(), 5);
    TEST_EQUAL(keys.at(0), "asdf");
    TEST_EQUAL(keys.at(1), "fail");
    TEST_EQUAL(keys.at(2), "hmm");
    TEST_EQUAL(keys.at(3), "poi");
    TEST_EQUAL(keys.at(4), "xyz");
}

/**
 * getValues_test
 */
void
DataItems_DataMap_Test::getValues_test()
{
    DataMap object = initTestObject();

    std::vector<DataItem*> values = object.getValues();
    TEST_EQUAL(values.size(), 5);
    TEST_EQUAL(values.at(0)->toString(), "test");
    TEST_EQUAL(values.at(2)->toString(), "42");
    TEST_EQUAL(values.at(3)->toString(), "");
    TEST_EQUAL(values.at(4)->toString(), "42.500000");
}

/**
 * contains_test
 */
void
DataItems_DataMap_Test::contains_test()
{
    DataMap object = initTestObject();
    TEST_EQUAL(object.contains("poi"), true);
    TEST_EQUAL(object.contains("asdf"), true);
    TEST_EQUAL(object.contains("hmm"), true);
    TEST_EQUAL(object.contains("xyz"), true);

    TEST_EQUAL(object.contains("12345"), false);
}

/**
 * create test data-map
 *
 * @return data-map for tests
 */
DataMap
DataItems_DataMap_Test::initTestObject()
{
    DataMap object;
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);

    object.insert("poi", defaultValue.copy());
    object.insert("asdf", stringValue.copy());
    object.insert("hmm", intValue.copy());
    object.insert("xyz", floatValue.copy());
    object.insert("fail", nullptr);

    return object;
}

}  // namespace Kitsunemimi
