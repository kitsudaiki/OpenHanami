/**
 *  @file    jsonItem_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "json_item_test.h"
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi
{

JsonItem_Test::JsonItem_Test()
    : Kitsunemimi::CompareTestHelper("JsonItem_Test")
{
    constructor_test();
    assigmentOperator_test();
    setValue_test();
    insert_test();
    append_test();
    replaceItem_test();
    deleteContent_test();

    getItemContent_test();
    get_test();
    getString_getInt_getFloat_test();

    size_test();
    getKeys_test();
    contains_test();

    isValid_test();
    isNull_test();
    isMap_isArray_isValue_test();
    isString_isInteger_isFloat_isBool_test();

    remove_test();
}

/**
 * @brief constructor_test
 */
void
JsonItem_Test::constructor_test()
{
    JsonItem emptyItem;
    TEST_EQUAL(emptyItem.isValid(), false);

    std::map<std::string, JsonItem> emptyMap;
    JsonItem objectItem(emptyMap);
    TEST_EQUAL(objectItem.isMap(), true);

    JsonItem objectCopyItem(objectItem);
    TEST_EQUAL(objectCopyItem.isMap(), true);

    std::vector<JsonItem> emptyArray;
    JsonItem arrayItem(emptyArray);
    TEST_EQUAL(arrayItem.isArray(), true);

    DataMap testMap;
    JsonItem mapItem(&testMap, true);
    TEST_EQUAL(mapItem.isMap(), true);

    JsonItem stringItem("test");
    TEST_EQUAL(stringItem.isValue(), true);

    JsonItem intItem(42);
    TEST_EQUAL(intItem.isValue(), true);

    JsonItem floatItem(42.0f);
    TEST_EQUAL(floatItem.isValue(), true);

    JsonItem longItem(42l);
    TEST_EQUAL(longItem.isValue(), true);

    JsonItem doubleItem(42.0);
    TEST_EQUAL(doubleItem.isValue(), true);

    JsonItem boolItem(true);
    TEST_EQUAL(boolItem.isValue(), true);
}

/**
 * @brief assigmentOperator_test
 */
void
JsonItem_Test::assigmentOperator_test()
{
    JsonItem testItem = getTestItem();

    JsonItem copy;
    copy = testItem;

    TEST_EQUAL(copy.toString(), testItem.toString());

    DataMap testMap;
    testMap.insert("test", new DataValue("test"));
    testItem = &testMap;
    TEST_EQUAL(testItem.toString(), testMap.toString());
}

/**
 * @brief setValue_test
 */
void
JsonItem_Test::setValue_test()
{
    JsonItem testItem;
    TEST_EQUAL(testItem.setValue("test"), true);
    TEST_EQUAL(testItem.setValue(42), true);
    TEST_EQUAL(testItem.setValue(42.0f), true);
    TEST_EQUAL(testItem.setValue(42l), true);
    TEST_EQUAL(testItem.setValue(42.0), true);
    TEST_EQUAL(testItem.setValue(true), true);

    TEST_EQUAL(testItem.isValid(), true);


    // negative test
    std::map<std::string, JsonItem> emptyMap;
    JsonItem objectItem(emptyMap);
    TEST_EQUAL(objectItem.setValue("test"), false);
}

/**
 * @brief insert_test
 */
void
JsonItem_Test::insert_test()
{
    JsonItem testItem;
    TEST_EQUAL(testItem.insert("key", JsonItem(42)), true);
    TEST_EQUAL(testItem.insert("key", JsonItem("24"), true), true);
    TEST_EQUAL(testItem["key"].getString(), "24");
    TEST_EQUAL(testItem.isMap(), true);

    // negative test
    TEST_EQUAL(testItem.insert("key", JsonItem(43)), false);
    TEST_EQUAL(testItem.insert("fail", JsonItem()), false);
}

/**
 * @brief append_test
 */
void
JsonItem_Test::append_test()
{
    JsonItem testItem;
    TEST_EQUAL(testItem.append(JsonItem(42)), true);
    TEST_EQUAL(testItem.isArray(), true);
    TEST_EQUAL(testItem[0].getInt(), 42);

    // negative test
    TEST_EQUAL(testItem.append(JsonItem()), false);
}

/**
 * @brief replaceItem_test
 */
void
JsonItem_Test::replaceItem_test()
{
    JsonItem testItem;
    testItem.append(JsonItem(42));
    testItem.append(JsonItem("42"));
    TEST_EQUAL(testItem[0].toString(), "42");
    TEST_EQUAL(testItem.replaceItem(0, JsonItem("ok")), true);
    TEST_EQUAL(testItem[0].toString(), "ok");

    // negative test
    TEST_EQUAL(testItem.replaceItem(10, JsonItem("fail")), false);
    TEST_EQUAL(testItem.replaceItem(0, JsonItem()), false);
}

/**
 * @brief deleteContent_test
 */
void
JsonItem_Test::deleteContent_test()
{
    JsonItem testItem;
    testItem.append(JsonItem(42));
    testItem.append(JsonItem("42"));
    TEST_EQUAL(testItem.isNull(), false);
    TEST_EQUAL(testItem.deleteContent(), true);
    TEST_EQUAL(testItem.deleteContent(), false);
    TEST_EQUAL(testItem.isNull(), true);
}

/**
 * @brief getItemContent_test
 */
void
JsonItem_Test::getItemContent_test()
{
    JsonItem testItem = getTestItem();
    DataItem* itemPtr = testItem.getItemContent();

    TEST_EQUAL(itemPtr->toString(true), testItem.toString(true));
}

/**
 * @brief get_test
 */
void
JsonItem_Test::get_test()
{
    JsonItem testItem = getTestItem();
    TEST_EQUAL(testItem["loop"][0]["x"].toString(), "42");
    TEST_EQUAL(testItem.get("loop").get(0).get("x").toString(), "42");
    TEST_EQUAL(testItem.get("loop").get(0).get("x").setValue("43"), true);
    TEST_EQUAL(testItem.get("loop").get(0).get("x").toString(), "43");
}

/**
 * @brief getString_getInt_getFloat_test
 */
void
JsonItem_Test::getString_getInt_getFloat_test()
{
    JsonItem stringValue("test");
    JsonItem intValue(42);
    JsonItem floatValue(42.5f);
    JsonItem longValue(42l);
    JsonItem doubleValue(42.5);
    JsonItem boolValue(true);

    // string-value
    TEST_EQUAL(stringValue.getString(), "test");
    TEST_EQUAL(stringValue.getInt(), 0);
    TEST_EQUAL(stringValue.getFloat(), 0.0f);
    TEST_EQUAL(stringValue.getLong(), 0l);
    TEST_EQUAL(stringValue.getDouble(), 0.0);
    TEST_EQUAL(stringValue.getBool(), false);

    // int-value
    TEST_EQUAL(intValue.getString(), "");
    TEST_EQUAL(intValue.getInt(), 42);
    TEST_EQUAL(intValue.getFloat(), 0.0f);
    TEST_EQUAL(intValue.getLong(), 42l);
    TEST_EQUAL(intValue.getDouble(), 0.0);
    TEST_EQUAL(intValue.getBool(), false);

    // float-value
    TEST_EQUAL(floatValue.getString(), "");
    TEST_EQUAL(floatValue.getInt(), 0);
    TEST_EQUAL(floatValue.getFloat(), 42.5f);
    TEST_EQUAL(floatValue.getLong(), 0l);
    TEST_EQUAL(floatValue.getDouble(), 42.5);
    TEST_EQUAL(floatValue.getBool(), false);

    // long-value
    TEST_EQUAL(longValue.getString(), "");
    TEST_EQUAL(longValue.getInt(), 42);
    TEST_EQUAL(longValue.getFloat(), 0.0f);
    TEST_EQUAL(longValue.getLong(), 42l);
    TEST_EQUAL(longValue.getDouble(), 0.0);
    TEST_EQUAL(longValue.getBool(), false);

    // double-value
    TEST_EQUAL(doubleValue.getString(), "");
    TEST_EQUAL(doubleValue.getInt(), 0);
    TEST_EQUAL(doubleValue.getFloat(), 42.5f);
    TEST_EQUAL(doubleValue.getLong(), 0l);
    TEST_EQUAL(doubleValue.getDouble(), 42.5);
    TEST_EQUAL(doubleValue.getBool(), false);

    // bool-value
    TEST_EQUAL(boolValue.getString(), "");
    TEST_EQUAL(boolValue.getInt(), 0);
    TEST_EQUAL(boolValue.getFloat(), 0.0f);
    TEST_EQUAL(boolValue.getLong(), 0l);
    TEST_EQUAL(boolValue.getDouble(), 0.0);
    TEST_EQUAL(boolValue.getBool(), true);
}

/**
 * @brief size_test
 */
void
JsonItem_Test::size_test()
{
    JsonItem testItem = getTestItem();
    TEST_EQUAL(testItem.size(), 3);
}

/**
 * @brief getKeys_test
 */
void
JsonItem_Test::getKeys_test()
{
    JsonItem testItem = getTestItem();
    std::vector<std::string> keys = testItem.getKeys();
    TEST_EQUAL(keys.size(), 3);
    TEST_EQUAL(keys.at(0), "item");
    TEST_EQUAL(keys.at(1), "item2");
    TEST_EQUAL(keys.at(2), "loop");
}

/**
 * @brief contains_test
 */
void
JsonItem_Test::contains_test()
{
    JsonItem testItem = getTestItem();
    TEST_EQUAL(testItem.contains("item"), true);
    TEST_EQUAL(testItem.contains("fail"), false);
}

/**
 * @brief isValid_test
 */
void
JsonItem_Test::isValid_test()
{
    JsonItem emptyItem;
    TEST_EQUAL(emptyItem.isValid(), false);
    JsonItem testItem = getTestItem();
    TEST_EQUAL(testItem.isValid(), true);
}

/**
 * @brief isNull_test
 */
void
JsonItem_Test::isNull_test()
{
    JsonItem emptyItem;
    TEST_EQUAL(emptyItem.isNull(), true);
    JsonItem testItem = getTestItem();
    TEST_EQUAL(testItem.isNull(), false);
}

/**
 * @brief isMap_isArray_isValue_test
 */
void
JsonItem_Test::isMap_isArray_isValue_test()
{
    std::map<std::string, JsonItem> emptyMap;
    JsonItem objectItem(emptyMap);
    TEST_EQUAL(objectItem.isMap(), true);

    std::vector<JsonItem> emptyArray;
    JsonItem arrayItem(emptyArray);
    TEST_EQUAL(arrayItem.isArray(), true);

    JsonItem stringItem("test");
    TEST_EQUAL(stringItem.isValue(), true);
}

/**
 * @brief isString_isInteger_isFloat_isBool_test
 */
void
JsonItem_Test::isString_isInteger_isFloat_isBool_test()
{
    JsonItem stringItem("test");
    TEST_EQUAL(stringItem.isString(), true);
    TEST_EQUAL(stringItem.isInteger(), false);
    TEST_EQUAL(stringItem.isFloat(), false);
    TEST_EQUAL(stringItem.isBool(), false);

    JsonItem intItem(42);
    TEST_EQUAL(intItem.isString(), false);
    TEST_EQUAL(intItem.isInteger(), true);
    TEST_EQUAL(intItem.isFloat(), false);
    TEST_EQUAL(intItem.isBool(), false);

    JsonItem floatItem(42.0f);
    TEST_EQUAL(floatItem.isString(), false);
    TEST_EQUAL(floatItem.isInteger(), false);
    TEST_EQUAL(floatItem.isFloat(), true);
    TEST_EQUAL(floatItem.isBool(), false);

    JsonItem longItem(42l);
    TEST_EQUAL(longItem.isString(), false);
    TEST_EQUAL(longItem.isInteger(), true);
    TEST_EQUAL(longItem.isFloat(), false);
    TEST_EQUAL(longItem.isBool(), false);

    JsonItem doubleItem(42.0);
    TEST_EQUAL(doubleItem.isString(), false);
    TEST_EQUAL(doubleItem.isInteger(), false);
    TEST_EQUAL(doubleItem.isFloat(), true);
    TEST_EQUAL(doubleItem.isBool(), false);

    JsonItem boolItem(true);
    TEST_EQUAL(boolItem.isString(), false);
    TEST_EQUAL(boolItem.isInteger(), false);
    TEST_EQUAL(boolItem.isFloat(), false);
    TEST_EQUAL(boolItem.isBool(), true);
}

/**
 * @brief remove_test
 */
void
JsonItem_Test::remove_test()
{
    JsonItem testItem = getTestItem();

    TEST_EQUAL(testItem.remove("item"), true);
    TEST_EQUAL(testItem.remove("item"), false);
    TEST_EQUAL(testItem.size(), 2);
}

/**
 * @brief get a item for tests
 *
 * @return json-item with test-content
 */
JsonItem
JsonItem_Test::getTestItem()
{
    std::string input("{\n"
                      "    item: {\n"
                      "        sub_item: \"test_value\"\n"
                      "    },\n"
                      "    item2: {\n"
                      "        sub_item2: \"something\"\n"
                      "    },\n"
                      "    loop: [\n"
                      "        {\n"
                      "            x: 42\n"
                      "        },\n"
                      "        {\n"
                      "            x: 42.000000\n"
                      "        },\n"
                      "        1234,\n"
                      "        {\n"
                      "            x: -42.000000\n"
                      "        }\n"
                      "    ]\n"
                      "}");

    JsonItem output;
    ErrorContainer error;
    output.parse(input, error);

    assert(output.isValid());

    return output;
}

}  // namespace Kitsunemimi
