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
    : Kitsunemimi::MemoryLeakTestHelpter("JsonItem_Test")
{
    constructor_test();
    assigmentOperator_test();
    insert_test();
    append_test();
    replaceItem_test();
    deleteContent_test();

    getItemContent_test();
    get_test();
}

/**
 * @brief constructor_test
 */
void
JsonItem_Test::constructor_test()
{
    REINIT_TEST();
    JsonItem* emptyItem = new JsonItem();
    delete emptyItem;
    CHECK_MEMORY();

    REINIT_TEST();
    std::map<std::string, JsonItem>* emptyMap = new std::map<std::string, JsonItem>();
    JsonItem* objectItem = new JsonItem(*emptyMap);
    delete objectItem;
    delete emptyMap;
    CHECK_MEMORY();

    REINIT_TEST();
    emptyMap = new std::map<std::string, JsonItem>();
    objectItem = new JsonItem(*emptyMap);
    JsonItem* objectCopyItem = new JsonItem(objectItem);
    delete objectItem;
    delete objectCopyItem;
    delete emptyMap;
    CHECK_MEMORY();

    REINIT_TEST();
    DataMap* input = new DataMap();
    input->insert("test", new DataValue("test"));
    objectItem = new JsonItem(input);
    delete objectItem;
    delete input;
    CHECK_MEMORY();

    REINIT_TEST();
    std::vector<JsonItem>* emptyArray = new std::vector<JsonItem>();
    JsonItem* arrayItem = new JsonItem(*emptyArray);
    delete emptyArray;
    delete arrayItem;
    CHECK_MEMORY();

    REINIT_TEST();
    JsonItem* stringItem = new JsonItem("test");
    delete stringItem;
    CHECK_MEMORY();

    REINIT_TEST();
    JsonItem* intItem = new JsonItem(42);
    delete intItem;
    CHECK_MEMORY();

    REINIT_TEST();
    JsonItem* floatItem = new JsonItem(42.0f);
    delete floatItem;
    CHECK_MEMORY();

    REINIT_TEST();
    JsonItem* longItem = new JsonItem(42l);
    delete longItem;
    CHECK_MEMORY();

    REINIT_TEST();
    JsonItem* doubleItem = new JsonItem(42.0);
    delete doubleItem;
    CHECK_MEMORY();

    REINIT_TEST();
    JsonItem* boolItem = new JsonItem(true);
    delete boolItem;
    CHECK_MEMORY();
}

/**
 * @brief assigmentOperator_test
 */
void
JsonItem_Test::assigmentOperator_test()
{
    JsonItem testItem = getTestItem();

    REINIT_TEST();
    JsonItem* copy = new JsonItem();
    *copy = testItem;
    delete copy;
    CHECK_MEMORY();


    REINIT_TEST();
    JsonItem* testItem2 = new JsonItem();
    DataMap* testMap = new DataMap();
    testMap->insert("test", new DataValue("test"));
    *testItem2 = testMap;
    delete testMap;
    delete testItem2;
    CHECK_MEMORY();
}

/**
 * @brief insert_test
 */
void
JsonItem_Test::insert_test()
{
    REINIT_TEST();
    JsonItem* testItem = new JsonItem();
    testItem->insert("key", JsonItem(42));
    testItem->insert("key", JsonItem("24"), true);
    testItem->isMap();
    delete testItem;
    CHECK_MEMORY();
}

/**
 * @brief append_test
 */
void
JsonItem_Test::append_test()
{
    REINIT_TEST();
    JsonItem* testItem = new JsonItem();
    testItem->append(JsonItem(42));
    testItem->isArray();
    delete testItem;
    CHECK_MEMORY();
}

/**
 * @brief replaceItem_test
 */
void
JsonItem_Test::replaceItem_test()
{
    REINIT_TEST();
    JsonItem* testItem = new JsonItem();
    JsonItem* okJson = new JsonItem("ok");
    JsonItem* _42Json = new JsonItem("42");
    testItem->append(*_42Json);
    testItem->replaceItem(0, *okJson);
    delete testItem;
    delete okJson;
    delete _42Json;
    CHECK_MEMORY();
}

/**
 * @brief deleteContent_test
 */
void
JsonItem_Test::deleteContent_test()
{
    REINIT_TEST();
    JsonItem* testItem = new JsonItem();
    testItem->append(JsonItem(42));
    testItem->append(JsonItem("42"));
    testItem->deleteContent();
    delete testItem;
    CHECK_MEMORY();
}

/**
 * @brief getItemContent_test
 */
void
JsonItem_Test::getItemContent_test()
{
    REINIT_TEST();
    JsonItem* testItem = new JsonItem();
    DataItem* itemPtr = testItem->getItemContent();
    delete testItem;
    delete itemPtr;
    CHECK_MEMORY();
}

/**
 * @brief get_test
 */
void
JsonItem_Test::get_test()
{
    std::string testString = " ";
    JsonItem testItem = getTestItem();

    REINIT_TEST();
    testString = testItem["loop"][0]["x"].toString();
    testString = testItem.get("loop").get(0).get("x").toString();
    testString = testItem.get("loop").get(0).get("x").setValue(43);
    CHECK_MEMORY();
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
