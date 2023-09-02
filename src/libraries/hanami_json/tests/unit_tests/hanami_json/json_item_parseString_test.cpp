/**
 *  @file    jsonItems_parseString_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "json_item_parseString_test.h"
#include <hanami_json/json_item.h>
#include <hanami_common/items/data_items.h>

namespace Hanami
{

JsonItem_ParseString_Test::JsonItem_ParseString_Test()
    : Hanami::CompareTestHelper("JsonItems_ParseString_Test")
{
    parseString_test();
}

/**
 * parseString_test
 */
void
JsonItem_ParseString_Test::parseString_test()
{
    // positive test
    std::string input("{\"item\": "
                      "{ \"sub_item\": \"test_value\"},"
                      "\"item2\": "
                      "{ \"sub_item2\": \"something\"},"
                      "\"loop\": "
                      "[ {\"x\" :42 }, {\"x\" :42.0 }, 1234, {\"x\" :-42.0, \"y\": true, \"z\": false, \"w\": null}]"
                      "}");

    JsonItem paredItem;
    ErrorContainer error;
    bool result = paredItem.parse(input, error);
    TEST_EQUAL(result, true);
    if(result == false) {
        LOG_ERROR(error);
    }

    std::string outputStringMaps = paredItem.toString(true);
    std::string compareMaps("{\n"
                            "    \"item\": {\n"
                            "        \"sub_item\": \"test_value\"\n"
                            "    },\n"
                            "    \"item2\": {\n"
                            "        \"sub_item2\": \"something\"\n"
                            "    },\n"
                            "    \"loop\": [\n"
                            "        {\n"
                            "            \"x\": 42\n"
                            "        },\n"
                            "        {\n"
                            "            \"x\": 42.000000\n"
                            "        },\n"
                            "        1234,\n"
                            "        {\n"
                            "            \"w\": null,\n"
                            "            \"x\": -42.000000,\n"
                            "            \"y\": true,\n"
                            "            \"z\": false\n"
                            "        }\n"
                            "    ]\n"
                            "}");
    TEST_EQUAL(outputStringMaps, compareMaps);

    input = "[ {x :\"test1\" }, {x :\"test2\" }, {x :\"test3\" }]";
    result = paredItem.parse(input, error);
    TEST_EQUAL(result, true);


    // empty test
    input = "";
    JsonItem emptyItem;
    result = emptyItem.parse(input, error);
    TEST_EQUAL(result, true);
    TEST_EQUAL(emptyItem.getItemContent()->getType(), Hanami::DataItem::MAP_TYPE);
    TEST_EQUAL(emptyItem.getItemContent()->toMap()->size(), 0);

    // negative test
    input = "{item: \n"
            "{ sub_item: \"test_value\"}, \n"
            "item2: \n"
            "[ sub_item2: \"something\"}, \n"  // error at the beginning of this line
            "loop: \n"
            "[ {x :\"test1\" }, {x :\"test2\" }, {x :\"test3\" }]\n"
            "}";

    JsonItem output;
    result = output.parse(input, error);
    TEST_EQUAL(result, false);

    const std::string expectedError =
            "+---------------------+-------------------------------------------+\n"
            "| Error-Message Nr. 0 | ERROR while parsing json-formated string  |\n"
            "|                     | parser-message: syntax error              |\n"
            "|                     | line-number: 4                            |\n"
            "|                     | position in line: 12                      |\n"
            "|                     | broken part in string: \":\"                |\n"
            "+---------------------+-------------------------------------------+\n";
    TEST_EQUAL(error.toString(), expectedError);
}

}

