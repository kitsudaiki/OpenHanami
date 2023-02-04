/**
 *  @file    jsonItems_parseString_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "json_item_parseString_test.h"
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi
{

JsonItem_ParseString_Test::JsonItem_ParseString_Test()
    : Kitsunemimi::MemoryLeakTestHelpter("JsonItems_ParseString_Test")
{
    parseString_test();
}

/**
 * parseString_test
 */
void
JsonItem_ParseString_Test::parseString_test()
{

    const std::string validInput1 =
            "{\"item\": "
            "{ \"sub_item\": \"test_value\"},"
            "\"item2\": "
            "{ \"sub_item2\": \"something\"},"
            "\"loop\": "
            "[ {\"x\" :42 }, {\"x\" :42.0 }, 1234, {\"x\" :-42.0, \"y\": true, \"z\": false, \"w\": null}]"
            "}";
    const std::string validInput2 =
            "[ {x :\"test1\" }, {x :\"test2\" }, {x :\"test3\" }]";
    const std::string invalidInput =
            "{item: \n"
            "{ sub_item: \"test_value\"}, \n"
            "item2: \n"
            "[ sub_item2: \"something\"}, \n"  // error at the beginning of this line
            "loop: \n"
            "[ {x :\"test1\" }, {x :\"test2\" }, {x :\"test3\" }]\n"
            "}";

    ErrorContainer error;

    // make one untested run to allow parser to allocate one-time global stuff
    JsonItem* paredItem = new JsonItem();
    paredItem->parse(validInput1, error);
    paredItem->parse(invalidInput, error);
    error._errorMessages.clear();
    error._possibleSolution.clear();
    delete paredItem;

    // parse valid string
    REINIT_TEST();
    paredItem = new JsonItem();
    paredItem->parse(validInput1, error);
    paredItem->parse(validInput1, error);
    paredItem->parse(validInput2, error);
    delete paredItem;
    CHECK_MEMORY();


    // parse invalid string
    REINIT_TEST();
    paredItem = new JsonItem();
    paredItem->parse(invalidInput, error);
    error._errorMessages.clear();
    error._possibleSolution.clear();
    delete paredItem;
    CHECK_MEMORY();
}

}  // namespace Kitsunemimi

