/**
 *  @file       arg_parser_test.cpp
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  Apache License Version 2.0
 *
 *      Copyright 2022 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use this file except in compliance with the License.
 *      You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

#include "arg_parser_test.h"

#include <hanami_common/items/data_items.h>
#include <hanami_args/arg_parser.h>

namespace Kitsunemimi
{

ArgParser_Test::ArgParser_Test()
    : Kitsunemimi::CompareTestHelper("ArgParser_Test")
{
    registerArgument_test();
    getArgument_test();
    convertValue_test();

    registerString_test();
    registerInteger_test();
    registerFloat_test();
    registerBoolean_test();

    parse_test();

    getNumberOfValues_test();
    getStringValues_test();
    getIntValues_test();
    getFloatValues_test();
    getBoolValues_test();

    getStringValue_test();
    getIntValue_test();
    getFloatValue_test();
    getBoolValue_test();
}

/**
 * @brief registerArgument_test
 */
void
ArgParser_Test::registerArgument_test()
{
    ArgParser parser;
    ErrorContainer error;

    TEST_EQUAL(parser.registerArgument("",
                                       "this is an example",
                                       ArgParser::ArgType::INT_TYPE,
                                       true,
                                       false,
                                       true,
                                       error)
               , false);
    TEST_EQUAL(parser.registerArgument("xyz,",
                                       "this is an example",
                                       ArgParser::ArgType::INT_TYPE,
                                       true,
                                       false,
                                       true,
                                       error)
               , true);
    TEST_EQUAL(parser.registerArgument(",a",
                                       "this is an example",
                                       ArgParser::ArgType::INT_TYPE,
                                       true,
                                       false,
                                       true,
                                       error)
               , false);
    TEST_EQUAL(parser.registerArgument("asdf,asdf",
                                       "this is an example",
                                       ArgParser::ArgType::INT_TYPE,
                                       true,
                                       false,
                                       true,
                                       error)
               , false);

    TEST_EQUAL(parser.registerArgument("asdf,a",
                                       "this is an example",
                                       ArgParser::ArgType::INT_TYPE,
                                       true,
                                       false,
                                       true,
                                       error)
               , true);
    TEST_EQUAL(parser.registerArgument("asdf,a",
                                       "this is an example",
                                       ArgParser::ArgType::INT_TYPE,
                                       true,
                                       false,
                                       true,
                                       error)
               , false);
}

/**
 * @brief getArgument_test
 */
void
ArgParser_Test::getArgument_test()
{
    ArgParser parser;
    bool isNullptr = false;
    ErrorContainer error;

    parser.registerArgument("asdf,a",
                            "this is an example",
                            ArgParser::ArgType::INT_TYPE,
                            true,
                            false,
                            true,
                            error);

    isNullptr = parser.getArgument("xyz") == nullptr;
    TEST_EQUAL(isNullptr, true);

    isNullptr = parser.getArgument("--asdf") == nullptr;
    TEST_EQUAL(isNullptr, false);

    isNullptr = parser.getArgument("-a") == nullptr;
    TEST_EQUAL(isNullptr, false);

    ArgParser::ArgDefinition* ret = parser.getArgument("-a");
    TEST_EQUAL(ret->type, ArgParser::ArgType::INT_TYPE);
    TEST_EQUAL(ret->helpText, std::string("this is an example"));
    TEST_EQUAL(ret->required, true);
    TEST_EQUAL(ret->withoutFlag, false);
    TEST_EQUAL(ret->longIdentifier, "--asdf");
    TEST_EQUAL(ret->shortIdentifier, "-a");
}

/**
 * @brief convertValue_test
 */
void
ArgParser_Test::convertValue_test()
{
    ArgParser parser;
    DataItem* result = nullptr;
    bool isNullptr = false;
    ErrorContainer error;

    // string-type
    // check if result is nullptr
    isNullptr = parser.convertValue("asdf", ArgParser::ArgType::STRING_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);
    isNullptr = parser.convertValue("1", ArgParser::ArgType::STRING_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);

    // check result value
    result = parser.convertValue("asdf", ArgParser::ArgType::STRING_TYPE);
    TEST_EQUAL(result->toValue()->getValueType(), DataValue::STRING_TYPE);
    TEST_EQUAL(result->getString(), "asdf");

    // int-type
    // check if result is nullptr
    isNullptr = parser.convertValue("1", ArgParser::ArgType::INT_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);
    isNullptr = parser.convertValue("asdf", ArgParser::ArgType::INT_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, true);

    // check result value
    result = parser.convertValue("42", ArgParser::ArgType::INT_TYPE);
    TEST_EQUAL(result->toValue()->getValueType(), DataValue::INT_TYPE);
    TEST_EQUAL(result->getInt(), 42);

    // float-type
    // check if result is nullptr
    isNullptr = parser.convertValue("1.0", ArgParser::ArgType::FLOAT_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);
    isNullptr = parser.convertValue("1", ArgParser::ArgType::FLOAT_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);
    isNullptr = parser.convertValue("asdf", ArgParser::ArgType::FLOAT_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, true);

    // check result value
    result = parser.convertValue("42.25", ArgParser::ArgType::FLOAT_TYPE);
    TEST_EQUAL(result->toValue()->getValueType(), DataValue::FLOAT_TYPE);
    TEST_EQUAL(result->getDouble(), 42.25);

    // bool-type
    // check if result is nullptr
    isNullptr = parser.convertValue("1", ArgParser::ArgType::BOOL_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);
    isNullptr = parser.convertValue("true", ArgParser::ArgType::BOOL_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);
    isNullptr = parser.convertValue("True", ArgParser::ArgType::BOOL_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);
    isNullptr = parser.convertValue("0", ArgParser::ArgType::BOOL_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);
    isNullptr = parser.convertValue("false", ArgParser::ArgType::BOOL_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);
    isNullptr = parser.convertValue("False", ArgParser::ArgType::BOOL_TYPE) == nullptr;
    TEST_EQUAL(isNullptr, false);

    // check result value
    result = parser.convertValue("true", ArgParser::ArgType::BOOL_TYPE);
    TEST_EQUAL(result->toValue()->getValueType(), DataValue::BOOL_TYPE);
    TEST_EQUAL(result->getBool(), true);
}

/**
 * @brief registerString_test
 */
void
ArgParser_Test::registerString_test()
{
    ArgParser parser;
    ErrorContainer error;

    TEST_EQUAL(parser.registerString("", "this is an example", error, true, false), false);
    TEST_EQUAL(parser.registerString("asdf", "this is an example", error, true, false), true);
}

/**
 * @brief registerInteger_test
 */
void
ArgParser_Test::registerInteger_test()
{
    ArgParser parser;
    ErrorContainer error;

    TEST_EQUAL(parser.registerInteger("", "this is an example", error, true, false), false);
    TEST_EQUAL(parser.registerInteger("asdf", "this is an example", error, true, false), true);
}

/**
 * @brief registerFloat_test
 */
void
ArgParser_Test::registerFloat_test()
{
    ArgParser parser;
    ErrorContainer error;

    TEST_EQUAL(parser.registerFloat("", "this is an example", error, true, false), false);
    TEST_EQUAL(parser.registerFloat("asdf", "this is an example", error, true, false), true);
}

/**
 * @brief registerBoolean_test
 */
void
ArgParser_Test::registerBoolean_test()
{
    ArgParser parser;
    ErrorContainer error;

    TEST_EQUAL(parser.registerBoolean("", "this is an example", error, true, false), false);
    TEST_EQUAL(parser.registerBoolean("asdf", "this is an example", error, true, false), true);
}

/**
 * @brief parse_test
 */
void
ArgParser_Test::parse_test()
{
    ArgParser parser;
    ErrorContainer error;

    int argc = 15;
    const char* argv[15];
    argv[0] = "-";
    argv[1] = "--test";
    argv[2] = "test1";
    argv[3] = "--test";
    argv[4] = "test2";
    argv[5] = "--integer";
    argv[6] = "1337";
    argv[7] = "-f";
    argv[8] = "123.5";
    argv[9] = "-b";
    argv[10] = "true";
    argv[11] = "poi";
    argv[12] = "42";
    argv[13] = "42.25";
    argv[14] = "false";

    parser.registerString("test", " ", error);
    parser.registerInteger("integer,i", " ", error);
    parser.registerFloat("float,f", " ", error);
    parser.registerBoolean("bool,b", " ", error);
    parser.registerString("first_arg", "first argument", error, true, true);
    parser.registerInteger("secondArg", "second argument", error, true, true);
    parser.registerFloat("thirdArg", "third argument", error, true, true);
    parser.registerBoolean("lastArg", "last argument", error, true, true);

    TEST_EQUAL(parser.parse(argc, argv, error), true);

    // negative test: set argument `bool` to a non-bool value
    argv[6] = "asdf";
    TEST_EQUAL(parser.parse(argc, argv, error), false);
    argv[6] = "true";

    // negative test: set a value without flag to a false type
    argv[12] = "asdf";
    TEST_EQUAL(parser.parse(argc, argv, error), false);
     argv[12] = "42";

    // negative test: register a required value, which is not given in the arguments
    parser.registerBoolean("fail", "this is a boolean", error, true);
    TEST_EQUAL(parser.parse(argc, argv, error), false);
}

/**
 * @brief getNumberOfValues_test
 */
void
ArgParser_Test::getNumberOfValues_test()
{
    ArgParser parser;
    prepareTest(&parser);

    TEST_EQUAL(parser.getNumberOfValues("test"), 2);
    TEST_EQUAL(parser.getNumberOfValues("bool"), 1);
    TEST_EQUAL(parser.getNumberOfValues("first_arg"), 1);
}

/**
 * @brief getStringValues_test
 */
void
ArgParser_Test::getStringValues_test()
{
    ArgParser parser;
    prepareTest(&parser);

    const std::vector<std::string> ret = parser.getStringValues("test");
    TEST_EQUAL(ret.size(), 2);
    TEST_EQUAL(ret.at(0), "test1");
    TEST_EQUAL(ret.at(1), "test2");

    const std::vector<std::string> ret2 = parser.getStringValues("first_arg");
    TEST_EQUAL(ret2.size(), 1);
    TEST_EQUAL(ret2.at(0), "poi");
}

/**
 * @brief getIntValues_test
 */
void
ArgParser_Test::getIntValues_test()
{
    ArgParser parser;
    prepareTest(&parser);

    const std::vector<long> ret = parser.getIntValues("integer");
    TEST_EQUAL(ret.size(), 1);
    TEST_EQUAL(ret.at(0), 1337);

    const std::vector<long> ret2 = parser.getIntValues("secondArg");
    TEST_EQUAL(ret2.size(), 1);
    TEST_EQUAL(ret2.at(0), 42);
}

/**
 * @brief getFloatValues_test
 */
void
ArgParser_Test::getFloatValues_test()
{
    ArgParser parser;
    prepareTest(&parser);

    const std::vector<double> ret = parser.getFloatValues("f");
    TEST_EQUAL(ret.size(), 1);
    TEST_EQUAL(ret.at(0), 123.5);

    const std::vector<double> ret2 = parser.getFloatValues("thirdArg");
    TEST_EQUAL(ret2.size(), 1);
    TEST_EQUAL(ret2.at(0), 42.25);
}

/**
 * @brief getBoolValues_test
 */
void
ArgParser_Test::getBoolValues_test()
{
    ArgParser parser;
    prepareTest(&parser);

    const std::vector<bool> ret = parser.getBoolValues("bool");
    TEST_EQUAL(ret.size(), 1);
    TEST_EQUAL(ret.at(0), true);

    const std::vector<bool> ret2 = parser.getBoolValues("lastArg");
    TEST_EQUAL(ret2.size(), 1);
    TEST_EQUAL(ret2.at(0), false);
}

/**
 * @brief getStringValue_test
 */
void
ArgParser_Test::getStringValue_test()
{
    ArgParser parser;
    prepareTest(&parser);

    const std::string ret2 = parser.getStringValue("first_arg");
    TEST_EQUAL(ret2, "poi");
}

/**
 * @brief getIntValue_test
 */
void
ArgParser_Test::getIntValue_test()
{
    ArgParser parser;
    prepareTest(&parser);

    const long ret2 = parser.getIntValue("secondArg");
    TEST_EQUAL(ret2, 42);
}

/**
 * @brief getFloatValue_test
 */
void
ArgParser_Test::getFloatValue_test()
{
    ArgParser parser;
    prepareTest(&parser);

    const double ret2 = parser.getFloatValue("thirdArg");
    TEST_EQUAL(ret2, 42.25);
}

/**
 * @brief getBoolValue_test
 */
void
ArgParser_Test::getBoolValue_test()
{
    ArgParser parser;
    prepareTest(&parser);

    const bool ret2 = parser.getBoolValue("lastArg");
    TEST_EQUAL(ret2, false);
}

/**
 * @brief ArgParser_Test::prepareTest
 * @param parser
 */
void
ArgParser_Test::prepareTest(ArgParser *parser)
{
    ErrorContainer error;

    int argc = 15;
    const char* argv[15];
    argv[0] = "-";
    argv[1] = "--test";
    argv[2] = "test1";
    argv[3] = "--test";
    argv[4] = "test2";
    argv[5] = "--integer";
    argv[6] = "1337";
    argv[7] = "-f";
    argv[8] = "123.5";
    argv[9] = "-b";
    argv[10] = "true";
    argv[11] = "poi";
    argv[12] = "42";
    argv[13] = "42.25";
    argv[14] = "false";

    parser->registerString("test", " ", error);
    parser->registerInteger("integer,i", " ", error);
    parser->registerFloat("float,f", " ", error);
    parser->registerBoolean("bool,b", " ", error);
    parser->registerString("first_arg", "first argument", error, true, true);
    parser->registerInteger("secondArg", "second argument", error, true, true);
    parser->registerFloat("thirdArg", "third argument", error, true, true);
    parser->registerBoolean("lastArg", "last argument", error, true, true);

    assert(parser->parse(argc, argv, error));
}

}
