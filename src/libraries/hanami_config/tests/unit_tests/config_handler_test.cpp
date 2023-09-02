/**
 *  @file       config_handler_test.cpp
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

#include "config_handler_test.h"

#include <hanami_config/config_handler.h>
#include <hanami_common/files/text_file.h>
#include <hanami_common/items/data_items.h>
#include <hanami_common/methods/file_methods.h>

namespace Hanami
{

ConfigHandler_Test::ConfigHandler_Test()
    : Hanami::CompareTestHelper("ConfigHandler_Test")
{
    initTestCase();

    readConfig_test();

    // private methods
    registerType_test();
    isRegistered_test();
    getRegisteredType_test();
    checkType_test();

    // public methods
    registerString_test();
    registerInteger_test();
    registerFloat_test();
    registerBoolean_test();
    registerStringArray_test();
    getString_test();
    getInteger_test();
    getFloat_test();
    getBoolean_test();
    getStringArray_test();

    cleanupTestCase();
}

/**
 * initTestCase
 */
void
ConfigHandler_Test::initTestCase()
{
    ErrorContainer error;
    Hanami::writeFile(m_testFilePath, getTestString(), error, true);
}

/**
 * @brief readConfig_test
 */
void
ConfigHandler_Test::readConfig_test()
{
    ConfigHandler configHandler;
    ErrorContainer error;

    TEST_EQUAL(configHandler.initConfig("/tmp/asönganergupuneruigndf.ini", error), false);
    TEST_EQUAL(configHandler.initConfig(m_testFilePath, error), true);
}

/**
 * @brief registerType_test
 */
void
ConfigHandler_Test::registerType_test()
{
    ConfigHandler configHandler;
    Hanami::ErrorContainer error;
    std::string groupName = "groupName";

    TEST_EQUAL(configHandler.registerValue(groupName, "key1", "", ConfigHandler::ConfigType::STRING_TYPE, false, nullptr, error), true);
    TEST_EQUAL(configHandler.registerValue(groupName, "key2", "", ConfigHandler::ConfigType::STRING_TYPE, false, nullptr, error), true);
    TEST_EQUAL(configHandler.registerValue(groupName, "key1", "", ConfigHandler::ConfigType::STRING_TYPE, false, nullptr, error), false);
}

/**
 * @brief isRegistered_test
 */
void
ConfigHandler_Test::isRegistered_test()
{
    ConfigHandler configHandler;
    Hanami::ErrorContainer error;
    std::string groupName = "groupName";

    configHandler.registerValue(groupName, "key1", "", ConfigHandler::ConfigType::STRING_TYPE, false, nullptr, error);

    TEST_EQUAL(configHandler.isRegistered("groupName", "key1"), true);
    TEST_EQUAL(configHandler.isRegistered("groupName", "key2"), false);
}

/**
 * @brief getRegisteredType_test
 */
void
ConfigHandler_Test::getRegisteredType_test()
{
    ConfigHandler configHandler;
    Hanami::ErrorContainer error;
    std::string groupName = "groupName";
    DataValue value("");

    configHandler.registerValue(groupName, "key1", "", ConfigHandler::ConfigType::STRING_TYPE, false, &value, error);
    configHandler.registerValue(groupName, "key2", "", ConfigHandler::ConfigType::INT_TYPE, false, &value, error);

    TEST_EQUAL(configHandler.getRegisteredType("groupName", "key1"), ConfigHandler::ConfigType::STRING_TYPE);
    TEST_EQUAL(configHandler.getRegisteredType("groupName", "key2"), ConfigHandler::ConfigType::INT_TYPE);
}

/**
 * @brief checkType_test
 */
void
ConfigHandler_Test::checkType_test()
{
    ConfigHandler configHandler;
    ErrorContainer error;

    configHandler.initConfig(m_testFilePath, error);

    TEST_EQUAL(configHandler.checkType("DEFAULT", "string_val", ConfigHandler::ConfigType::INT_TYPE), false);
    TEST_EQUAL(configHandler.checkType("DEFAULT", "string_val", ConfigHandler::ConfigType::STRING_TYPE), true);
    TEST_EQUAL(configHandler.checkType("asdf", "string_val", ConfigHandler::ConfigType::STRING_TYPE), true);
}

/**
 * @brief registerString_test
 */
void
ConfigHandler_Test::registerString_test()
{
    ConfigHandler configHandler;
    ErrorContainer error;

    configHandler.initConfig(m_testFilePath, error);

    configHandler.registerString("DEFAULT", "int_val", "", error, "default");
    configHandler.registerString("DEFAULT", "itemName", "", error, "default");
    configHandler.registerString("DEFAULT", "string_val", "", error, "default");
    configHandler.registerString("DEFAULT", "string_val", "", error, "default");
}

/**
 * @brief registerInteger_test
 */
void
ConfigHandler_Test::registerInteger_test()
{
    ConfigHandler configHandler;
    ErrorContainer error;

    configHandler.initConfig(m_testFilePath, error);

    configHandler.registerInteger("DEFAULT", "string_val", "", error, 42);
    configHandler.registerInteger("DEFAULT", "itemName", "", error, 42);
    configHandler.registerInteger("DEFAULT", "int_val", "", error, 42);
    configHandler.registerInteger("DEFAULT", "int_val", "", error, 42);
}

/**
 * @brief registerFloat_test
 */
void
ConfigHandler_Test::registerFloat_test()
{
    ConfigHandler configHandler;
    ErrorContainer error;

    configHandler.initConfig(m_testFilePath, error);

    configHandler.registerFloat("DEFAULT", "string_val", "", error, 42.0);
    configHandler.registerFloat("DEFAULT", "itemName", "", error, 42.0);
    configHandler.registerFloat("DEFAULT", "float_val", "", error, 42.0);
    configHandler.registerFloat("DEFAULT", "float_val", "", error, 42.0);
}

/**
 * @brief registerBoolean_test
 */
void
ConfigHandler_Test::registerBoolean_test()
{
    ConfigHandler configHandler;
    ErrorContainer error;

    configHandler.initConfig(m_testFilePath, error);

    configHandler.registerBoolean("DEFAULT", "string_val", "", error, true);
    configHandler.registerBoolean("DEFAULT", "itemName", "", error, true);
    configHandler.registerBoolean("DEFAULT", "bool_value", "", error, true);
    configHandler.registerBoolean("DEFAULT", "bool_value", "", error, true);
}

/**
 * @brief registerStringArray_test
 */
void
ConfigHandler_Test::registerStringArray_test()
{
    ConfigHandler configHandler;
    std::vector<std::string> defaultValue;
    ErrorContainer error;

    configHandler.initConfig(m_testFilePath, error);
    defaultValue.push_back("test");

    configHandler.registerStringArray("DEFAULT", "string_val", "", error, defaultValue);
    configHandler.registerStringArray("DEFAULT", "itemName", "", error, defaultValue);
    configHandler.registerStringArray("DEFAULT", "string_list", "", error, defaultValue);
    configHandler.registerStringArray("DEFAULT", "string_list", "", error, defaultValue);
}

/**
 * @brief getString_test
 */
void
ConfigHandler_Test::getString_test()
{
    ConfigHandler configHandler;
    bool success = false;
    ErrorContainer error;

    TEST_EQUAL(configHandler.registerString("DEFAULT", "string_val", "", error, "xyz"), true);
    TEST_EQUAL(configHandler.initConfig(m_testFilePath, error), true);
    TEST_EQUAL(configHandler.getString("DEFAULT", "string_val", success), "asdf.asdf");
    TEST_EQUAL(success, true);

}

/**
 * @brief getInteger_test
 */
void
ConfigHandler_Test::getInteger_test()
{
    ConfigHandler configHandler;
    bool success = false;
    ErrorContainer error;

    TEST_EQUAL(configHandler.registerInteger("DEFAULT", "int_val", "", error, 42), true);
    TEST_EQUAL(configHandler.initConfig(m_testFilePath, error), true);
    TEST_EQUAL(configHandler.getInteger("DEFAULT", "int_val", success), 2);
    TEST_EQUAL(success, true);
}

/**
 * @brief getFloat_test
 */
void
ConfigHandler_Test::getFloat_test()
{
    ConfigHandler configHandler;
    bool success = false;
    ErrorContainer error;

    TEST_EQUAL(configHandler.registerFloat("DEFAULT", "float_val", "", error, 42.0), true);
    TEST_EQUAL(configHandler.initConfig(m_testFilePath, error), true);
    TEST_EQUAL(configHandler.getFloat("DEFAULT", "float_val", success), 123.0);
    TEST_EQUAL(success, true);
}

/**
 * @brief getBoolean_test
 */
void
ConfigHandler_Test::getBoolean_test()
{
    ConfigHandler configHandler;
    bool success = false;
    ErrorContainer error;

    TEST_EQUAL(configHandler.registerBoolean("DEFAULT", "bool_value", "", error, false), true);
    TEST_EQUAL(configHandler.initConfig(m_testFilePath, error), true);
    TEST_EQUAL(configHandler.getBoolean("DEFAULT", "bool_value", success), true);
    TEST_EQUAL(success, true);
}

/**
 * @brief getStringArray_test
 */
void
ConfigHandler_Test::getStringArray_test()
{
    ConfigHandler configHandler;
    bool success = false;
    std::vector<std::string> defaultValue;
    std::vector<std::string> ret;
    ErrorContainer error;
    defaultValue.push_back("test");

    TEST_EQUAL(configHandler.registerStringArray("DEFAULT", "string_list", "", error, defaultValue), true);
    TEST_EQUAL(configHandler.initConfig(m_testFilePath, error), true);
    ret = configHandler.getStringArray("DEFAULT", "string_list", success);
    TEST_EQUAL(ret.size(), 3);
    TEST_EQUAL(success, true);
}

/**
 * cleanupTestCase
 */
void
ConfigHandler_Test::cleanupTestCase()
{
    ErrorContainer error;
    Hanami::deleteFileOrDir(m_testFilePath, error);
}

/**
 * @brief ConfigHandler_Test::getTestString
 * @return
 */
const std::string
ConfigHandler_Test::getTestString()
{
    const std::string testString(
                "[DEFAULT]\n"
                "string_val = asdf.asdf\n"
                "int_val = 2\n"
                "float_val = 123.0\n"
                "string_list = a,b,c\n"
                "bool_value = true\n"
                "\n");
    return testString;
}

}
