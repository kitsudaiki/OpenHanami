/**
 *  @file       config_handler.cpp
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

#include <libKitsunemimiConfig/config_handler.h>

#include <libKitsunemimiCommon/items/data_items.h>
#include <libKitsunemimiCommon/files/text_file.h>
#include <libKitsunemimiIni/ini_item.h>

namespace Kitsunemimi
{

ConfigHandler* ConfigHandler::m_config = nullptr;

/**
 * @brief read a ini config-file
 *
 * @param configFilePath absolute path to the config-file to read
 * @param error reference for error-output
 *
 * @return false, if reading or parsing the file failed, else true
 */
bool
initConfig(const std::string &configFilePath,
           ErrorContainer &error)
{
    if(ConfigHandler::m_config != nullptr)
    {
        LOG_WARNING("config is already initialized.");
        return true;
    }

    ConfigHandler::m_config = new ConfigHandler();
    return ConfigHandler::m_config->initConfig(configFilePath, error);
}

/**
 * @brief request if config is valid
 *
 * @return true, if valid, else false
 */
bool
isConfigValid()
{
    if(ConfigHandler::m_config == nullptr) {
        return false;
    }

    return ConfigHandler::m_config->isConfigValid();
}

/**
 * @brief reset configuration (primary to test different configs in one test)
 */
void
resetConfig()
{
    if(ConfigHandler::m_config != nullptr)
    {
        delete ConfigHandler::m_config;
        ConfigHandler::m_config = nullptr;
    }
}

/**
 * @brief register string config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 */
void
registerString(const std::string &groupName,
               const std::string &itemName,
               ErrorContainer &error,
               const std::string &defaultValue,
               const bool required)
{
    if(ConfigHandler::m_config == nullptr) {
        return;
    }

    ConfigHandler::m_config->registerString(groupName, itemName, error, defaultValue, required);
}

/**
 * @brief register int/long config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 */
void
registerInteger(const std::string &groupName,
                const std::string &itemName,
                ErrorContainer &error,
                const long defaultValue,
                const bool required)
{
    if(ConfigHandler::m_config == nullptr) {
        return;
    }

    ConfigHandler::m_config->registerInteger(groupName, itemName, error, defaultValue, required);
}

/**
 * @brief register float/double config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 */
void
registerFloat(const std::string &groupName,
              const std::string &itemName,
              ErrorContainer &error,
              const double defaultValue,
              const bool required)
{
    if(ConfigHandler::m_config == nullptr) {
        return;
    }

    ConfigHandler::m_config->registerFloat(groupName, itemName, error, defaultValue, required);
}

/**
 * @brief register bool config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 */
void
registerBoolean(const std::string &groupName,
                const std::string &itemName,
                ErrorContainer &error,
                const bool defaultValue,
                const bool required)
{
    if(ConfigHandler::m_config == nullptr) {
        return;
    }

    ConfigHandler::m_config->registerBoolean(groupName, itemName, error, defaultValue, required);
}

/**
 * @brief register string-array config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 */
void
registerStringArray(const std::string &groupName,
                    const std::string &itemName,
                    ErrorContainer &error,
                    const std::vector<std::string> &defaultValue,
                    const bool required)
{
    if(ConfigHandler::m_config == nullptr) {
        return;
    }

    ConfigHandler::m_config->registerStringArray(groupName, itemName, error, defaultValue,required);
}

/**
 * @brief get string-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return empty string, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
const std::string
getString(const std::string &groupName,
          const std::string &itemName,
          bool &success)
{
    success = true;

    if(ConfigHandler::m_config == nullptr)
    {
        success = false;
        return "";
    }

    return ConfigHandler::m_config->getString(groupName, itemName, success);
}

/**
 * @brief get long-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return 0, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
long
getInteger(const std::string &groupName,
           const std::string &itemName,
           bool &success)
{
    success = true;

    if(ConfigHandler::m_config == nullptr)
    {
        success = false;
        return 0;
    }

    return ConfigHandler::m_config->getInteger(groupName, itemName, success);
}

/**
 * @brief get double-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return 0.0, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
double
getFloat(const std::string &groupName,
         const std::string &itemName,
         bool &success)
{
    success = true;

    if(ConfigHandler::m_config == nullptr)
    {
        success = false;
        return 0.0;
    }

    return ConfigHandler::m_config->getFloat(groupName, itemName, success);
}

/**
 * @brief get bool-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return false, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
bool
getBoolean(const std::string &groupName,
           const std::string &itemName,
           bool &success)
{
    success = true;

    if(ConfigHandler::m_config == nullptr)
    {
        success = false;
        return false;
    }

    return ConfigHandler::m_config->getBoolean(groupName, itemName, success);
}

/**
 * @brief get string-array-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return empty string-array, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
const std::vector<std::string>
getStringArray(const std::string &groupName,
               const std::string &itemName,
               bool &success)
{
    std::vector<std::string> result;
    success = true;

    if(ConfigHandler::m_config == nullptr)
    {
        success = false;
        return result;
    }

    return ConfigHandler::m_config->getStringArray(groupName, itemName, success);
}

/**
 * @brief ConfigHandler::ConfigHandler
 */
ConfigHandler::ConfigHandler() {}

/**
 * @brief ConfigHandler::~ConfigHandler
 */
ConfigHandler::~ConfigHandler()
{
    delete m_iniItem;
}

/**
 * @brief read a ini config-file
 *
 * @param configFilePath absolute path to the config-file to read
 * @param error reference for error-output
 *
 * @return false, if reading or parsing the file failed, else true
 */
bool
ConfigHandler::initConfig(const std::string &configFilePath,
                          ErrorContainer &error)
{
    // read file
    m_configFilePath = configFilePath;
    std::string fileContent = "";
    bool ret = readFile(fileContent, m_configFilePath, error);
    if(ret == false)
    {
        error.addMeesage("Error while reading config-file \"" + configFilePath + "\"");
        LOG_ERROR(error);
        return false;
    }

    // parse file content
    m_iniItem = new IniItem();
    std::string parseErrorMessage = "";
    bool result = m_iniItem->parse(fileContent, error);
    if(result == false)
    {
        error.addMeesage("Error while parsing config-file \"" + configFilePath + "\"");
        return false;
    }

    return true;
}

/**
 * @brief request if config is valid
 *
 * @return true, if valid, else false
 */
bool
ConfigHandler::isConfigValid() const
{
    return m_configValid;
}

/**
 * @brief register string config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if false type or item-name and group-name are already registered
 */
void
ConfigHandler::registerString(const std::string &groupName,
                              const std::string &itemName,
                              ErrorContainer &error,
                              const std::string &defaultValue,
                              const bool required)
{
    std::string finalGroupName = groupName;
    if(registerValue(finalGroupName, itemName, STRING_TYPE, required, error) == false) {
        return;
    }

    // set default-type, in case the nothing was already set
    m_iniItem->set(finalGroupName, itemName, defaultValue);

    return;
}

/**
 * @brief register int/long config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if false type or item-name and group-name are already registered
 */
void
ConfigHandler::registerInteger(const std::string &groupName,
                               const std::string &itemName,
                               ErrorContainer &error,
                               const long defaultValue,
                               const bool required)
{
    std::string finalGroupName = groupName;
    if(registerValue(finalGroupName, itemName, INT_TYPE, required, error) == false) {
        return;
    }

    // set default-type, in case the nothing was already set
    m_iniItem->set(finalGroupName, itemName, defaultValue);

    return;
}

/**
 * @brief register float/double config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if false type or item-name and group-name are already registered
 */
void
ConfigHandler::registerFloat(const std::string &groupName,
                             const std::string &itemName,
                             ErrorContainer &error,
                             const double defaultValue,
                             const bool required)
{
    std::string finalGroupName = groupName;
    if(registerValue(finalGroupName, itemName, FLOAT_TYPE, required, error) == false) {
        return;
    }

    // set default-type, in case the nothing was already set
    m_iniItem->set(finalGroupName, itemName, defaultValue);

    return;
}

/**
 * @brief register bool config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if false type or item-name and group-name are already registered
 */
void
ConfigHandler::registerBoolean(const std::string &groupName,
                               const std::string &itemName,
                               ErrorContainer &error,
                               const bool defaultValue,
                               const bool required)
{
    std::string finalGroupName = groupName;
    if(registerValue(finalGroupName, itemName, BOOL_TYPE, required, error) == false) {
        return;
    }

    // set default-type, in case the nothing was already set
    m_iniItem->set(finalGroupName, itemName, defaultValue);

    return;
}

/**
 * @brief register string-array config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if false type or item-name and group-name are already registered
 */
void
ConfigHandler::registerStringArray(const std::string &groupName,
                                   const std::string &itemName,
                                   ErrorContainer &error,
                                   const std::vector<std::string> &defaultValue,
                                   const bool required)
{
    std::string finalGroupName = groupName;
    if(registerValue(finalGroupName, itemName, STRING_ARRAY_TYPE, required, error) == false) {
        return;
    }

    // set default-type, in case the nothing was already set
    m_iniItem->set(finalGroupName, itemName, defaultValue);

    return;
}

/**
 * @brief get string-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return empty string, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
const std::string
ConfigHandler::getString(const std::string &groupName,
                         const std::string &itemName,
                         bool &success)
{
    success = true;

    // compare with registered type
    if(getRegisteredType(groupName, itemName) != ConfigType::STRING_TYPE)
    {
        success = false;
        return "";
    }

    // get value from config
    return m_iniItem->get(groupName, itemName)->toValue()->getString();
}

/**
 * @brief get long-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return 0, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
long
ConfigHandler::getInteger(const std::string &groupName,
                          const std::string &itemName,
                          bool &success)
{
    success = true;

    // compare with registered type
    if(getRegisteredType(groupName, itemName) != ConfigType::INT_TYPE)
    {
        success = false;
        return 0l;
    }

    // get value from config
    return m_iniItem->get(groupName, itemName)->toValue()->getLong();
}

/**
 * @brief get double-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return 0.0, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
double
ConfigHandler::getFloat(const std::string &groupName,
                        const std::string &itemName,
                        bool &success)
{
    success = true;

    // compare with registered type
    if(getRegisteredType(groupName, itemName) != ConfigType::FLOAT_TYPE)
    {
        success = false;
        return 0.0;
    }

    // get value from config
    return m_iniItem->get(groupName, itemName)->toValue()->getDouble();
}

/**
 * @brief get bool-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return false, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
bool
ConfigHandler::getBoolean(const std::string &groupName,
                          const std::string &itemName,
                          bool &success)
{
    success = true;

    // compare with registered type
    if(getRegisteredType(groupName, itemName) != ConfigType::BOOL_TYPE)
    {
        success = false;
        return false;
    }

    // get value from config
    return m_iniItem->get(groupName, itemName)->toValue()->getBool();
}

/**
 * @brief get string-array-value from config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param success reference to bool-value with the result. returns false if item-name and group-name
 *                are not registered, else true.
 *
 * @return empty string-array, if item-name and group-name are not registered, else value from the
 *         config-file or the defined default-value.
 */
const std::vector<std::string>
ConfigHandler::getStringArray(const std::string &groupName,
                              const std::string &itemName,
                              bool &success)
{
    std::vector<std::string> result;
    success = true;

    // compare with registered type
    if(getRegisteredType(groupName, itemName) != ConfigType::STRING_ARRAY_TYPE)
    {
        success = false;
        return result;
    }

    // get and transform result from the config-file
    DataArray* array = m_iniItem->get(groupName, itemName)->toArray();
    for(uint32_t i = 0; i < array->size(); i++)
    {
        result.push_back(array->get(i)->toValue()->getString());
    }

    return result;
}

/**
 * @brief check if defined type match with the type of the value within the config-file
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param type type-identifier
 *
 * @return true, if type match with the config-file, else false
 */
bool
ConfigHandler::checkType(const std::string &groupName,
                         const std::string &itemName,
                         const ConfigType type)
{
    // get value from config-file
    DataItem* currentItem = m_iniItem->get(groupName, itemName);

    // precheck
    if(currentItem == nullptr) {
        return true;
    }

    // check for array
    if(currentItem->getType() == DataItem::ARRAY_TYPE
            && type == ConfigType::STRING_ARRAY_TYPE)
    {
        return true;
    }

    // check value
    if(currentItem->getType() == DataItem::VALUE_TYPE)
    {
        DataValue* value = currentItem->toValue();

        // check string
        if(value->getValueType() == DataValue::STRING_TYPE
                && type == ConfigType::STRING_TYPE)
        {
            return true;
        }

        // check integer
        if(value->getValueType() == DataValue::INT_TYPE
                && type == ConfigType::INT_TYPE)
        {
            return true;
        }

        // check float
        if(value->getValueType() == DataValue::FLOAT_TYPE
                && type == ConfigType::FLOAT_TYPE)
        {
            return true;
        }

        // check boolean
        if(value->getValueType() == DataValue::BOOL_TYPE
                && type == ConfigType::BOOL_TYPE)
        {
            return true;
        }
    }

    return false;
}

/**
 * @brief check, if an item-name and group-name are already registered
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 *
 * @return true, if item-name and group-name is already registered, else false
 */
bool
ConfigHandler::isRegistered(const std::string &groupName,
                            const std::string &itemName)
{
    if(getRegisteredType(groupName, itemName) == ConfigType::UNDEFINED_TYPE) {
        return false;
    }

    return true;
}

/**
 * @brief register type
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param type type-identifier to register
 *
 * @return false, if item-name and group-name are already registered, else true
 */
bool
ConfigHandler::registerType(const std::string &groupName,
                            const std::string &itemName,
                            const ConfigType type)
{
    // precheck if already exist
    if(isRegistered(groupName, itemName) == true) {
        return false;
    }

    std::map<std::string, std::map<std::string, ConfigType>>::iterator outerIt;

    // add groupName, if not exist
    outerIt = m_registeredConfigs.find(groupName);
    if(outerIt == m_registeredConfigs.end())
    {
        std::map<std::string, ConfigType> newEntry;
        m_registeredConfigs.insert(
                    std::pair<std::string, std::map<std::string, ConfigType>>(groupName, newEntry));
    }

    // add new value
    outerIt = m_registeredConfigs.find(groupName);
    if(outerIt != m_registeredConfigs.end())
    {
        outerIt->second.insert(std::make_pair(itemName, type));
    }

    return true;
}

/**
 * @brief get registered config-type
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 *
 * @return undefined-type, if item-name and group-name are not registered, else the registered type
 */
ConfigHandler::ConfigType
ConfigHandler::getRegisteredType(const std::string &groupName,
                                 const std::string &itemName)
{
    std::map<std::string, std::map<std::string, ConfigType>>::const_iterator outerIt;
    outerIt = m_registeredConfigs.find(groupName);

    if(outerIt != m_registeredConfigs.end())
    {
        std::map<std::string, ConfigType>::const_iterator innerIt;
        innerIt = outerIt->second.find(itemName);

        if(innerIt != outerIt->second.end())
        {
            return innerIt->second;
        }
    }

    return UNDEFINED_TYPE;
}

/**
 * @brief register single value in the config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @param type type of the value to register
 * @param required if true, then the value must be in the config-file
 * @param error reference for error-output
 *
 * @return true, if successfull, else false
 */
bool
ConfigHandler::registerValue(std::string &groupName,
                             const std::string &itemName,
                             const ConfigType type,
                             const bool required,
                             ErrorContainer &error)
{
    // if group-name is empty, then use the default-group
    if(groupName.size() == 0) {
        groupName = "DEFAULT";
    }

    // check type against config-file
    if(checkType(groupName, itemName, type) == false)
    {
        error.addMeesage("Config registration failed because item has the false value type: \n"
                         "    group: \'" + groupName + "\'\n"
                         "    item: \'" + itemName + "\'");
        LOG_ERROR(error);
        m_configValid = false;
        return false;
    }

    // check if value is required
    if(required
            && m_iniItem->get(groupName, itemName) == nullptr)
    {
        error.addMeesage("Config registration failed because required "
                         "value was not set in the config: \n"
                         "    group: \'" + groupName + "\'\n"
                         "    item: \'" + itemName + "\'");
        LOG_ERROR(error);
        m_configValid = false;
        return false;
    }

    // try to register type
    if(registerType(groupName, itemName, type) == false)
    {
        error.addMeesage("Config registration failed because item is already registered: \n"
                         "    group: \'" + groupName + "\'\n"
                         "    item: \'" + itemName + "\'");
        LOG_ERROR(error);
        m_configValid = false;
        return false;
    }

    return true;
}

}
