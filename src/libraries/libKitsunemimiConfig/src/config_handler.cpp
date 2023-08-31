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

ConfigHandler* ConfigHandler::instance = nullptr;

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
    const bool ret = readFile(fileContent, m_configFilePath, error);
    if(ret == false)
    {
        error.addMeesage("Error while reading config-file \"" + configFilePath + "\"");
        LOG_ERROR(error);
        return false;
    }

    // parse file content
    m_iniItem = new IniItem();
    std::string parseErrorMessage = "";
    const bool result = m_iniItem->parse(fileContent, error);
    if(result == false)
    {
        error.addMeesage("Error while parsing config-file \"" + configFilePath + "\"");
        LOG_ERROR(error);
        return false;
    }

    // check config against the registered entries
    for(auto& [groupName, groupConfig] : m_registeredConfigs)
    {
        for(auto& [itemName, entry] : groupConfig)
        {
            if(checkEntry(groupName, itemName, entry, error) == false)
            {
                error.addMeesage("Error while checking config-file \"" + configFilePath + "\"");
                LOG_ERROR(error);
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief register string config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @prama comment comment for the docu about the config-entry
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if item-name and group-name are already registered, else true
 */
bool
ConfigHandler::registerString(const std::string &groupName,
                              const std::string &itemName,
                              const std::string &comment,
                              ErrorContainer &error,
                              const std::string &defaultValue,
                              const bool required)
{
    DataValue convertedDefault(defaultValue);
    std::string finalGroupName = groupName;
    return registerValue(finalGroupName,
                         itemName,
                         comment,
                         STRING_TYPE,
                         required,
                         &convertedDefault,
                         error);
}

/**
 * @brief register int/long config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @prama comment comment for the docu about the config-entry
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if item-name and group-name are already registered, else true
 */
bool
ConfigHandler::registerInteger(const std::string &groupName,
                               const std::string &itemName,
                               const std::string &comment,
                               ErrorContainer &error,
                               const long defaultValue,
                               const bool required)
{
    DataValue convertedDefault(defaultValue);
    std::string finalGroupName = groupName;
    return registerValue(finalGroupName,
                         itemName,
                         comment,
                         INT_TYPE,
                         required,
                         &convertedDefault,
                         error);
}

/**
 * @brief register float/double config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @prama comment comment for the docu about the config-entry
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if item-name and group-name are already registered, else true
 */
bool
ConfigHandler::registerFloat(const std::string &groupName,
                             const std::string &itemName,
                             const std::string &comment,
                             ErrorContainer &error,
                             const double defaultValue,
                             const bool required)
{
    DataValue convertedDefault(defaultValue);
    std::string finalGroupName = groupName;
    return registerValue(finalGroupName,
                         itemName,
                         comment,
                         FLOAT_TYPE,
                         required,
                         &convertedDefault,
                         error);
}

/**
 * @brief register bool config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @prama comment comment for the docu about the config-entry
 * @param error reference for error-output
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if item-name and group-name are already registered, else true
 */
bool
ConfigHandler::registerBoolean(const std::string &groupName,
                               const std::string &itemName,
                               const std::string &comment,
                               ErrorContainer &error,
                               const bool defaultValue,
                               const bool required)
{
    DataValue convertedDefault(defaultValue);
    std::string finalGroupName = groupName;
    return registerValue(finalGroupName,
                         itemName,
                         comment,
                         BOOL_TYPE,
                         required,
                         &convertedDefault,
                         error);
}

/**
 * @brief register string-array config value
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @prama comment comment for the docu about the config-entry
 * @param defaultValue default value, if nothing was set inside of the config
 * @param required if true, then the value must be in the config-file (default: false)
 *
 * @return false, if item-name and group-name are already registered, else true
 */
bool
ConfigHandler::registerStringArray(const std::string &groupName,
                                   const std::string &itemName,
                                   const std::string &comment,
                                   ErrorContainer &error,
                                   const std::vector<std::string> &defaultValue,
                                   const bool required)
{
    DataArray convertedDefault;
    for(const std::string &value : defaultValue) {
        convertedDefault.append(new DataValue(value));
    }

    std::string finalGroupName = groupName;
    return registerValue(finalGroupName,
                         itemName,
                         comment,
                         STRING_ARRAY_TYPE,
                         required,
                         &convertedDefault,
                         error);
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
    return m_registeredConfigs[groupName][itemName].value->getString();
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
    return m_registeredConfigs[groupName][itemName].value->getLong();
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
    return m_registeredConfigs[groupName][itemName].value->getDouble();
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
    return m_registeredConfigs[groupName][itemName].value->getBool();
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
    DataArray* array = m_registeredConfigs[groupName][itemName].value->toArray();
    for(uint32_t i = 0; i < array->size(); i++)
    {
        result.push_back(array->get(i)->toValue()->getString());
    }

    return result;
}

/**
 * @brief check entry against the read config-file and override the default
 *        value if set by the config
 *
 * @param groupName group-name of the entry to check
 * @param itemName item-name of the entry to check
 * @param entry entry to check
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ConfigHandler::checkEntry(const std::string &groupName,
                          const std::string &itemName,
                          ConfigEntry &entry,
                          ErrorContainer &error)
{
    // check type against config-file
    if(checkType(groupName, itemName, entry.type) == false)
    {
        error.addMeesage("Config registration failed because item has the false value type: \n"
                         "    group: \'" + groupName + "\'\n"
                         "    item: \'" + itemName + "\'");
        return false;
    }

    // check if value is required
    DataItem* currentVal = m_iniItem->get(groupName, itemName);
    if(entry.isRequired
            && currentVal == nullptr)
    {
        error.addMeesage("Config registration failed because required "
                         "value was not set in the config: \n"
                         "    group: \'" + groupName + "\'\n"
                         "    item: \'" + itemName + "\'");
        return false;
    }

    // overwrite the registered default-value with the value of the config
    if(currentVal != nullptr)
    {
        if(entry.value != nullptr) {
            delete entry.value;
        }
        entry.value = currentVal->copy();
    }

    return true;
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
    const auto outerIt = m_registeredConfigs.find(groupName);
    if(outerIt == m_registeredConfigs.end()) {
        return false;
    }

    const auto innerIt = outerIt->second.find(itemName);
    if(innerIt == outerIt->second.end()) {
        return false;
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
    const auto outerIt = m_registeredConfigs.find(groupName);
    if(outerIt != m_registeredConfigs.end())
    {
        const auto innerIt = outerIt->second.find(itemName);
        if(innerIt != outerIt->second.end())
        {
            if(innerIt->second.value == nullptr) {
                return UNDEFINED_TYPE;
            }
            return innerIt->second.type;
        }
    }

    return UNDEFINED_TYPE;
}

/**
 * @brief register single value in the config
 *
 * @param groupName name of the group
 * @param itemName name of the item within the group
 * @prama comment comment for the docu about the config-entry
 * @param type type of the value to register
 * @param required if true, then the value must be in the config-file
 * @param defaultValue default-value of the item, for the case that
 *                     it is not set by the read config-file
 * @param error reference for error-output
 *
 * @return true, if successfull, else false
 */
bool
ConfigHandler::registerValue(std::string &groupName,
                             const std::string &itemName,
                             const std::string &comment,
                             const ConfigType type,
                             const bool required,
                             DataItem* defaultValue,
                             ErrorContainer &error)
{
    // if group-name is empty, then use the default-group
    if(groupName.size() == 0) {
        groupName = "DEFAULT";
    }

    // precheck if already exist
    if(isRegistered(groupName, itemName) == true)
    {
        error.addMeesage("Config registration failed because item is already registered: \n"
                         "    group: \'" + groupName + "\'\n"
                         "    item: \'" + itemName + "\'");
        LOG_ERROR(error);
        return false;
    }

    // add groupName, if not exist
    if(m_registeredConfigs.find(groupName) == m_registeredConfigs.end())
    {
        std::map<std::string, ConfigEntry> newEntry;
        m_registeredConfigs.try_emplace(groupName, newEntry);
    }

    // add new value
    auto outerIt = m_registeredConfigs.find(groupName);
    ConfigEntry entry;
    entry.type = type;
    entry.isRequired = required;
    entry.comment = comment;
    if(defaultValue != nullptr) {
        entry.value = defaultValue->copy();
    }
    outerIt->second.emplace(itemName, entry);

    return true;
}

/**
 * @brief generate markdown-text with all registered config-entries
 *
 * @param docu reference for the output of the final document
 */
void
ConfigHandler::createDocumentation(std::string &docu)
{
    for(auto& [groupName, groupConfig] : m_registeredConfigs)
    {
        docu.append("## " + groupName + "\n\n");
        docu.append("| Item | Description |\n");
        docu.append("| --- | --- |\n");

        for(auto& [itemName, entry] : groupConfig)
        {
            docu.append("| " + itemName + "| ");
            docu.append("**Description**: " + entry.comment + "<br>");
            if(entry.isRequired) {
                docu.append("**Required**: TRUE<br>");
            } else {
                docu.append("**Required**: FALSE<br>");
            }
            if(entry.value != nullptr
                    && entry.isRequired == false)
            {
                docu.append("**Default**: " + entry.value->toString() + "<br>");
            }
            docu.append(" |\n");
        }
        docu.append("\n");
    }
}

}
