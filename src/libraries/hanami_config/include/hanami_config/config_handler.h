/**
 *  @file       config_handler.h
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

#ifndef CONFIG_HANDLER_H
#define CONFIG_HANDLER_H

#include <iostream>
#include <vector>
#include <map>
#include <hanami_common/logger.h>

#define INIT_CONFIG Hanami::ConfigHandler::getInstance()->initConfig

#define REGISTER_STRING_CONFIG Hanami::ConfigHandler::getInstance()->registerString
#define REGISTER_INT_CONFIG Hanami::ConfigHandler::getInstance()->registerInteger
#define REGISTER_FLOAT_CONFIG Hanami::ConfigHandler::getInstance()->registerFloat
#define REGISTER_BOOL_CONFIG Hanami::ConfigHandler::getInstance()->registerBoolean
#define REGISTER_STRING_ARRAY_CONFIG Hanami::ConfigHandler::getInstance()->registerStringArray

#define GET_STRING_CONFIG Hanami::ConfigHandler::getInstance()->getString
#define GET_INT_CONFIG Hanami::ConfigHandler::getInstance()->getInteger
#define GET_FLOAT_CONFIG Hanami::ConfigHandler::getInstance()->getFloat
#define GET_BOOL_CONFIG Hanami::ConfigHandler::getInstance()->getBoolean
#define GET_STRING_ARRAY_CONFIG Hanami::ConfigHandler::getInstance()->getStringArray

namespace Hanami
{
class DataItem;
class IniItem;

class ConfigHandler_Test;

//==================================================================================================

class ConfigHandler
{
public:
    static ConfigHandler* getInstance()
    {
        if(instance == nullptr) {
            instance = new ConfigHandler();
        }
        return instance;
    }

    bool initConfig(const std::string &configFilePath,
                    ErrorContainer &error);
    void createDocumentation(std::string &docu);

    // register config-options
    bool registerString(const std::string &groupName,
                        const std::string &itemName,
                        const std::string &comment,
                        ErrorContainer &error,
                        const std::string &defaultValue = "",
                        const bool required = false);
    bool registerInteger(const std::string &groupName,
                         const std::string &itemName,
                         const std::string &comment,
                         ErrorContainer &error,
                         const long defaultValue = 0,
                         const bool required = false);
    bool registerFloat(const std::string &groupName,
                       const std::string &itemName,
                       const std::string &comment,
                       ErrorContainer &error,
                       const double defaultValue = 0.0,
                       const bool required = false);
    bool registerBoolean(const std::string &groupName,
                         const std::string &itemName,
                         const std::string &comment,
                         ErrorContainer &error,
                         const bool defaultValue = false,
                         const bool required = false);
    bool registerStringArray(const std::string &groupName,
                             const std::string &itemName,
                             const std::string &comment,
                             ErrorContainer &error,
                             const std::vector<std::string> &defaultValue = {},
                             const bool required = false);

    // getter
    const std::string getString(const std::string &groupName,
                                const std::string &itemName,
                                bool &success);
    long getInteger(const std::string &groupName,
                    const std::string &itemName,
                    bool &success);
    double getFloat(const std::string &groupName,
                    const std::string &itemName,
                    bool &success);
    bool getBoolean(const std::string &groupName,
                    const std::string &itemName,
                    bool &success);
    const std::vector<std::string> getStringArray(const std::string &groupName,
                                                  const std::string &itemName,
                                                  bool &success);

    static Hanami::ConfigHandler* m_config;

private:
    friend ConfigHandler_Test;

    ConfigHandler();
    ~ConfigHandler();
    static ConfigHandler* instance;

    enum ConfigType
    {
        UNDEFINED_TYPE,
        STRING_TYPE,
        INT_TYPE,
        FLOAT_TYPE,
        BOOL_TYPE,
        STRING_ARRAY_TYPE
    };

    struct ConfigEntry
    {
        bool isRequired = false;
        ConfigType type = UNDEFINED_TYPE;
        DataItem* value = nullptr;
        std::string comment = "";
    };

    bool checkEntry(const std::string &groupName,
                    const std::string &itemName,
                    ConfigEntry &entry,
                    ErrorContainer &error);
    bool checkType(const std::string &groupName,
                   const std::string &itemName,
                   const ConfigType type);
    bool isRegistered(const std::string &groupName,
                      const std::string &itemName);

    ConfigType getRegisteredType(const std::string &groupName,
                                 const std::string &itemName);

    bool registerValue(std::string &groupName,
                       const std::string &itemName,
                       const std::string &comment,
                       const ConfigType type,
                       const bool required,
                       DataItem* defaultValue,
                       ErrorContainer &error);

    std::string m_configFilePath = "";
    IniItem* m_iniItem = nullptr;
    std::map<std::string, std::map<std::string, ConfigEntry>> m_registeredConfigs;
};

}


#endif // CONFIG_HANDLER_H
