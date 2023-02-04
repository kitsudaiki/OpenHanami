/**
 * @file        config.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
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
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{
namespace Hanami
{

/**
 * @brief register basic configs for DEFAULT-seciont
 *
 * @param error reference for error-output
 */
void
registerBasicConfigs(ErrorContainer &error)
{
    REGISTER_BOOL_CONFIG(   "DEFAULT", "debug",    error, false,      false);
    REGISTER_STRING_CONFIG( "DEFAULT", "log_path", error, "/var/log", false);
    REGISTER_STRING_CONFIG( "DEFAULT", "database", error, "",         false);
}

/**
 * @brief register configs to connect to other components
 *
 * @param configGroups list of components to initialize connection to these
 * @param createServer true to spawn a server and not only client-connections
 *
 * @param error reference for error-output
 */
void
registerBasicConnectionConfigs(const std::vector<std::string> &configGroups,
                               const bool createServer,
                               ErrorContainer &error)
{
    if(createServer)
    {
        REGISTER_INT_CONFIG(    "DEFAULT", "port",      error, 0,  false);
        REGISTER_STRING_CONFIG( "DEFAULT", "address",   error, "", true);
    }

    for(const std::string& groupName : configGroups)
    {
        REGISTER_INT_CONFIG(    groupName, "port",      error, 0,  false);
        REGISTER_STRING_CONFIG( groupName, "address",   error, "", false);
    }
}

}  // namespace Hanami
}  // namespace Kitsunemimi
