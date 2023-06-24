/**
 * @file        config.h
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

#ifndef TSUGUMITESTER_CONFIG_H
#define TSUGUMITESTER_CONFIG_H

#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiHanamiCommon/config.h>
#include <libKitsunemimiCommon/logger.h>

/**
 * @brief register configs
 */
void
registerConfigs(Kitsunemimi::ErrorContainer &error)
{
    Kitsunemimi::Hanami::registerBasicConfigs(error);

    REGISTER_STRING_CONFIG( "connection", "host",      error, "", true);
    REGISTER_INT_CONFIG(    "connection", "port",      error, 0,  true);
    REGISTER_STRING_CONFIG( "connection", "test_user", error, "", true);
    REGISTER_STRING_CONFIG( "connection", "test_pw",   error, "", true);

    REGISTER_STRING_CONFIG( "test_data", "type",           error, "", true);
    REGISTER_STRING_CONFIG( "test_data", "learn_inputs",   error, "", true);
    REGISTER_STRING_CONFIG( "test_data", "learn_labels",   error, "", true);
    REGISTER_STRING_CONFIG( "test_data", "request_inputs", error, "", true);
    REGISTER_STRING_CONFIG( "test_data", "request_labels", error, "", true);
    REGISTER_STRING_CONFIG( "test_data", "base_inputs",    error, "", true);
}

#endif // TSUGUMITESTER_CONFIG_H
