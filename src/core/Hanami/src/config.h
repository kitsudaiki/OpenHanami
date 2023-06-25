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

#ifndef HANAMI_CONFIG_H
#define HANAMI_CONFIG_H

#include <libKitsunemimiConfig/config_handler.h>

/**
 * @brief define all available entries in the config file with default-values
 */
inline void
registerConfigs(Kitsunemimi::ErrorContainer &error)
{
    // DEFAULT-section
    REGISTER_BOOL_CONFIG(   "DEFAULT", "debug",    error, false,      false);
    REGISTER_STRING_CONFIG( "DEFAULT", "log_path", error, "/var/log", false);
    REGISTER_STRING_CONFIG( "DEFAULT", "database", error, "",         false);

    // storage-section
    REGISTER_STRING_CONFIG( "storage", "data_set_location",         error, "", true );
    REGISTER_STRING_CONFIG( "storage", "cluster_snapshot_location", error, "", true );

    // auth-section
    REGISTER_STRING_CONFIG( "auth", "token_key_path",    error, "",   true );
    REGISTER_INT_CONFIG(    "auth", "token_expire_time", error, 3600, true );
    REGISTER_STRING_CONFIG( "auth", "policies",          error, "",   true );

    // http-section
    const std::string httpGroup = "http";
    REGISTER_BOOL_CONFIG(   httpGroup, "enable",            error, false);
    REGISTER_BOOL_CONFIG(   httpGroup, "enable_dashboard",  error, false);
    REGISTER_STRING_CONFIG( httpGroup, "dashboard_files",   error, "");
    REGISTER_STRING_CONFIG( httpGroup, "certificate",       error, "",        true);
    REGISTER_STRING_CONFIG( httpGroup, "key",               error, "",        true);
    REGISTER_STRING_CONFIG( httpGroup, "ip",                error, "0.0.0.0", true);
    REGISTER_INT_CONFIG(    httpGroup, "port",              error, 12345,     true);
    REGISTER_INT_CONFIG(    httpGroup, "number_of_threads", error, 4);
}

#endif // HANAMI_CONFIG_H
