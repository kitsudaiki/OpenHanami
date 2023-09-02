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
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

#ifndef HANAMI_CONFIG_H
#define HANAMI_CONFIG_H

#include <hanami_config/config_handler.h>

/**
 * @brief define all available entries in the config file with default-values
 */
inline void
registerConfigs(Kitsunemimi::ErrorContainer &error)
{
    // DEFAULT-section
    REGISTER_BOOL_CONFIG(   "DEFAULT",
                            "debug",
                            "Flag to enable debug-output in logging.",
                            error,
                            false,
                            false);
    REGISTER_STRING_CONFIG( "DEFAULT",
                            "log_path",
                            "Path to the directory, where the log-files should be written into.",
                            error,
                            "/var/log",
                            false);
    REGISTER_STRING_CONFIG( "DEFAULT",
                            "database",
                            "Path to the sqlite3 database-file for all local sql-tables of hanami.",
                            error,
                            "/etc/hanami/hanami_db",
                            false);

    // storage-section
    REGISTER_STRING_CONFIG( "storage",
                            "data_set_location",
                            "Local storage location, where all uploaded data-set should be written into.",
                            error,
                            "/etc/hanami/train_data",
                            false );
    REGISTER_STRING_CONFIG( "storage",
                            "checkpoint_location",
                            "Local storage location, where all uploaded data-set should be written into.",
                            error,
                            "/etc/hanami/cluster_snapshots",
                            false );

    // auth-section
    REGISTER_STRING_CONFIG( "auth",
                            "token_key_path",
                            "Local path to the file with the key for signing and validating the jwt-token.",
                            error,
                            "",
                            true );
    REGISTER_INT_CONFIG(    "auth",
                            "token_expire_time",
                            "Number of seconds, until a jwt-token expired.",
                            error,
                            3600,
                            false );
    REGISTER_STRING_CONFIG( "auth",
                            "policies",
                            "Local path to the file with the endpoint-policies.",
                            error,
                            "/etc/hanami/policies",
                            false );

    // http-section
    const std::string httpGroup = "http";
    REGISTER_BOOL_CONFIG(   httpGroup,
                            "enable",
                            "Flag to enable the http-endpoint.",
                            error,
                            false,
                            false);
    REGISTER_BOOL_CONFIG(   httpGroup,
                            "enable_dashboard",
                            "Flag to enable the dashboard.",
                            error,
                            false,
                            false);
    REGISTER_STRING_CONFIG( httpGroup,
                            "dashboard_files",
                            "Local path to the directory, which contains the files of the dashboard.",
                            error,
                            "",
                            true);
    REGISTER_STRING_CONFIG( httpGroup,
                            "certificate",
                            "Local path to the file with the certificate for the https-connection.",
                            error,
                            "",
                            true);
    REGISTER_STRING_CONFIG( httpGroup,
                            "key",
                            "Local path to the file with the key for the https-connection.",
                            error,
                            "",
                            true);
    REGISTER_STRING_CONFIG( httpGroup,
                            "ip",
                            "IP-address, where the http-server should listen.",
                            error,
                            "0.0.0.0",
                            false);
    REGISTER_INT_CONFIG(    httpGroup,
                            "port",
                            "Port, where the http-server should listen.",
                            error,
                            1337,
                            true);
    REGISTER_INT_CONFIG(    httpGroup,
                            "number_of_threads",
                            "Number of threads in the thread-pool for processing http-requests.",
                            error,
                            4,
                            false);
}

#endif // HANAMI_CONFIG_H
