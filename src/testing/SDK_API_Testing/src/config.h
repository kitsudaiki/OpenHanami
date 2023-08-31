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
#include <libKitsunemimiCommon/logger.h>

/**
 * @brief register configs
 */
void
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

    REGISTER_STRING_CONFIG( "connection",
                            "host",
                            "IP-address where the hanami-instance is listening.",
                            error,
                            "",
                            true);
    REGISTER_INT_CONFIG(    "connection",
                            "port",
                            "Port where the hanami-instance is listening.",
                            error,
                            0,
                            true);
    REGISTER_STRING_CONFIG( "connection",
                            "test_user",
                            "Login-name of the user, which is used for testing.",
                            error,
                            "",
                            true);
    REGISTER_STRING_CONFIG( "connection",
                            "test_pw",
                            "Passphrase of the user, which is used for testing.",
                            error,
                            "",
                            true);

    const std::string testDataGroup = "test_data";
    REGISTER_STRING_CONFIG( testDataGroup,
                            "type",
                            "Type of the test ('mnist' or 'csv'). "
                                "IMPORTANT: only the mnist-input is supported at the moment.",
                            error,
                            "mnist",
                            false);
    REGISTER_STRING_CONFIG( testDataGroup,
                            "train_inputs",
                            "Local path to the file with the mnist train inputs.",
                            error,
                            "",
                            true);
    REGISTER_STRING_CONFIG( testDataGroup,
                            "train_labels",
                            "Local path to the file with the mnist train lables.",
                            error,
                            "",
                            true);
    REGISTER_STRING_CONFIG( testDataGroup,
                            "request_inputs",
                            "Local path to the file with the mnist request inputs.",
                            error,
                            "",
                            true);
    REGISTER_STRING_CONFIG( testDataGroup,
                            "request_labels",
                            "Local path to the file with the mnist request labels.",
                            error,
                            "",
                            true);
    /*REGISTER_STRING_CONFIG( testDataGroup,
                            "base_inputs",
                            "Local path to the file with the csv-data.",
                            error,
                            "",
                            true);*/
}

#endif // TSUGUMITESTER_CONFIG_H
