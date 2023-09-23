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

#include <hanami_config/config_handler.h>
#include <hanami_common/logger.h>

/**
 * @brief register configs
 */
void
registerConfigs()
{
    // DEFAULT-section
    const std::string defaultGroup = "DEFAULT";

    REGISTER_BOOL_CONFIG(defaultGroup, "debug")
            .setComment("Flag to enable debug-output in logging.")
            .setDefault(false);

    REGISTER_STRING_CONFIG(defaultGroup, "log_path")
            .setComment("Path to the directory, where the log-files should be written into.")
            .setDefault("/var/log");

    // connection-section
    const std::string connectionGroup = "connection";

    REGISTER_STRING_CONFIG(connectionGroup, "host")
            .setComment("IP-address where the hanami-instance is listening.")
            .setRequired(true);

    REGISTER_INT_CONFIG(connectionGroup, "port")
            .setComment("Port where the hanami-instance is listening.")
            .setRequired(true);

    REGISTER_STRING_CONFIG(connectionGroup, "test_user")
            .setComment("Login-name of the user, which is used for testing.")
            .setRequired(true);

    REGISTER_STRING_CONFIG(connectionGroup, "test_pw")
            .setComment("Passphrase of the user, which is used for testing.")
            .setRequired(true);

    // test_data-section
    const std::string testDataGroup = "test_data";

    REGISTER_STRING_CONFIG( testDataGroup, "type")
            .setComment("Type of the test ('mnist' or 'csv'). "
                        "IMPORTANT: only the mnist-input is supported at the moment.")
            .setDefault("mnist");

    REGISTER_STRING_CONFIG(testDataGroup, "train_inputs")
            .setComment("Local path to the file with the mnist train inputs.")
            .setRequired(true);

    REGISTER_STRING_CONFIG(testDataGroup, "train_labels")
            .setComment("Local path to the file with the mnist train lables.")
            .setRequired(true);

    REGISTER_STRING_CONFIG(testDataGroup, "request_inputs")
            .setComment("Local path to the file with the mnist request inputs.")
            .setRequired(true);

    REGISTER_STRING_CONFIG(testDataGroup, "request_labels")
            .setComment("Local path to the file with the mnist request labels.")
            .setRequired(true);
}

#endif // TSUGUMITESTER_CONFIG_H
