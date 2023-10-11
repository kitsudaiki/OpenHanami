/**
 * @file        main.cpp
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

#include <args.h>
#include <config.h>
#include <hanami_sdk/common/websocket_client.h>
#include <rest_api_tests/rest_api_tests.h>

#include <iostream>
#include <thread>

int
main(int argc, char *argv[])
{
    Hanami::ErrorContainer error;
    Hanami::initConsoleLogger(true);

    // create and init argument-parser
    Hanami::ArgParser argParser;
    registerArguments(&argParser);

    // parse cli-input
    if (argParser.parse(argc, argv, error) == false) {
        LOG_ERROR(error);
        return 1;
    }

    // init and check config-file
    std::string configPath = argParser.getStringValue("config");
    if (configPath == "") {
        configPath = "/etc/hanami/hanami_testing.conf";
    }
    registerConfigs();
    if (INIT_CONFIG(configPath, error) == false) {
        LOG_ERROR(error);
        return 1;
    }

    // get config-parameter for logger
    bool success = false;
    const bool enableDebug = GET_BOOL_CONFIG("DEFAULT", "debug", success);
    if (success == false) {
        return 1;
    }

    const std::string logPath = GET_STRING_CONFIG("DEFAULT", "log_path", success);
    if (success == false) {
        return 1;
    }

    // init logger
    Hanami::initConsoleLogger(enableDebug);
    Hanami::initFileLogger(logPath, "hanami_testing", enableDebug);

    runRestApiTests();

    return 0;
}
