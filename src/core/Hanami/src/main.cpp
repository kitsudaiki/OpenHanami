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

#include <thread>

#include <common.h>

#include <hanami_root.h>
#include <args.h>
#include <config.h>
#include <core/callbacks.h>
#include <callbacks.h>

#include <api/v1/blossom_initializing.h>

#include <libKitsunemimiArgs/arg_parser.h>
#include <libKitsunemimiCommon/logger.h>

int
main(int argc, char *argv[])
{
    Kitsunemimi::ErrorContainer error;
    HanamiRoot rootObj;

    Kitsunemimi::initConsoleLogger(true);

    // create and init argument-parser
    Kitsunemimi::ArgParser argParser;
    registerArguments(&argParser, error);

    // parse cli-input
    if(argParser.parse(argc, argv, error) == false)
    {
        LOG_ERROR(error);
        return 1;
    }

    // init and check config-file
    std::string configPath = argParser.getStringValue("config");
    if(configPath == "") {
        configPath = "/etc/hanami/hanami.conf";
    }
    if(Kitsunemimi::initConfig(configPath, error) == false)
    {
        LOG_ERROR(error);
        return 1;
    }
    registerConfigs(error);
    if(Kitsunemimi::isConfigValid() == false) {
        return 1;
    }

    // get config-parameter for logger
    bool success = false;
    const bool enableDebug = GET_BOOL_CONFIG("DEFAULT", "debug", success);
    if(success == false) {
        return 1;
    }

    const std::string logPath = GET_STRING_CONFIG("DEFAULT", "log_path", success);
    if(success == false) {
        return 1;
    }

    // init logger
    Kitsunemimi::initConsoleLogger(enableDebug);
    Kitsunemimi::initFileLogger(logPath, "hanami", enableDebug);

    // init blossoms
    if(rootObj.init(error) == false)
    {
        LOG_ERROR(error);
        return 1;
    }

    initBlossoms();

    rootObj.initThreads();

    // sleep forever
    std::this_thread::sleep_until(std::chrono::time_point<std::chrono::system_clock>::max());

    return 0;
}
