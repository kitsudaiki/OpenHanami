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

#include <iostream>

#include <config.h>
#include <args.h>
#include <thread>

#include <rest_api_tests/rest_api_tests.h>

#include <libKitsunemimiHanamiCommon/generic_main.h>
#include <libHanamiAiSdk/common/websocket_client.h>

using Kitsunemimi::Hanami::initMain;

int main(int argc, char *argv[])
{
    Kitsunemimi::ErrorContainer error;
    Kitsunemimi::initConsoleLogger(false);
    if(initMain(argc, argv, "tsugumi", &registerArguments, &registerConfigs, error) == false)
    {
        LOG_ERROR(error);
        return 1;
    }

    runRestApiTests();

    return 0;
}
