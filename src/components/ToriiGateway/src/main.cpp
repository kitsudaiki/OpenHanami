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
#include <callbacks.h>
#include <torii_root.h>

#include <libKitsunemimiHanamiCommon/generic_main.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <libAzukiHeart/azuki_input.h>

using Kitsunemimi::Hanami::HanamiMessaging;
using Kitsunemimi::Hanami::initMain;
using Kitsunemimi::Sakura::Session;

int main(int argc, char *argv[])
{
    Kitsunemimi::ErrorContainer error;
    if(initMain(argc, argv, "torii", &registerArguments, &registerConfigs, error) == false) {
        return 1;
    }

    // init included components
    Azuki::initAzukiBlossoms();

    // initialize server and connections based on the config-file
    const std::vector<std::string> groupNames = { "misaki", "azuki", "shiori", "kyouko"};
    if(HanamiMessaging::getInstance()->initialize("torii",
                                                  groupNames,
                                                  nullptr,
                                                  &streamDataCallback,
                                                  &genericCallback,
                                                  error,
                                                  true) == false)
    {
        LOG_ERROR(error);
        return 1;
    }

    // init gateway
    ToriiGateway rootObj;
    if(rootObj.init() == false) {
        return 1;
    }

    // sleep forever
    std::this_thread::sleep_until(std::chrono::time_point<std::chrono::system_clock>::max());

    return 0;
}
