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

#include <kyouko_root.h>
#include <args.h>
#include <config.h>
#include <core/callbacks.h>
#include <callbacks.h>

#include <libAzukiHeart/azuki_send.h>
#include <libMisakiGuard/misaki_input.h>

#include <api/blossom_initializing.h>

#include <libKitsunemimiArgs/arg_parser.h>
#include <libKitsunemimiCommon/logger.h>

#include <libKitsunemimiHanamiCommon/generic_main.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <libAzukiHeart/azuki_input.h>
#include <libMisakiGuard/misaki_input.h>

using Kitsunemimi::Hanami::HanamiMessaging;
using Kitsunemimi::Hanami::initMain;

int
main(int argc, char *argv[])
{
    Kitsunemimi::ErrorContainer error;
    KyoukoRoot rootObj;

    if(initMain(argc, argv, "kyouko", &registerArguments, &registerConfigs, error) == false) {
        return 1;
    }

    initBlossoms();

    // init blossoms
    if(rootObj.init(error) == false)
    {
        LOG_ERROR(error);
        return 1;
    }

    // init included components
    Azuki::initAzukiBlossoms();
    Misaki::initMisakiBlossoms();

    rootObj.initThreads();

    // initialize server and connections based on the config-file
    const std::vector<std::string> groupNames = {"misaki", "shiori", "azuki"};
    if(HanamiMessaging::getInstance()->initialize("kyouko",
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

    Azuki::setSpeedToMinimum(error);

    // init internal token for access to other components
    std::string token = "";
    if(Misaki::getInternalToken(token, "kyouko", error) == false)
    {
        error.addMeesage("Failed to get internal token");
        LOG_ERROR(error);
        return 1;
    }
    KyoukoRoot::componentToken = new std::string();
    *KyoukoRoot::componentToken = token;

    // sleep forever
    std::this_thread::sleep_until(std::chrono::time_point<std::chrono::system_clock>::max());

    return 0;
}
