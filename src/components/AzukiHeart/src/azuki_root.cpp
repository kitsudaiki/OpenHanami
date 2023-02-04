/**
 * @file        azuki_root.cpp
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

#include "azuki_root.h"

#include <api/blossom_initializing.h>
#include <core/thread_binder.h>
#include <core/power_measuring.h>
#include <core/speed_measuring.h>
#include <core/temperature_measuring.h>

#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiCommon/files/text_file.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiJwt/jwt.h>

#include <libKitsunemimiHanamiCommon/component_support.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <libKitsunemimiSakuraHardware/host.h>

#include <libMisakiGuard/misaki_input.h>

using namespace Kitsunemimi::Hanami;
using Kitsunemimi::Hanami::SupportedComponents;
using Kitsunemimi::Hanami::HanamiMessaging;

std::string* AzukiRoot::componentToken = nullptr;
ThreadBinder* AzukiRoot::threadBinder = nullptr;
PowerMeasuring* AzukiRoot::powerMeasuring = nullptr;
SpeedMeasuring* AzukiRoot::speedMeasuring = nullptr;
TemperatureMeasuring* AzukiRoot::temperatureMeasuring = nullptr;
Kitsunemimi::Sakura::Host* AzukiRoot::host = nullptr;

/**
 * @brief constructor
 */
AzukiRoot::AzukiRoot() {}

/**
 * @brief init azuki
 *
 * @return true, if successful, else false
 */
bool
AzukiRoot::init()
{
    Kitsunemimi::ErrorContainer error;
    initBlossoms();

    // init internal token for access to other components
    std::string token = "";
    if(Misaki::getInternalToken(token, "azuki", error) == false)
    {
        error.addMeesage("Failed to get internal token");
        LOG_ERROR(error);
        return 1;
    }
    AzukiRoot::componentToken = new std::string();
    *AzukiRoot::componentToken = token;

    // init overview of all resources of the host
    AzukiRoot::host = new Kitsunemimi::Sakura::Host();
    if(AzukiRoot::host->initHost(error) == false)
    {
        error.addMeesage("Failed read resource-information of the local host");
        LOG_ERROR(error);
        return 1;
    }

    // create thread-binder
    AzukiRoot::threadBinder = new ThreadBinder();
    AzukiRoot::threadBinder->startThread();

    // create power-measuring-loop
    AzukiRoot::powerMeasuring = new PowerMeasuring();
    AzukiRoot::powerMeasuring->startThread();

    // create speed-measuring-loop
    AzukiRoot::speedMeasuring = new SpeedMeasuring();
    AzukiRoot::speedMeasuring->startThread();

    // create temperature-measuring-loop
    AzukiRoot::temperatureMeasuring = new TemperatureMeasuring();
    AzukiRoot::temperatureMeasuring->startThread();

    return true;
}
