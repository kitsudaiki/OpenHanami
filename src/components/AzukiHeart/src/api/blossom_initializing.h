/**
 * @file        blossom_initializing.h
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

#ifndef AZUKIHEART_BLOSSOM_INITIALIZING_H
#define AZUKIHEART_BLOSSOM_INITIALIZING_H

#include <libKitsunemimiCommon/logger.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <api/v1/system_info/get_system_info.h>

#include <api/v1/threading/get_thread_mapping.h>

#include <api/v1/measurements/power_consumption.h>
#include <api/v1/measurements/temperature_production.h>
#include <api/v1/measurements/speed.h>

using Kitsunemimi::Hanami::HanamiMessaging;

void
initBlossoms()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();

    assert(interface->addBlossom("system", "get_info", new GetSystemInfo()));
    assert(interface->addEndpoint("v1/system_info",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  Kitsunemimi::Hanami::BLOSSOM_TYPE,
                                  "system",
                                  "get_info"));

    assert(interface->addBlossom("threading", "get_mapping", new GetThreadMapping()));
    assert(interface->addEndpoint("v1/threading",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  Kitsunemimi::Hanami::BLOSSOM_TYPE,
                                  "threading",
                                  "get_mapping"));

    assert(interface->addBlossom("measurements", "get_power_consumption", new PowerConsumption()));
    assert(interface->addEndpoint("v1/power_consumption",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  Kitsunemimi::Hanami::BLOSSOM_TYPE,
                                  "measurements",
                                  "get_power_consumption"));

    assert(interface->addBlossom("measurements", "get_speed", new Speed()));
    assert(interface->addEndpoint("v1/speed",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  Kitsunemimi::Hanami::BLOSSOM_TYPE,
                                  "measurements",
                                  "get_speed"));

    assert(interface->addBlossom("measurements",
                                 "get_temperature_production",
                                 new ThermalProduction()));
    assert(interface->addEndpoint("v1/temperature_production",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  Kitsunemimi::Hanami::BLOSSOM_TYPE,
                                  "measurements",
                                  "get_temperature_production"));
}

#endif // AZUKIHEART_BLOSSOM_INITIALIZING_H
