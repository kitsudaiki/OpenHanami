/**
 * @file        temperature_production.cpp
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

#include "temperature_production.h"

#include <hanami_hardware/temperature_measuring.h>
#include <hanami_root.h>

ThermalProduction::ThermalProduction() : Blossom("Request the temperature-measurement of the CPU")
{
    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("temperature", SAKURA_MAP_TYPE)
        .setComment("Json-object with temperature-measurements");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
ThermalProduction::runTask(BlossomIO& blossomIO,
                           const json&,
                           BlossomStatus&,
                           Hanami::ErrorContainer&)
{
    blossomIO.output["temperature"] = TemperatureMeasuring::getInstance()->getJson();

    return true;
}
