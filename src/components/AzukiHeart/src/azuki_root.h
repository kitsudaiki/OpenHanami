/**
 * @file        azuki_root.h
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

#ifndef AZUKIHEART_AZUKIROOT_H
#define AZUKIHEART_AZUKIROOT_H

#include <string>

#include <libKitsunemimiCommon/logger.h>

class ThreadBinder;
class SpeedMeasuring;
class PowerMeasuring;
class TemperatureMeasuring;

namespace Kitsunemimi::Sakura {
class Host;
}

class AzukiRoot
{
public:
    AzukiRoot();

    bool init();

    static std::string* componentToken;
    static ThreadBinder* threadBinder;
    static SpeedMeasuring* speedMeasuring;
    static PowerMeasuring* powerMeasuring;
    static TemperatureMeasuring* temperatureMeasuring;
    static Kitsunemimi::Sakura::Host* host;
};

#endif // AZUKIHEART_AZUKIROOT_H
