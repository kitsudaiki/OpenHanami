/**
 * @file        cpu_package.h
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

#ifndef KITSUNEMIMI_SAKURA_HARDWARE_CPUPACKAGE_H
#define KITSUNEMIMI_SAKURA_HARDWARE_CPUPACKAGE_H

#include <string>
#include <iostream>
#include <vector>

#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi {
class DataMap;
}

namespace Kitsunemimi::Sakura
{
class CpuCore;

class CpuPackage
{
public:
    CpuPackage(const uint32_t packageId);
    ~CpuPackage();

    const uint32_t packageId;

    CpuCore* getCore(const uint32_t coreId) const;
    CpuCore* addCore(const uint32_t coreId);

    double getThermalSpec() const;
    double getTotalPackagePower();

    const std::string toJsonString();
    Kitsunemimi::DataMap* toJson();

    std::vector<CpuCore*> cpuCores;
};

}

#endif // KITSUNEMIMI_SAKURA_HARDWARE_CPUPACKAGE_H
