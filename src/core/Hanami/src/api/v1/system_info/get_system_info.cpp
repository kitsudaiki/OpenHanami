/**
 * @file        create_token.h
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

#include "get_system_info.h"
#include <hanami_root.h>

#include <libKitsunemimiHanamiCommon/enums.h>

#include <libKitsunemimiSakuraHardware/host.h>

using namespace Kitsunemimi::Hanami;

GetSystemInfo::GetSystemInfo()
    : Blossom("Get all available information of the local system.\n"
                                   "    - Topology of the cpu-resources\n"
                                   "    - Speed of the cpu")
{
    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("info",
                        SAKURA_MAP_TYPE,
                        "All available information of the local system.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetSystemInfo::runTask(BlossomIO &blossomIO,
                       const Kitsunemimi::DataMap &,
                       BlossomStatus &,
                       Kitsunemimi::ErrorContainer &)
{
    // creat output
    blossomIO.output.insert("info", HanamiRoot::host->toJson());

    return true;
}
