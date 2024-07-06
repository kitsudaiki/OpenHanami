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

#include "get_thread_mapping.h"

#include <core/thread_binder.h>
#include <hanami_root.h>

GetThreadMappingV1M0::GetThreadMappingV1M0()
    : Blossom(
        "Get Mapping of the all threads of all components "
        "to its bound cpu-core")
{
    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("thread_map", SAKURA_MAP_TYPE)
        .setComment("Map with all thread-names and its core-id as json-string.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetThreadMappingV1M0::runTask(BlossomIO& blossomIO,
                              const json&,
                              BlossomStatus&,
                              Hanami::ErrorContainer&)
{
    blossomIO.output["thread_map"] = ThreadBinder::getInstance()->getMapping();

    return true;
}
