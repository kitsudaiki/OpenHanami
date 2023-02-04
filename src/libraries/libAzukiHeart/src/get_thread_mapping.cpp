/**
 * @file        get_thread_mapping.cpp
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

#include <libKitsunemimiCommon/threading/thread.h>
#include <libKitsunemimiCommon/threading/thread_handler.h>

using namespace Kitsunemimi;

namespace Azuki
{

GetThreadMapping::GetThreadMapping()
    : Hanami::Blossom("Collect all thread-names with its acutal mapped core-id's")
{
    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("thread_map",
                        Hanami::SAKURA_MAP_TYPE,
                        "Map with all thread-names and its core-id as json-string.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetThreadMapping::runTask(Hanami::BlossomIO &blossomIO,
                          const DataMap &,
                          Hanami::BlossomStatus &,
                          ErrorContainer &)
{
    ThreadHandler* threadHandler = ThreadHandler::getInstance();

    const std::vector<std::string> names = threadHandler->getRegisteredNames();

    DataMap* result = new DataMap();

    for(const std::string &name : names)
    {
        const std::vector<Thread*> threads = threadHandler->getThreads(name);
        DataArray* threadArray = new DataArray();
        for(Thread* thread : threads)
        {
            const std::vector<uint64_t> coreIds = thread->getCoreIds();
            DataArray* cores = new DataArray();
            for(const uint64_t coreId : coreIds) {
                cores->append(new DataValue(static_cast<long>(coreId)));
            }
            threadArray->append(cores);
        }
        result->insert(name, threadArray);
    }

    blossomIO.output.insert("thread_map", result);

    return true;
}

}  // namespace Azuki
