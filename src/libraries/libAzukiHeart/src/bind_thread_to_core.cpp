/**
 * @file        bind_thread_to_core.cpp
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

#include "bind_thread_to_core.h"

#include <libKitsunemimiCommon/threading/thread.h>
#include <libKitsunemimiCommon/threading/thread_handler.h>

#include <libKitsunemimiHanamiCommon/enums.h>

using namespace Kitsunemimi;

namespace Azuki
{

BindThreadToCore::BindThreadToCore()
    : Hanami::Blossom("Bind threads of a specific thead-type-name to a specific core.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("thread_name",
                       Hanami::SAKURA_STRING_TYPE,
                       true,
                       "Thread-type-name of the threads, which should be bound to the core.");
    assert(addFieldBorder("thread_name", 3, 256));
    assert(addFieldRegex("thread_name", "[a-zA-Z][a-zA-Z_0-9\\-]*"));

    registerInputField("core_ids",
                       Hanami::SAKURA_ARRAY_TYPE,
                       true,
                       "Core-ids to bind to.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
BindThreadToCore::runTask(Hanami::BlossomIO &blossomIO,
                          const DataMap &,
                          Hanami::BlossomStatus &status,
                          ErrorContainer &error)
{
    const std::string threadName = blossomIO.input.get("thread_name").getString();
    const DataArray* coreIdsArray = blossomIO.input.get("core_ids").getItemContent()->toArray();

    // convert core-ids
    std::vector<uint64_t> coreIds;
    for(uint64_t i = 0; i < coreIdsArray->size(); i++) {
        coreIds.push_back(coreIdsArray->get(i)->toValue()->getLong());
    }

    ThreadHandler* threadHandler = ThreadHandler::getInstance();

    // get threads
    const std::vector<Thread*> threads = threadHandler->getThreads(threadName);
    if(threads.size() == 0)
    {
        status.statusCode = Hanami::NOT_FOUND_RTYPE;
        status.errorMessage = "No threads found for thread-name '" + threadName + "'";
        error.addMeesage(status.errorMessage);
        return false;
    }

    // bind threads to core-ids
    for(Thread* thread : threads)
    {
        if(thread->bindThreadToCores(coreIds) == false)
        {
            status.statusCode = Hanami::BAD_REQUEST_RTYPE;
            status.errorMessage = "Core-id '" + coreIdsArray->toString() + "' is out of range.";
            error.addMeesage(status.errorMessage);
            return false;
        }
    }

    return true;
}

}  // namespace Azuki
