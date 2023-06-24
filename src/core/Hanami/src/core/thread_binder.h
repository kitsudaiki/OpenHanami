/**
 * @file        thread_binder.h
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

#ifndef AZUKIHEART_THREADBINDER_H
#define AZUKIHEART_THREADBINDER_H

#include <mutex>

#include <libKitsunemimiCommon/threading/thread.h>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/items/data_items.h>

class ThreadBinder
        : public Kitsunemimi::Thread
{
public:
    ThreadBinder();

    Kitsunemimi::DataMap* getMapping();

protected:
    void run();

private:
    const std::string convertCoreIdList(const std::vector<uint64_t> coreIds);
    bool changeInternalCoreIds(const std::vector<std::string> &threadNames,
                               const std::vector<uint64_t> coreIds);
    bool makeInternalRequest(Kitsunemimi::DataMap *completeMap,
                             Kitsunemimi::ErrorContainer &);

    bool fillCoreIds(std::vector<uint64_t> &coreIds,
                     std::vector<uint64_t> &processingCoreIds);

    std::mutex m_mapLock;
    Kitsunemimi::DataMap m_completeMap;
    std::string m_lastMapping = "";
    std::vector<uint64_t> m_controlCoreIds;
    std::vector<uint64_t> m_processingCoreIds;
};

#endif // AZUKIHEART_THREADBINDER_H
