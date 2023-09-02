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

#ifndef HANAMI_THREADBINDER_H
#define HANAMI_THREADBINDER_H

#include <mutex>

#include <hanami_common/threading/thread.h>
#include <hanami_common/logger.h>
#include <hanami_common/items/data_items.h>

class ThreadBinder
        : public Kitsunemimi::Thread
{
public:
    static ThreadBinder* getInstance()
    {
        if(instance == nullptr) {
            instance = new ThreadBinder();
        }
        return instance;
    }

    Kitsunemimi::DataMap* getMapping();

protected:
    void run();

private:
    ThreadBinder();
    static ThreadBinder* instance;

    bool fillCoreIds(std::vector<uint64_t> &coreIds,
                     std::vector<uint64_t> &processingCoreIds,
                     Kitsunemimi::ErrorContainer &error);

    std::mutex m_mapLock;
    Kitsunemimi::DataMap m_completeMap;

    std::vector<uint64_t> m_controlCoreIds;
    std::vector<uint64_t> m_processingCoreIds;
};

#endif // HANAMI_THREADBINDER_H
