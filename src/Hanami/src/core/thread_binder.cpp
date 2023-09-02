/**
 * @file        thread_binder.cpp
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

#include "thread_binder.h"
#include <hanami_root.h>

#include <hanami_hardware/host.h>
#include <hanami_hardware/cpu_core.h>
#include <hanami_hardware/cpu_package.h>
#include <hanami_hardware/cpu_thread.h>

#include <hanami_json/json_item.h>
#include <hanami_common/threading/thread.h>
#include <hanami_common/threading/thread_handler.h>

ThreadBinder* ThreadBinder::instance = nullptr;

ThreadBinder::ThreadBinder()
    : Kitsunemimi::Thread("ThreadBinder")
{
}

/**
 * @brief ThreadBinder::getMappingString
 * @return
 */
Kitsunemimi::DataMap*
ThreadBinder::getMapping()
{
    std::lock_guard<std::mutex> guard(m_mapLock);
    return m_completeMap.copy()->toMap();
}

/**
 * @brief fill lists with ids for the binding
 *
 * @param controlCoreIds reference to the list for all ids of control-processes
 * @param processingCoreIds reference to the list for all ids of processing-processes
 * @param error reference for error-output
 *
 * @return false, if a list is empty, else true
 */
bool
ThreadBinder::fillCoreIds(std::vector<uint64_t> &controlCoreIds,
                          std::vector<uint64_t> &processingCoreIds,
                          Kitsunemimi::ErrorContainer &error)
{
    Kitsunemimi::Sakura::CpuCore* phyCore = nullptr;
    Kitsunemimi::Sakura::Host* host = Kitsunemimi::Sakura::Host::getInstance();

    if(host->cpuPackages.size() == 0)
    {
        error.addMeesage("Failed to read number of cpu-packages from host.");
        return false;
    }

    if(host->cpuPackages[0]->cpuCores.size() == 0)
    {
        error.addMeesage("Failed to read number of cpu-cores from host.");
        return false;
    }

    // control-cores
    phyCore = host->cpuPackages[0]->cpuCores[0];
    for(Kitsunemimi::Sakura::CpuThread* singleThread : phyCore->cpuThreads) {
        controlCoreIds.push_back(singleThread->threadId);
    }

    // processing-cores
    for(uint64_t i = 1; i < host->cpuPackages[0]->cpuCores.size(); i++)
    {
        phyCore = host->cpuPackages[0]->cpuCores[i];
        for(Kitsunemimi::Sakura::CpuThread* singleThread : phyCore->cpuThreads) {
            processingCoreIds.push_back(singleThread->threadId);
        }
    }

    return true;
}

/**
 * @brief ThreadBinder::run
 */
void
ThreadBinder::run()
{
    Kitsunemimi::ErrorContainer error;
    if(fillCoreIds(m_controlCoreIds, m_processingCoreIds, error) == false)
    {
        error.addMeesage("Failed to initialize cpu-thread-lists for thread-binder.");
        LOG_ERROR(error);
        return;
    }

    Kitsunemimi::ThreadHandler* threadHandler = Kitsunemimi::ThreadHandler::getInstance();

    while(m_abort == false)
    {
        sleep(10);

        m_mapLock.lock();

        m_completeMap.clear();
        const std::vector<std::string> threadNames = threadHandler->getRegisteredNames();

        Kitsunemimi::ErrorContainer error;

        for(const std::string &name : threadNames)
        {
            // update thread-binding
            const std::vector<Kitsunemimi::Thread*> threads = threadHandler->getThreads(name);
            for(Kitsunemimi::Thread* thread : threads)
            {
                if(name == "CpuProcessingUnit")
                {
                    if(thread->bindThreadToCores(m_processingCoreIds) == false) {
                        continue;
                    }
                }
                else
                {
                    if(thread->bindThreadToCores(m_controlCoreIds) == false) {
                        continue;
                    }
                }
            }

            // update list for output
            Kitsunemimi::DataArray* idList = new Kitsunemimi::DataArray();
            if(name == "CpuProcessingUnit")
            {
                for(const uint64_t id : m_processingCoreIds) {
                    idList->append(new Kitsunemimi::DataValue((long)id));
                }
            }
            else
            {
                for(const uint64_t id : m_controlCoreIds) {
                    idList->append(new Kitsunemimi::DataValue((long)id));
                }
            }
            m_completeMap.insert(name, idList, true);
        }

        // debug-output
        // std::cout<<"#############################################################"<<std::endl;
        // std::cout<<m_completeMap.toString(true)<<std::endl;
        // std::cout<<"#############################################################"<<std::endl;
        //LOG_DEBUG(newMapping.toString(true));

        m_mapLock.unlock();
    }
}
