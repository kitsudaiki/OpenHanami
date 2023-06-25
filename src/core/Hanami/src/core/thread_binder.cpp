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

#include <libKitsunemimiSakuraHardware/host.h>
#include <libKitsunemimiSakuraHardware/cpu_core.h>
#include <libKitsunemimiSakuraHardware/cpu_package.h>
#include <libKitsunemimiSakuraHardware/cpu_thread.h>



#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/threading/thread.h>
#include <libKitsunemimiCommon/threading/thread_handler.h>

ThreadBinder* ThreadBinder::instance = nullptr;

ThreadBinder::ThreadBinder()
    : Kitsunemimi::Thread("Azuki_ThreadBinder")
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
 * @brief change core-ids of the threads of azuki itself
 *
 * @param threadNames name of the thread-type
 * @param coreId is of the core (physical thread) to bind to
 *
 * @return true, if successful, false if core-id is out-of-range
 */
bool
ThreadBinder::changeInternalCoreIds(const std::vector<std::string> &threadNames,
                                    const std::vector<uint64_t> coreIds)
{
    Kitsunemimi::ThreadHandler* threadHandler = Kitsunemimi::ThreadHandler::getInstance();
    for(const std::string &name : threadNames)
    {
        const std::vector<Kitsunemimi::Thread*> threads = threadHandler->getThreads(name);
        for(Kitsunemimi::Thread* thread : threads)
        {
            if(thread->bindThreadToCores(coreIds) == false) {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief request the thread-mapping of azuki itself
 *
 * @param completeMap pointer for the result to attach the thread-mapping of azuki
 *
 * @return always true
 */
bool
ThreadBinder::makeInternalRequest(Kitsunemimi::DataMap* completeMap,
                                  Kitsunemimi::ErrorContainer &)
{
    Kitsunemimi::ThreadHandler* threadHandler = Kitsunemimi::ThreadHandler::getInstance();
    const std::vector<std::string> names = threadHandler->getRegisteredNames();
    Kitsunemimi::DataMap* result = new Kitsunemimi::DataMap();

    for(const std::string &name : names)
    {
        const std::vector<Kitsunemimi::Thread*> threads = threadHandler->getThreads(name);
        Kitsunemimi::DataArray* threadArray = new Kitsunemimi::DataArray();
        for(Kitsunemimi::Thread* thread : threads)
        {
            const std::vector<uint64_t> coreIds = thread->getCoreIds();
            Kitsunemimi::DataArray* cores = new Kitsunemimi::DataArray();
            for(const uint64_t coreId : coreIds) {
                cores->append(new Kitsunemimi::DataValue(static_cast<long>(coreId)));
            }
            threadArray->append(cores);
        }
        result->insert(name, threadArray);
    }

    completeMap->insert("azuki", result);

    return true;
}

/**
 * @brief fill lists with ids for the binding
 *
 * @param controlCoreIds reference to the list for all ids of control-processes
 * @param processingCoreIds reference to the list for all ids of processing-processes
 *
 * @return false, if a list is empty, else true
 */
bool
ThreadBinder::fillCoreIds(std::vector<uint64_t> &controlCoreIds,
                          std::vector<uint64_t> &processingCoreIds)
{
    Kitsunemimi::Sakura::CpuCore* phyCore = nullptr;
    Kitsunemimi::Sakura::Host* host = Kitsunemimi::Sakura::Host::getInstance();

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

    if(controlCoreIds.size() == 0) {
        // TODO: error
        return false;
    }

    if(processingCoreIds.size() == 0) {
        // TODO: error
        return false;
    }

    return true;
}

/**
 * @brief ThreadBinder::run
 */
void
ThreadBinder::run()
{
    if(fillCoreIds(m_controlCoreIds, m_processingCoreIds) == false) {
        return;
    }

    while(m_abort == false)
    {
        m_mapLock.lock();

        Kitsunemimi::ErrorContainer error;

        // get thread-mapping of all components
        Kitsunemimi::DataMap newMapping;

        const std::string newMappingStr = newMapping.toString();
        if(m_lastMapping != newMappingStr)
        {
            m_completeMap = newMapping;
            // debug-output
            //std::cout<<"#############################################################"<<std::endl;
            //std::cout<<newMapping.toString(true)<<std::endl;
            //std::cout<<"#############################################################"<<std::endl;
            LOG_DEBUG(newMapping.toString(true));

            // update thread-binding for all components
            for(auto const& [name, value] : newMapping.map)
            {
                const std::vector<std::string> threadNames = value->toMap()->getKeys();
                if(name == "azuki") {
                    changeInternalCoreIds(threadNames, m_controlCoreIds);
                }
            }

            m_lastMapping = newMappingStr;
        }

        m_mapLock.unlock();

        sleep(10);
    }
}

/**
 * @brief convert list to a string
 *
 * @param coreIds list with ids
 *
 * @return comma-separated string with the input-values
 */
const std::string
ThreadBinder::convertCoreIdList(const std::vector<uint64_t> coreIds)
{
    std::string result = "";
    for(uint64_t i = 0; i < coreIds.size(); i++)
    {
        if(i != 0) {
            result.append(",");
        }
        result.append(std::to_string(coreIds[i]));
    }

    return result;
}
