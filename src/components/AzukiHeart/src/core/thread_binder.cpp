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
#include <azuki_root.h>

#include <libKitsunemimiSakuraHardware/host.h>
#include <libKitsunemimiSakuraHardware/cpu_core.h>
#include <libKitsunemimiSakuraHardware/cpu_package.h>
#include <libKitsunemimiSakuraHardware/cpu_thread.h>

#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiCommon/component_support.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>

#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/threading/thread.h>
#include <libKitsunemimiCommon/threading/thread_handler.h>

using namespace Kitsunemimi::Hanami;
using Kitsunemimi::Hanami::HanamiMessagingClient;
using Kitsunemimi::Hanami::HanamiMessaging;
using Kitsunemimi::Hanami::SupportedComponents;

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
 * @brief tell another component a new thread-binding
 *
 * @param component name of the component which has to be modified
 * @param request base-request for the trigger
 * @param threadNames name of the thread-type
 * @param coreId is of the core (physical thread) to bind to
 */
void
ThreadBinder::changeRemoteCoreIds(const std::string &component,
                                  Kitsunemimi::Hanami::RequestMessage &request,
                                  const std::vector<std::string> &threadNames)
{
    for(const std::string &name : threadNames)
    {
        Kitsunemimi::ErrorContainer error;
        Kitsunemimi::Hanami::ResponseMessage response;

        // set values for thread-binding
        request.inputValues = "{ \"token\":\""
                              + *AzukiRoot::componentToken
                              + "\",\"core_ids\":[";

        if(name == "CpuProcessingUnit") {
            request.inputValues.append(convertCoreIdList(m_processingCoreIds));
        } else {
            request.inputValues.append(convertCoreIdList(m_controlCoreIds));
        }

        request.inputValues.append("],\"thread_name\":\"" + name + "\"}");

        // get internal client for interaction with shiori
        HanamiMessagingClient* client = HanamiMessaging::getInstance()->getOutgoingClient(component);
        if(client == nullptr)
        {
            error.addMeesage("Failed to get client to '" + component + "'");
            error.addSolution("Check if '" + component + "' is correctly configured");
            return;
        }

        // trigger remote-action for thread-binding
        if(client->triggerSakuraFile(response, request, error) == false)
        {
            LOG_ERROR(error);
        }

        // check response
        if(response.success == false)
        {
            error.addMeesage(response.responseContent);
            LOG_ERROR(error);
        }
    }
}


/**
 * @brief request thread-mapping of another component
 *
 * @param completeMap pointer for the result to attach the thread-mapping of the requested component
 * @param component name of the component of which the thread-mapping should be requested
 * @param request request for getting the thread-mapping of the remote-component
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ThreadBinder::requestComponent(Kitsunemimi::DataMap* completeMap,
                               const std::string &component,
                               const Kitsunemimi::Hanami::RequestMessage &request,
                               Kitsunemimi::ErrorContainer &error)
{
    // get internal client for interaction with shiori
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->getOutgoingClient(component);
    if(client == nullptr)
    {
        error.addMeesage("Failed to get client to '" + component + "'");
        error.addSolution("Check if '" + component + "' is correctly configured");
        return false;
    }

    Kitsunemimi::Hanami::ResponseMessage response;
    if(client->triggerSakuraFile(response, request, error) == false) {
        return false;
    }

    // check request-result
    if(response.success == false) {
        return false;
    }

    // parse response
    Kitsunemimi::JsonItem jsonItem;
    if(jsonItem.parse(response.responseContent, error) == false) {
        return false;
    }

    // check if response has valid content
    if(jsonItem.get("thread_map").getItemContent()->isMap() == false) {
        return false;
    }

    // add part to the complete map
    completeMap->insert(component, jsonItem.get("thread_map").getItemContent()->copy()->toMap());

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
 * @brief get thread-mapping of all components
 *
 * @param completeMap map with mapping of all threads of all components
 * @param token token for the access to the other components
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ThreadBinder::requestThreadMapping(Kitsunemimi::DataMap* completeMap,
                                   const std::string &token,
                                   Kitsunemimi::ErrorContainer &error)
{
    // create request
    Kitsunemimi::Hanami::RequestMessage request;
    request.id = "v1/get_thread_mapping";
    request.httpType = Kitsunemimi::Hanami::GET_TYPE;
    request.inputValues = "{ \"token\" : \"" + token + "\"}";

    SupportedComponents* scomp = SupportedComponents::getInstance();

    //----------------------------------------------------------------------------------------------
    // request from azuki itself
    if(makeInternalRequest(completeMap, error) == false) {
        return false;
    }
    //----------------------------------------------------------------------------------------------
    if(scomp->support[Kitsunemimi::Hanami::KYOUKO]
            && requestComponent(completeMap, "kyouko", request, error) == false)
    {
        return false;
    }
    //----------------------------------------------------------------------------------------------
    if(scomp->support[Kitsunemimi::Hanami::MISAKI]
            && requestComponent(completeMap, "misaki", request, error) == false)
    {
        return false;
    }
    //----------------------------------------------------------------------------------------------
    if(scomp->support[Kitsunemimi::Hanami::SHIORI]
            && requestComponent(completeMap, "shiori", request, error) == false)
    {
        return false;
    }
    //----------------------------------------------------------------------------------------------
    if(scomp->support[Kitsunemimi::Hanami::NOZOMI]
            && requestComponent(completeMap, "nozomi", request, error) == false)
    {
        return false;
    }
    //----------------------------------------------------------------------------------------------
    if(scomp->support[Kitsunemimi::Hanami::INORI]
            && requestComponent(completeMap, "inori", request, error) == false)
    {
        return false;
    }
    //----------------------------------------------------------------------------------------------
    if(requestComponent(completeMap, "torii", request, error) == false) {
        return false;
    }
    //----------------------------------------------------------------------------------------------

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

    // control-cores
    phyCore = AzukiRoot::host->cpuPackages[0]->cpuCores[0];
    for(Kitsunemimi::Sakura::CpuThread* singleThread : phyCore->cpuThreads) {
        controlCoreIds.push_back(singleThread->threadId);
    }

    // processing-cores
    for(uint64_t i = 1; i < AzukiRoot::host->cpuPackages[0]->cpuCores.size(); i++)
    {
        phyCore = AzukiRoot::host->cpuPackages[0]->cpuCores[i];
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
        if(requestThreadMapping(&newMapping, *AzukiRoot::componentToken, error) == false) {
            LOG_ERROR(error);
        }

        const std::string newMappingStr = newMapping.toString();
        if(m_lastMapping != newMappingStr)
        {
            m_completeMap = newMapping;
            // debug-output
            //std::cout<<"#############################################################"<<std::endl;
            //std::cout<<newMapping.toString(true)<<std::endl;
            //std::cout<<"#############################################################"<<std::endl;
            LOG_DEBUG(newMapping.toString(true));

            // create request for thread-binding
            Kitsunemimi::Hanami::RequestMessage request;
            request.id = "v1/bind_thread_to_core";
            request.httpType = Kitsunemimi::Hanami::POST_TYPE;

            // update thread-binding for all components
            std::map<std::string,Kitsunemimi:: DataItem*>::const_iterator it;
            for(it = newMapping.map.begin();
                it != newMapping.map.end();
                it++)
            {
                const std::vector<std::string> threadNames = it->second->toMap()->getKeys();
                if(it->first == "azuki") {
                    changeInternalCoreIds(threadNames, m_controlCoreIds);
                } else {
                    changeRemoteCoreIds(it->first, request, threadNames);
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
