/**
 * @file        hanami_root.h
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

#ifndef HANAMI_HANAMI_ROOT_H
#define HANAMI_HANAMI_ROOT_H

#include <common.h>
#include <database/cluster_table.h>
#include <database/template_table.h>
#include <database/users_table.h>
#include <database/projects_table.h>

#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiHanamiPolicies/policy.h>

class ClusterHandler;
class SegmentQueue;
class ProcessingUnitHandler;
class WebSocketServer;
class HttpServer;
class HttpWebsocketThread;
class TempFileHandler;
class ThreadBinder;
class SpeedMeasuring;
class PowerMeasuring;
class TemperatureMeasuring;
class Blossom;

using namespace Kitsunemimi::Hanami;

namespace Kitsunemimi::Sakura {
class Host;
}

namespace Kitsunemimi {
class GpuInterface;
}

class HanamiRoot
{

public:
    HanamiRoot();
    ~HanamiRoot();

    bool init(Kitsunemimi::ErrorContainer &error);
    bool initThreads();

    // blossoms
    bool triggerBlossom(Kitsunemimi::DataMap& result,
                        const std::string &blossomName,
                        const std::string &blossomGroupName,
                        const Kitsunemimi::DataMap &context,
                        const Kitsunemimi::DataMap &initialValues,
                        BlossomStatus &status,
                        Kitsunemimi::ErrorContainer &error);
    bool doesBlossomExist(const std::string &groupName,
                          const std::string &itemName);
    bool addBlossom(const std::string &groupName,
                    const std::string &itemName,
                    Blossom *newBlossom);
    Blossom* getBlossom(const std::string &groupName,
                        const std::string &itemName);

    // endpoints
    bool mapEndpoint(EndpointEntry &result,
                     const std::string &id,
                     const Kitsunemimi::Hanami::HttpRequestType type);
    bool addEndpoint(const std::string &id,
                     const Kitsunemimi::Hanami::HttpRequestType &httpType,
                     const SakuraObjectType &sakuraType,
                     const std::string &group,
                     const std::string &name);


    WebSocketServer* websocketServer = nullptr;

    static Kitsunemimi::GpuInterface* gpuInterface;
    static Kitsunemimi::Jwt* jwt;
    static HttpServer* httpServer;
    static HanamiRoot* root;
    static uint32_t* m_randomValues;
    static bool useGpu;
    static bool useCuda;

    std::map<std::string, std::map<HttpRequestType, EndpointEntry>> endpointRules;

private:
    uint32_t m_serverId = 0;
    std::vector<HttpWebsocketThread*> m_threads;
    std::map<std::string, std::map<std::string, Blossom*>> m_registeredBlossoms;

    bool initHttpServer();
    bool initSakuraServer();
    bool initDatabase(Kitsunemimi::ErrorContainer &error);
    bool initPolicies(Kitsunemimi::ErrorContainer &error);
    bool initJwt(Kitsunemimi::ErrorContainer &error);
};

#endif //HANAMI_HANAMI_ROOT_H
