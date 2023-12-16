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
#include <cryptopp/secblock.h>
#include <hanami_common/buffer/item_buffer.h>

class ClusterHandler;
class ClusterQueue;
class ProcessingUnitHandler;
class WebSocketServer;
class HttpServer;
class HttpWebsocketThread;
class TempFileHandler;
class ThreadBinder;
class SpeedMeasuring;
class PowerMeasuring;
class TemperatureMeasuring;
using namespace Hanami;

namespace Hanami
{
class Host;
class GpuInterface;
}  // namespace Hanami

class HanamiRoot
{
   public:
    HanamiRoot();
    ~HanamiRoot();

    bool init(Hanami::ErrorContainer& error);
    bool initThreads();
    bool initHttpServer();

    WebSocketServer* websocketServer = nullptr;

    static Hanami::GpuInterface* gpuInterface;
    static HttpServer* httpServer;
    static HanamiRoot* root;
    static uint32_t* randomValues;
    static CryptoPP::SecByteBlock tokenKey;
    static Hanami::ItemBuffer cpuSynapseBlocks;
    static Hanami::ItemBuffer gpuSynapseBlocks;
    static bool useCuda;

   private:
    std::vector<HttpWebsocketThread*> m_threads;

    bool initDataDirectory(Hanami::ErrorContainer& error);
    bool initDatabase(Hanami::ErrorContainer& error);
    bool initPolicies(Hanami::ErrorContainer& error);
    bool initJwt(Hanami::ErrorContainer& error);

    void clearCluster(Hanami::ErrorContainer& error);
};

#endif  // HANAMI_HANAMI_ROOT_H
