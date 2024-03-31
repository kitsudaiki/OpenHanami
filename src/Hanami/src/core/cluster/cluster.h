/**
 * @file        cluster.h
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

#ifndef HANAMI_CLUSTER_H
#define HANAMI_CLUSTER_H

#include <api/endpoint_processing/http_websocket_thread.h>
#include <common.h>
#include <core/cluster/task.h>
#include <core/processing/objects.h>
#include <hanami_cluster_parser/cluster_meta.h>
#include <hanami_common/buffer/data_buffer.h>
#include <hanami_common/buffer/item_buffer.h>

#include <atomic>

class TaskHandle_State;
class LogicalHost;

namespace Hanami
{
class Statemachine;
}  // namespace Hanami

class Cluster
{
   public:
    Cluster(LogicalHost* host);
    Cluster(LogicalHost* host, const void* data, const uint64_t dataSize);
    ~Cluster();

    // processing
    CudaPointerHandle gpuPointer;
    LogicalHost* attachedHost;
    uint32_t reductionCounter = 0;
    ClusterProcessingMode mode = NORMAL_MODE;
    HttpWebsocketThread* msgClient = nullptr;
    Hanami::Statemachine* stateMachine = nullptr;
    TaskHandle_State* taskHandleState = nullptr;
    bool enableCreation = false;

    // cluster-data
    Hanami::DataBuffer clusterData;
    ClusterHeader* clusterHeader = nullptr;
    float* inputValues = nullptr;
    float* outputValues = nullptr;
    float* expectedValues = nullptr;
    std::map<std::string, Brick*> namedBricks;
    Brick* bricks = nullptr;
    NeuronBlock* neuronBlocks = nullptr;
    TempNeuronBlock* tempNeuronBlocks = nullptr;
    uint32_t numberOfNeuronBlocks = 0;

    // meta
    const std::string getUuid();
    const std::string getName();
    bool setName(const std::string& newName);
    bool init(const Hanami::ClusterMeta& clusterTemplate, const std::string& uuid);

    // stats
    uint64_t getDataSize() const;

    // tasks
    Task* getActualTask() const;
    uint64_t getActualTaskCycle() const;
    const TaskProgress getProgress(const std::string& taskUuid);
    bool removeTask(const std::string& taskUuid);
    bool isFinish(const std::string& taskUuid);
    void getAllProgress(std::map<std::string, TaskProgress>& result);
    void updateClusterState();

    // states
    bool goToNextState(const uint32_t nextStateId);
    void startForwardCycle();
    void startBackwardCycle();
    void startReductionCycle();
    bool setClusterState(const std::string& newState);

    // counter for parallel-processing
    bool incrementAndCompare(const uint32_t referenceValue);
    std::atomic<int> counter;

   private:
    std::mutex m_clusterStateLock;
};

#endif  // HANAMI_CLUSTER_H
