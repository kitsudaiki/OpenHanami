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

#include <common.h>
#include <core/cluster/task.h>
#include <core/processing/objects.h>
#include <api/endpoint_processing/http_websocket_thread.h>

#include <libKitsunemimiHanamiClusterParser/cluster_meta.h>

#include <libKitsunemimiCommon/buffer/data_buffer.h>
#include <libKitsunemimiCommon/buffer/item_buffer.h>

class TaskHandle_State;

namespace Kitsunemimi {
class EventQueue;
class Statemachine;
}

class Cluster
{
public:
    Cluster();
    Cluster(const void *data, const uint64_t dataSize);
    ~Cluster();

    // cluster-data
    Kitsunemimi::ItemBuffer clusterData;
    PointerHandler gpuPointer;

    ClusterHeader* clusterHeader = nullptr;
    SegmentSettings* clusterSettings = nullptr;
    float* inputValues = nullptr;
    float* outputValues = nullptr;
    float* expectedValues = nullptr;

    Brick* bricks = nullptr;
    uint32_t* brickOrder = nullptr;

    NeuronBlock* neuronBlocks = nullptr;
    SynapseConnection* synapseConnections = nullptr;
    SynapseBlock* synapseBlocks = nullptr;
    uint32_t numberOfBrickBlocks = 0;

    const std::string getUuid();
    const std::string getName();
    bool setName(const std::string newName);
    bool init(const Kitsunemimi::Hanami::ClusterMeta &clusterTemplate,
              const std::string &uuid);

    // tasks
    Task* getActualTask() const;
    uint64_t getActualTaskCycle() const;
    const TaskProgress getProgress(const std::string &taskUuid);
    bool removeTask(const std::string &taskUuid);
    bool isFinish(const std::string &taskUuid);
    void getAllProgress(std::map<std::string, TaskProgress> &result);
    void updateClusterState();

    // states
    bool goToNextState(const uint32_t nextStateId);
    void startForwardCycle();
    void startBackwardCycle();
    bool setClusterState(const std::string &newState);

    ClusterProcessingMode mode = NORMAL_MODE;
    HttpWebsocketThread* msgClient = nullptr;

    Kitsunemimi::Statemachine* stateMachine = nullptr;
    TaskHandle_State* taskHandleState = nullptr;

private:
    std::mutex m_segmentCounterLock;
    void initCuda();
};

#endif // HANAMI_CLUSTER_H
