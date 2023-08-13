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
#include <api/endpoint_processing/http_websocket_thread.h>

#include <libKitsunemimiCommon/buffer/data_buffer.h>

#include <libKitsunemimiHanamiClusterParser/segment_meta.h>

class CoreSegment;
class TaskHandle_State;

namespace Kitsunemimi {
class EventQueue;
class Statemachine;
}

class Cluster
{
public:
    Cluster();
    ~Cluster();

    enum ClusterProcessingMode
    {
        NORMAL_MODE = 0,
        LEARN_FORWARD_MODE = 1,
        LEARN_BACKWARD_MODE = 2,
    };

    struct MetaData
    {
        uint8_t objectType = CLUSTER_OBJECT;
        uint8_t version = 1;
        uint8_t padding1[6];
        uint64_t clusterSize = 0;

        kuuid uuid;
        char name[1024];

        uint32_t numberOfInputSegments = 0;
        uint32_t numberOfOutputSegments = 0;
        uint32_t numberOfSegments = 0;

        uint8_t padding2[956];
    };
    static_assert(sizeof(MetaData) == 2048);

    struct Settings
    {
        float lerningValue = 0.0f;
        uint8_t padding[252];
    };
    static_assert(sizeof(Settings) == 256);

    // cluster-data
    Kitsunemimi::DataBuffer clusterData;
    MetaData* networkMetaData = nullptr;
    Settings* networkSettings = nullptr;
    std::vector<CoreSegment*> coreSegments;

    const std::string getUuid();
    const std::string getName();
    bool setName(const std::string newName);
    bool init(const Kitsunemimi::Hanami::SegmentMeta &clusterTemplate,
              const std::string &uuid);
    bool connectSlot(const std::string &sourceSegmentName,
                     const std::string &sourceSlotName,
                     const std::string &targetSegmentName,
                     const std::string &targetSlotName);
    uint64_t getSegmentId(const std::string &name);

    // task-handling
    void updateClusterState();

    // tasks
    Task* getActualTask() const;
    uint64_t getActualTaskCycle() const;
    const TaskProgress getProgress(const std::string &taskUuid);
    bool removeTask(const std::string &taskUuid);
    bool isFinish(const std::string &taskUuid);
    void getAllProgress(std::map<std::string, TaskProgress> &result);

    bool goToNextState(const uint32_t nextStateId);
    void startForwardCycle();
    void startBackwardCycle();
    bool setClusterState(const std::string &newState);

    uint32_t segmentCounter = 0;
    ClusterProcessingMode mode = NORMAL_MODE;
    HttpWebsocketThread* msgClient = nullptr;

    Kitsunemimi::Statemachine* stateMachine = nullptr;
    TaskHandle_State* taskHandleState = nullptr;

private:
    std::mutex m_segmentCounterLock;
};

#endif // HANAMI_CLUSTER_H
