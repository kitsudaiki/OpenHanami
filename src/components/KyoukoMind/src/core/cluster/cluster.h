/**
 * @file        cluster_interface.h
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

#ifndef KYOUKOMIND_CLUSTER_INTERFACE_H
#define KYOUKOMIND_CLUSTER_INTERFACE_H

#include <common.h>
#include <core/cluster/task.h>

#include <libKitsunemimiHanamiClusterParser/cluster_meta.h>
#include <libKitsunemimiHanamiSegmentParser/segment_meta.h>

class AbstractSegment;
class InputSegment;
class OutputSegment;
class AbstractSegment;
class TaskHandle_State;

namespace Kitsunemimi {
class EventQueue;
class Statemachine;
}

namespace Kitsunemimi::Hanami {
class HanamiMessagingClient;
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

        Kitsunemimi::Hanami::kuuid uuid;
        char name[1024];

        uint32_t numberOfInputSegments = 0;
        uint32_t numberOfOutputSegments = 0;
        uint32_t numberOfSegments = 0;

        uint8_t padding2[956];

        // total size: 2048 Byte
    };

    struct Settings
    {
        float lerningValue = 0.0f;

        uint8_t padding[252];
        // total size: 256 Byte
    };

    // cluster-data
    Kitsunemimi::DataBuffer clusterData;
    Cluster::MetaData* networkMetaData = nullptr;
    Cluster::Settings* networkSettings = nullptr;
    std::map<std::string, InputSegment*> inputSegments;
    std::map<std::string, OutputSegment*> outputSegments;
    std::map<std::string, AbstractSegment*> coreSegments;
    std::vector<AbstractSegment*> allSegments;

    const std::string getUuid();
    const std::string getName();
    bool setName(const std::string newName);
    bool init(const Kitsunemimi::Hanami::ClusterMeta &clusterTemplate,
              const std::map<std::string, Kitsunemimi::Hanami::SegmentMeta> &segmentTemplates,
              const std::string &uuid);
    bool connectSlot(const std::string &sourceSegmentName,
                     const std::string &sourceSlotName,
                     const std::string &targetSegmentName,
                     const std::string &targetSlotName);
    uint64_t getSegmentId(const std::string &name);

    // task-handling
    void updateClusterState();
    const std::string addImageLearnTask(const std::string &name,
                                        const std::string &userId,
                                        const std::string &projectId,
                                        float* inputData,
                                        const uint64_t numberOfInputsPerCycle,
                                        const uint64_t numberOfOuputsPerCycle,
                                        const uint64_t numberOfCycle);
    const std::string addImageRequestTask(const std::string &name,
                                          const std::string &userId,
                                          const std::string &projectId,
                                          float* inputData,
                                          const uint64_t numberOfInputsPerCycle,
                                          const uint64_t numberOfOuputsPerCycle,
                                          const uint64_t numberOfCycle);
    const std::string addTableLearnTask(const std::string &name,
                                        const std::string &userId,
                                        const std::string &projectId,
                                        float* inputData,
                                        float* outputData,
                                        const uint64_t numberOfInputs,
                                        const uint64_t numberOfOutputs,
                                        const uint64_t numberOfCycle);
    const std::string addTableRequestTask(const std::string &name,
                                          const std::string &userId,
                                          const std::string &projectId,
                                          float* inputData,
                                          const uint64_t numberOfInputs,
                                          const uint64_t numberOfOutputs,
                                          const uint64_t numberOfCycle);
    const std::string addClusterSnapshotSaveTask(const std::string &snapshotName,
                                                 const std::string &userId,
                                                 const std::string &projectId);
    const std::string addClusterSnapshotRestoreTask(const std::string &name,
                                                    const std::string &snapshotInfo,
                                                    const std::string &userId,
                                                    const std::string &projectId);

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
    Kitsunemimi::Hanami::HanamiMessagingClient* msgClient = nullptr;

private:
    Kitsunemimi::Statemachine* m_stateMachine = nullptr;
    TaskHandle_State* m_taskHandleState = nullptr;
    std::mutex m_segmentCounterLock;
};

#endif // KYOUKOMIND_CLUSTER_INTERFACE_H
