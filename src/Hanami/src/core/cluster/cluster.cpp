/**
 * @file        cluster.cpp
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

#include "cluster.h"

#include <api/websocket/cluster_io.h>
#include <core/cluster/cluster_init.h>
#include <core/cluster/statemachine_init.h>
#include <core/cluster/states/task_handle_state.h>
#include <core/cuda_functions.h>
#include <core/processing/logical_host.h>
#include <hanami_common/logger.h>
#include <hanami_common/statemachine.h>
#include <hanami_common/threading/thread.h>
#include <hanami_root.h>

/**
 * @brief constructor
 */
Cluster::Cluster(LogicalHost* host)
{
    attachedHost = host;
    stateMachine = new Hanami::Statemachine();
    taskHandleState = new TaskHandle_State(this);

    m_counter.store(0, std::memory_order_relaxed);

    initStatemachine(*stateMachine, this, taskHandleState);
}

/**
 * @brief constructor to create cluster from a checkpoint
 *
 * @param data pointer to data with checkpoint
 * @param dataSize size of checkpoint in number of bytes
 */
Cluster::Cluster(LogicalHost* host, const void* data, const uint64_t dataSize)
{
    attachedHost = host;

    m_counter.store(0, std::memory_order_relaxed);
}

/**
 * @brief destructor
 */
Cluster::~Cluster() { attachedHost->removeCluster(this); }

/**
 * @brief Cluster::incrementAndCompare
 * @param referenceValue
 */
bool
Cluster::incrementAndCompare(const uint32_t referenceValue)
{
    const int incrementedValue = m_counter.fetch_add(1, std::memory_order_relaxed);
    if (incrementedValue == referenceValue - 1) {
        m_counter.store(0, std::memory_order_relaxed);
        return true;
    }

    return false;
}

/**
 * @brief get uuid of the cluster
 *
 * @return uuid of the cluster
 */
const std::string
Cluster::getUuid()
{
    return clusterHeader.uuid.toString();
}

/**
 * @brief init the cluster
 *
 * @param parsedContent TODO
 * @param segmentTemplates TODO
 * @param uuid UUID of the new cluster
 *
 * @return true, if successful, else false
 */
bool
Cluster::init(const Hanami::ClusterMeta& clusterTemplate, const std::string& uuid)
{
    return initNewCluster(this, clusterTemplate, uuid);
}

/**
 * @brief get the name of the clsuter
 *
 * @return name of the cluster
 */
const std::string
Cluster::getName()
{
    // precheck
    if (clusterHeader.nameSize == 0 || clusterHeader.nameSize > 255) {
        return std::string("");
    }

    return std::string(clusterHeader.name, clusterHeader.nameSize);
}

/**
 * @brief set new name for the cluster
 *
 * @param newName new name
 *
 * @return true, if successful, else false
 */
bool
Cluster::setName(const std::string& newName)
{
    // precheck
    if (newName.size() > 255 || newName.size() == 0) {
        return false;
    }

    // copy string into char-buffer and set explicit the escape symbol to be absolut sure
    // that it is set to absolut avoid buffer-overflows
    strncpy(clusterHeader.name, newName.c_str(), newName.size());
    clusterHeader.name[newName.size()] = '\0';
    clusterHeader.nameSize = newName.size();

    return true;
}

/**
 * @brief start a new forward train-cycle
 */
void
Cluster::startForwardCycle()
{
    attachedHost->addClusterToHost(this);
}

/**
 * @brief start a new backward train-cycle
 */
void
Cluster::startBackwardCycle()
{
    attachedHost->addClusterToHost(this);
}

/**
 * @brief Cluster::startReductionCycle
 */
void
Cluster::startReductionCycle()
{
    attachedHost->addClusterToHost(this);
}

/**
 * @brief switch state of the cluster between task and direct mode
 *
 * @param newState new desired state
 *
 * @return true, if switch in statemachine was successful, else false
 */
bool
Cluster::setClusterState(const std::string& newState)
{
    if (newState == "DIRECT") {
        return goToNextState(SWITCH_TO_DIRECT_MODE);
    }

    if (newState == "TASK") {
        return goToNextState(SWITCH_TO_TASK_MODE);
    }

    return false;
}

void
countSynapses(const Cluster& cluster)
{
    SynapseBlock* synapseBlocks
        = Hanami::getItemData<SynapseBlock>(cluster.attachedHost->synapseBlocks);
    uint64_t synapseCounter = 0;
    uint64_t sectionCounter = 0;

    for (const Brick& brick : cluster.bricks) {
        for (const ConnectionBlock& block : brick.connectionBlocks) {
            SynapseBlock* synapseBlock = &synapseBlocks[block.targetSynapseBlockPos];
            for (uint32_t i = 0; i < 64; i++) {
                if (block.connections[i].origin.blockId != UNINIT_STATE_16) {
                    sectionCounter++;
                    for (uint32_t j = 0; j < 64; j++) {
                        Synapse* synpase = &synapseBlock->sections[i].synapses[j];
                        if (synpase->targetNeuronId != UNINIT_STATE_8) {
                            synapseCounter++;
                        }
                    }
                }
            }
        }
    }

    std::cout << "section-counter: " << sectionCounter << std::endl;

    std::cout << "synpase-counter: " << synapseCounter << std::endl;
}

/**
 * @brief update state of the cluster, which is caled for each finalized cluster
 */
void
Cluster::updateClusterState()
{
    std::lock_guard<std::mutex> guard(m_clusterStateLock);

    enableCreation = false;

    // trigger next lerning phase, if already in phase 1
    if (mode == ClusterProcessingMode::TRAIN_FORWARD_MODE) {
        mode = ClusterProcessingMode::TRAIN_BACKWARD_MODE;
        startBackwardCycle();
    }
    else if (mode == ClusterProcessingMode::TRAIN_BACKWARD_MODE) {
        reductionCounter++;
        if (reductionCounter >= 100 && clusterHeader.settings.enableReduction) {
            mode = ClusterProcessingMode::REDUCTION_MODE;
            startReductionCycle();
            reductionCounter = 0;
        }
        else {
            sendClusterTrainEndMessage(this);
            // countSynapses(*this);
            goToNextState(NEXT);
        }
    }
    else if (mode == ClusterProcessingMode::REDUCTION_MODE) {
        sendClusterTrainEndMessage(this);
        goToNextState(NEXT);
    }
    else if (mode == ClusterProcessingMode::NORMAL_MODE) {
        sendClusterNormalEndMessage(this);
        goToNextState(NEXT);
    }
}

/**
 * @brief get actual task
 *
 * @return pointer to the actual task
 */
Task*
Cluster::getCurrentTask()
{
    std::lock_guard<std::mutex> guard(m_taskMutex);

    return m_currentTask;
}

/**
 * @brief get task-progress
 *
 * @param taskUuid UUID of the task
 *
 * @return task-progress
 */
const TaskProgress
Cluster::getProgress(const std::string& taskUuid)
{
    std::lock_guard<std::mutex> guard(m_taskMutex);

    const auto it = m_taskMap.find(taskUuid);
    if (it != m_taskMap.end()) {
        return it->second.progress;
    }

    TaskProgress progress;
    return progress;
}

/**
 * @brief remove task from queue of abort the task, if actual in progress
 *
 * @param taskUuid UUID of the task
 *
 * @return always true
 */
bool
Cluster::removeTask(const std::string& taskUuid)
{
    std::lock_guard<std::mutex> guard(m_taskMutex);

    TaskState state = UNDEFINED_TASK_STATE;

    // check and update map
    auto itMap = m_taskMap.find(taskUuid);
    if (itMap != m_taskMap.end()) {
        state = itMap->second.progress.state;

        // if only queue but not activly processed at the moment, it can easily deleted
        if (state == QUEUED_TASK_STATE) {
            m_taskMap.erase(itMap);

            // update queue
            const auto itQueue = std::find(m_taskQueue.begin(), m_taskQueue.end(), taskUuid);
            if (itQueue != m_taskQueue.end()) {
                m_taskQueue.erase(itQueue);
            }

            return true;
        }

        // if task is active at the moment, then only mark it as aborted
        if (state == ACTIVE_TASK_STATE) {
            itMap->second.progress.state = ABORTED_TASK_STATE;
            return true;
        }

        // handle finished and aborted state
        if (state == FINISHED_TASK_STATE || state == ABORTED_TASK_STATE) {
            // input-data are automatically deleted, when the task was finished,
            // so removing from the list is enough
            m_taskMap.erase(itMap);
            return true;
        }
    }

    return false;
}

/**
 * @brief check if a task is finished
 *
 * @param taskUuid UUID of the task
 *
 * @return true, if task is finished, else false
 */
bool
Cluster::isFinish(const std::string& taskUuid)
{
    std::lock_guard<std::mutex> guard(m_taskMutex);

    TaskState state = UNDEFINED_TASK_STATE;

    const auto it = m_taskMap.find(taskUuid);
    if (it != m_taskMap.end()) {
        state = it->second.progress.state;
    }

    return state == FINISHED_TASK_STATE;
}

/**
 * @brief Cluster::getAllProgress
 * @param result
 */
void
Cluster::getAllProgress(std::map<std::string, TaskProgress>& result)
{
    for (const auto& [name, task] : m_taskMap) {
        result.emplace(task.uuid.toString(), task.progress);
    }
}

/**
 * @brief add new task
 *
 * @param uuid uuid of the new task for identification
 * @param task task itself
 *
 * @return false, if uuid already exist, else true
 */
Task*
Cluster::addNewTask()
{
    std::lock_guard<std::mutex> guard(m_taskMutex);

    Task newTask;
    const std::string taskUuid = newTask.uuid.toString();
    auto ret = m_taskMap.try_emplace(taskUuid, std::move(newTask));
    if (ret.second == false) {
        return nullptr;
    }

    m_taskQueue.push_back(taskUuid);

    return &m_taskMap[taskUuid];
}

/**
 * @brief run next task from the queue
 *
 * @return false, if task-queue if empty, else true
 */
bool
Cluster::getNextTask()
{
    // check number of tasks in queue
    if (m_taskQueue.size() == 0) {
        return false;
    }

    // remove task from queue
    const std::string nextUuid = m_taskQueue.front();
    m_taskQueue.pop_front();

    // init the new task
    auto it = m_taskMap.find(nextUuid);
    it->second.progress.state = ACTIVE_TASK_STATE;
    it->second.progress.startActiveTimeStamp = std::chrono::system_clock::now();
    m_currentTask = &it->second;

    return true;
}

/**
 * @brief switch statemachine of cluster to next state
 *
 * @param nextStateId id of the next state
 *
 * @return true, if statemachine switch was successful, else false
 */
bool
Cluster::goToNextState(const uint32_t nextStateId)
{
    return stateMachine->goToNextState(nextStateId);
}

/**
 * @brief finish current task
 */
TaskType
Cluster::finishTask()
{
    std::lock_guard<std::mutex> guard(m_taskMutex);

    // precheck
    if (m_currentTask != nullptr) {
        Hanami::ErrorContainer error;
        // TODO: handle error-ourpur
        m_currentTask->progress.state = FINISHED_TASK_STATE;
        m_currentTask->progress.endActiveTimeStamp = std::chrono::system_clock::now();

        m_currentTask = nullptr;
    }

    getNextTask();

    if (m_currentTask == nullptr) {
        return NO_TASK;
    }

    return m_currentTask->type;
}
