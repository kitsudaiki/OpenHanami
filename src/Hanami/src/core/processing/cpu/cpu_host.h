/**
 * @file        cpu_host.h
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

#ifndef CPUHOST_H
#define CPUHOST_H

#include <common.h>
#include <core/processing/logical_host.h>

class WorkerThread;

class CpuHost : public LogicalHost
{
   public:
    struct WorkerTask {
        Cluster* cluster = nullptr;
        uint32_t brickId = UNINIT_STATE_32;
        uint32_t blockId = UNINIT_STATE_32;
    };

    CpuHost(const uint32_t localId);
    ~CpuHost();

    void addClusterToHost(Cluster* cluster);
    Cluster* getClusterFromQueue();
    void addBrickToTaskQueue(Cluster* cluster, const u_int32_t brickId);

    bool moveCluster(Cluster* cluster);
    void syncWithHost(Cluster*);
    void removeCluster(Cluster* cluster);

    void addWorkerTaskToQueue(const WorkerTask task);
    const WorkerTask getWorkerTaskFromQueue();

   private:
    void trainClusterForward(Cluster* cluster);
    void trainClusterBackward(Cluster* cluster);
    void requestCluster(Cluster* cluster);
    void initBuffer(const uint32_t id);
    bool initWorkerThreads();

    std::vector<WorkerThread*> m_workerThreads;
    std::deque<WorkerTask> m_workerTaskQueue;
    std::mutex m_queue_lock;
};

#endif  // CPUHOST_H
