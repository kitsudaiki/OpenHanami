/**
 * @file        logical_host.h
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

#ifndef LOGICALHOST_H
#define LOGICALHOST_H

#include <core/cluster/cluster.h>
#include <core/processing/worker_thread.h>
#include <hanami_common/buffer/item_buffer.h>
#include <hanami_common/logger.h>
#include <stdint.h>

#include <atomic>
#include <deque>
#include <vector>

class Cluster;

void handleClientOutput(Cluster& cluster);

class LogicalHost
{
   public:
    enum HostType {
        UNDEFINED_HOST_TYPE = 0,
        CPU_HOST_TYPE = 1,
        CUDA_HOST_TYPE = 2,
    };

    LogicalHost(const uint32_t localId);
    virtual ~LogicalHost();

    virtual void addClusterToHost(Cluster* cluster) = 0;

    HostType getHostType() const;
    const std::string getUuid() const;
    uint64_t getTotalMemory();

    virtual bool moveCluster(Cluster* cluster) = 0;
    virtual void syncWithHost(Cluster* cluster) = 0;
    virtual void removeCluster(Cluster* cluster) = 0;

    Hanami::ItemBuffer synapseBlocks;

    void addWorkerTaskToQueue(const Hanami::WorkerTask task);
    const Hanami::WorkerTask getWorkerTaskFromQueue();

   protected:
    void continueAllThreads();

    uint64_t reductionCounter = 0;

    HostType m_hostType = UNDEFINED_HOST_TYPE;
    std::string m_uuid = "";
    uint32_t m_localId = 0;
    uint64_t m_totalMemory = 0;

    std::vector<WorkerThread*> m_workerThreads;
    std::deque<Hanami::WorkerTask> m_workerTaskQueue;
    std::mutex m_queue_lock;
};

#endif  // LOGICALHOST_H
