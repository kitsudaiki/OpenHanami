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

#include <common.h>
#include <core/cluster/cluster.h>
#include <hanami_common/buffer/item_buffer.h>
#include <hanami_common/logger.h>
#include <hanami_common/threading/thread.h>
#include <stdint.h>

#include <atomic>
#include <deque>
#include <vector>

class Cluster;

class LogicalHost : public Hanami::Thread
{
   public:
    enum HostType {
        UNDEFINED_HOST_TYPE = 0,
        CPU_HOST_TYPE = 1,
        CUDA_HOST_TYPE = 2,
    };

    LogicalHost(const uint32_t localId);
    virtual ~LogicalHost();

    void addClusterToQueue(Cluster* cluster);
    Cluster* getClusterFromQueue();

    HostType getHostType() const;
    const std::string getUuid() const;
    uint64_t getTotalMemory();

    virtual bool moveCluster(Cluster* cluster) = 0;
    virtual void syncWithHost(Cluster* cluster) = 0;
    virtual void removeCluster(Cluster* cluster) = 0;

    Hanami::ItemBuffer synapseBlocks;

   protected:
    void run();

    uint32_t getHighestOutput(const Cluster& cluster);
    void handleClientOutput(const Cluster& cluster);
    uint64_t reductionCounter = 0;

    std::atomic_flag m_queue_lock = ATOMIC_FLAG_INIT;
    std::deque<Cluster*> m_clusterQueue;
    HostType m_hostType = UNDEFINED_HOST_TYPE;
    std::string m_uuid = "";
    uint32_t m_localId = 0;
    uint64_t m_totalMemory = 0;

   private:
    virtual void trainClusterForward(Cluster* cluster) = 0;
    virtual void trainClusterBackward(Cluster* cluster) = 0;
    virtual void requestCluster(Cluster* cluster) = 0;
};

#endif  // LOGICALHOST_H
