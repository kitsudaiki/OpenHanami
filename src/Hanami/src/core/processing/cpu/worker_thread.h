/**
 * @file        worker_thread.h
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

#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H

#include <common.h>
#include <core/processing/cpu/cpu_host.h>
#include <hanami_common/threading/thread.h>

class Cluster;

class WorkerThread : public Hanami::Thread
{
   public:
    WorkerThread(CpuHost* host);
    ~WorkerThread();

   protected:
    void run();

   private:
    void processClusterForward(Cluster& cluster,
                               const uint32_t brickId,
                               const uint32_t blockId,
                               const bool doTrain);
    void processClusterBackward(Cluster& cluster, const uint32_t brickId, const uint32_t blockId);

    void handleTask(const CpuHost::WorkerTask task);
    void handleTrainForwardTask(CpuHost::WorkerTask task);
    void handleTrainBackwardTask(CpuHost::WorkerTask task);
    void handleReductionTask(const CpuHost::WorkerTask task);
    void handleProcessTask(const CpuHost::WorkerTask task);

    void handleInputForward(Cluster& cluster, InputInterface* inputInterface, const bool doTrain);

    CpuHost* m_host = nullptr;
    uint32_t m_inactiveCounter = 0;
};

#endif  // WORKERTHREAD_H
