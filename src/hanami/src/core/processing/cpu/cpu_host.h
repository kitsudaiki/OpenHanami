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

#include <core/processing/cpu/cpu_worker_thread.h>
#include <core/processing/logical_host.h>

class CpuWorkerThread;

class CpuHost : public LogicalHost
{
   public:
    CpuHost(const uint32_t localId);
    ~CpuHost();

    void addClusterToHost(Cluster* cluster);

    bool moveCluster(Cluster* cluster);
    void syncWithHost(Cluster*);
    void removeCluster(Cluster* cluster);

   protected:
    bool initWorkerThreads();

   private:
    void initBuffer();
};

#endif  // CPUHOST_H
