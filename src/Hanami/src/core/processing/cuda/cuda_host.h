/**
 * @file        cuda_host.h
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

#ifndef CUDAHOST_H
#define CUDAHOST_H

#include <core/processing/logical_host.h>

class CudaHost : public LogicalHost
{
   public:
    CudaHost();
    ~CudaHost();

    uint64_t getAvailableMemory();
    void hostSpecificCleanup(Cluster* cluster);

    bool moveCluster(LogicalHost* originHost, Cluster* cluster);

   private:
    void trainClusterForward(Cluster* cluster);
    void trainClusterBackward(Cluster* cluster);
    void requestCluster(Cluster* cluster);
};

#endif  // CUDAHOST_H
