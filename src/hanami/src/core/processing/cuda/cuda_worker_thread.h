/**
 * @file        cuda_worker_thread.h
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

#ifndef CUDAWORKERTHREAD_H
#define CUDAWORKERTHREAD_H

#include <core/processing/cuda/cuda_host.h>
#include <hanami_common/threading/thread.h>

class Cluster;

class CudaWorkerThread : public Hanami::Thread
{
   public:
    CudaWorkerThread(CudaHost* host);
    ~CudaWorkerThread();

   protected:
    void run();

   private:
    void trainClusterForward(Cluster* cluster);
    void trainClusterBackward(Cluster* cluster);
    void requestCluster(Cluster* cluster);

    CudaHost* m_host = nullptr;
};

#endif  // CUDAWORKERTHREAD_H
