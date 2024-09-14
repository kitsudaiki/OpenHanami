/**
 * @file        cuda_worker_thread.cpp
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

#include "cuda_worker_thread.h"

/**
 * @brief constructor
 *
 * @param host pointer to related cuda-host, which holds the task-queue for the worker
 */
CudaWorkerThread::CudaWorkerThread(CudaHost* host) : Hanami::Thread("WorkerThread")
{
    m_host = host;
}

/**
 * @brief destructor
 */
CudaWorkerThread::~CudaWorkerThread() {}

/**
 * @brief rum worker-thread and get tasks for the task-queue of the connected cpu-host
 */
void
CudaWorkerThread::run()
{
    Cluster* cluster = nullptr;

    while (m_abort == false) {
        if (cluster != nullptr) {
            // handle type of processing
            if (cluster->mode == ClusterProcessingMode::TRAIN_FORWARD_MODE) {
                trainClusterForward(cluster);
                // processNeuronsOfOutputHexagon<true>();
            }
            else if (cluster->mode == ClusterProcessingMode::TRAIN_BACKWARD_MODE) {
                // backpropagateOutput(*cluster);
                trainClusterBackward(cluster);
            }
            else {
                requestCluster(cluster);
                // processNeuronsOfOutputHexagon<false>(*cluster);
                handleClientOutput(*cluster);
            }
            cluster->updateClusterState();
        }
        else {
            // if no segments are available then sleep
            sleepThread(1000);
        }
    }
}

/**
 * @brief run forward-propagation on a cluster
 *
 * @param cluster cluster to process
 */
void
CudaWorkerThread::trainClusterForward(Cluster* cluster)
{
    const std::lock_guard<std::mutex> lock(m_host->cudaMutex);

    Hanami::ErrorContainer error;

    // see https://github.com/kitsudaiki/Hanami/issues/377
    /* // process input-hexagons
     for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
         Hexagon* hexagon = &cluster->hexagons[hexagonId];
         if (hexagon->isInputHexagon == false) {
             continue;
         }

         processNeuronsOfInputHexagonBackward<true>(
             hexagon, cluster->inputValues, &cluster->neuronBlocks);
     }

     // process all hexagons on cpu
     processing_CUDA(&cluster->gpuPointer,
                     &cluster->hexagons[0],
                     cluster->hexagons.size(),
                     &cluster->neuronBlocks,
                     cluster->numberOfNeuronBlocks,
                     true);

     // process output-hexagons
     for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
         Hexagon* hexagon = &cluster->hexagons[hexagonId];
         if (hexagon->isOutputHexagon == false) {
             continue;
         }
     }

     // update cluster
     if (updateCluster(*cluster)) {
         update_CUDA(&cluster->gpuPointer,
                     &cluster->neuronBlocks,
                     cluster->numberOfNeuronBlocks,
                     &cluster->hexagons[0],
                     cluster->hexagons.size());
     }*/
}

/**
 * @brief run back-propagation on a cluster
 *
 * @param cluster cluster to process
 */
void
CudaWorkerThread::trainClusterBackward(Cluster* cluster)
{
    const std::lock_guard<std::mutex> lock(m_host->cudaMutex);

    Hanami::ErrorContainer error;

    // process output-hexagons on cpu
    for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
        Hexagon* hexagon = &cluster->hexagons[hexagonId];
        if (hexagon->header.isOutputHexagon) {
            // see https://github.com/kitsudaiki/Hanami/issues/377
            /*if (backpropagateOutput(&cluster->hexagons[0],
                                    &cluster->outputNeurons[0],
                                    &cluster->neuronBlocks,
                                    &cluster->tempNeuronBlocks,
                                    cluster->outputValues,
                                    cluster->expectedValues,
                                    &cluster->clusterHeader.settings)
                == false)
            {
                return;
            }*/
        }
    }

    // see https://github.com/kitsudaiki/Hanami/issues/377
    // backpropagation over all hexagons on gpu
    /*backpropagation_CUDA(&cluster->gpuPointer,
                         &cluster->hexagons[0],
                         cluster->hexagons.size(),
                         &cluster->neuronBlocks,
                         &cluster->tempNeuronBlocks,
                         cluster->numberOfNeuronBlocks);

    // run reduction-process if enabled
    if (cluster->clusterHeader.settings.enableReduction) {
        if (reductionCounter == 100) {
            reduction_CUDA(&cluster->gpuPointer,
                           &cluster->hexagons[0],
                           cluster->hexagons.size(),
                           &cluster->neuronBlocks,
                           cluster->numberOfNeuronBlocks);
            if (updateCluster(*cluster)) {
                update_CUDA(&cluster->gpuPointer,
                            &cluster->neuronBlocks,
                            cluster->numberOfNeuronBlocks,
                            &cluster->hexagons[0],
                            cluster->hexagons.size());
            }
            reductionCounter = 0;
        }
        reductionCounter++;
    }*/
}

/**
 * @brief process segments
 *
 * @param cluster cluster to process
 */
void
CudaWorkerThread::requestCluster(Cluster* cluster)
{
    const std::lock_guard<std::mutex> lock(m_host->cudaMutex);

    Hanami::ErrorContainer error;

    // see https://github.com/kitsudaiki/Hanami/issues/377
    // process input-hexagons
    /*for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
        Hexagon* hexagon = &cluster->hexagons[hexagonId];
        if (hexagon->header.isInputHexagon == false) {
            continue;
        }

        processNeuronsOfInputHexagonBackward<false>(
            hexagon, cluster->inputValues, &cluster->neuronBlocks);
    }

    // process all hexagons on gpu
    processing_CUDA(&cluster->gpuPointer,
                    &cluster->hexagons[0],
                    cluster->hexagons.size(),
                    &cluster->neuronBlocks,
                    cluster->numberOfNeuronBlocks,
                    false);*/

    // process output-hexagons
    for (uint32_t hexagonId = 0; hexagonId < cluster->hexagons.size(); ++hexagonId) {
        Hexagon* hexagon = &cluster->hexagons[hexagonId];
        if (hexagon->header.isOutputHexagon == false) {
            continue;
        }
        // see https://github.com/kitsudaiki/Hanami/issues/377
        /*for (uint32_t blockId = 0; blockId < cluster->numberOfNeuronBlocks; ++blockId) {
            processNeuronsOfOutputHexagon(
                hexagon, cluster->outputValues, &cluster->neuronBlocks, blockId);
        }*/
    }
}
