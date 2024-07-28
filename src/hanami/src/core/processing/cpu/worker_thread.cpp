/**
 * @file        worker_thread.cpp
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

#include "worker_thread.h"

#include <core/cluster/objects.h>
#include <core/processing/cluster_io_functions.h>
#include <core/processing/cluster_resize.h>
#include <core/processing/cpu/backpropagation.h>
#include <core/processing/cpu/processing.h>
#include <core/processing/cpu/reduction.h>
#include <core/processing/logical_host.h>

/**
 * @brief constructor
 *
 * @param host pointer to related cpu-host, which holds the task-queue for the worker
 */
WorkerThread::WorkerThread(CpuHost* host) : Hanami::Thread("WorkerThread") { m_host = host; }

/**
 * @brief destructor
 */
WorkerThread::~WorkerThread() {}

/**
 * @brief rum worker-thread and get tasks for the task-queue of the connected cpu-host
 */
void
WorkerThread::run()
{
    while (m_abort == false) {
        CpuHost::WorkerTask task = m_host->getWorkerTaskFromQueue();
        if (task.cluster != nullptr) {
            handleTask(task);
        }
        else {
            if (m_inactiveCounter < 100) {
                usleep(10);
                m_inactiveCounter++;
            }
            else {
                m_inactiveCounter = 0;
                blockThread();
            }
        }
    }
}

/**
 * @brief handle a task from the task-queue
 *
 * @param task task to handle
 */
void
WorkerThread::handleTask(const CpuHost::WorkerTask task)
{
    if (task.cluster->mode == ClusterProcessingMode::TRAIN_FORWARD_MODE) {
        handleTrainForwardTask(task);
    }
    else if (task.cluster->mode == ClusterProcessingMode::TRAIN_BACKWARD_MODE) {
        handleTrainBackwardTask(task);
    }
    else if (task.cluster->mode == ClusterProcessingMode::REDUCTION_MODE) {
        handleReductionTask(task);
    }
    else {
        handleProcessTask(task);
    }
}

/**
 * @brief handle trainging task
 *
 * @param task task to handle
 */
void
WorkerThread::handleTrainForwardTask(CpuHost::WorkerTask task)
{
    Hexagon* hexagon = &task.cluster->hexagons[task.hexagonId];
    if (task.blockId == UNINIT_STATE_16) {
        // handle input-interface
        if (hexagon->inputInterface != nullptr) {
            WorkerThread::handleInputForward(*task.cluster, hexagon->inputInterface, true);
        }

        // handle special-case that there are no neuron-blocks to process
        if (hexagon->neuronBlocks.size() == 0) {
            if (task.hexagonId == task.cluster->hexagons.size() - 1) {
                updateCluster(*task.cluster);
                task.cluster->updateClusterState();
                return;
            }

            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId + 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
            return;
        }

        // share neuron-blocks to process
        for (uint32_t i = 0; i < hexagon->neuronBlocks.size(); i++) {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId;
            newTask.blockId = i;
            m_host->addWorkerTaskToQueue(newTask);
        }
        return;
    }

    // run backpropation
    processClusterForward(*task.cluster, task.hexagonId, task.blockId, true);
    if (task.cluster->incrementAndCompare(
            task.cluster->hexagons[task.hexagonId].neuronBlocks.size()))
    {
        if (hexagon->outputInterface != nullptr) {
            processNeuronsOfOutputHexagon<true>(
                task.cluster->hexagons, hexagon->outputInterface, task.hexagonId, rand());
        }

        if (task.hexagonId == task.cluster->hexagons.size() - 1) {
            updateCluster(*task.cluster);
            task.cluster->updateClusterState();
        }
        else {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId + 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
        }
    }
}

/**
 * @brief handle backpropagation task
 *
 * @param task task to handle
 */
void
WorkerThread::handleTrainBackwardTask(CpuHost::WorkerTask task)
{
    if (task.blockId == UNINIT_STATE_16) {
        Hexagon* hexagon = &task.cluster->hexagons[task.hexagonId];

        // handle output-interface
        if (hexagon->outputInterface != nullptr) {
            backpropagateOutput(task.cluster->hexagons,
                                hexagon->outputInterface,
                                &task.cluster->clusterHeader.settings,
                                task.hexagonId);
        }

        // handle special-case that there are no neuron-blocks to process
        if (hexagon->neuronBlocks.size() == 0) {
            if (task.hexagonId == 0) {
                task.cluster->updateClusterState();
                return;
            }

            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId - 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
            return;
        }

        // share neuron-blocks to process
        for (uint32_t i = 0; i < hexagon->neuronBlocks.size(); i++) {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId;
            newTask.blockId = i;
            m_host->addWorkerTaskToQueue(newTask);
        }
        return;
    }

    // run backpropation
    processClusterBackward(*task.cluster, task.hexagonId, task.blockId);
    if (task.cluster->incrementAndCompare(
            task.cluster->hexagons[task.hexagonId].neuronBlocks.size()))
    {
        if (task.hexagonId == 0) {
            task.cluster->updateClusterState();
        }
        else {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId - 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
        }
    }
}

/**
 * @brief handle reduction task
 *
 * @param task task to handle
 */
void
WorkerThread::handleReductionTask(const CpuHost::WorkerTask task)
{
    /*reduceCluster(*task.cluster, task.hexagonId, task.blockId);
    if
    (task.cluster->incrementAndCompare(task.cluster->hexagons[task.hexagonId].neuronBlocks.size()))
    { if (task.hexagonId == task.cluster->hexagons.size() - 1) { task.cluster->updateClusterState();
        }
        else {
            m_host->addHexagonToTaskQueue(task.cluster, task.hexagonId + 1);
            return;
        }
    }*/
}

/**
 * @brief handle process task
 *
 * @param task task to handle
 */
void
WorkerThread::handleProcessTask(const CpuHost::WorkerTask task)
{
    Hexagon* hexagon = &task.cluster->hexagons[task.hexagonId];

    if (task.blockId == UNINIT_STATE_16) {
        // handle input-interface
        if (hexagon->inputInterface != nullptr) {
            WorkerThread::handleInputForward(*task.cluster, hexagon->inputInterface, true);
        }

        // handle special-case that there are no neuron-blocks to process
        if (hexagon->neuronBlocks.size() == 0) {
            if (task.hexagonId == task.cluster->hexagons.size() - 1) {
                task.cluster->updateClusterState();
                return;
            }

            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId + 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
            return;
        }

        // share neuron-blocks to process
        for (uint32_t i = 0; i < hexagon->neuronBlocks.size(); i++) {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId;
            newTask.blockId = i;
            m_host->addWorkerTaskToQueue(newTask);
        }
        return;
    }

    // run backpropation
    processClusterForward(*task.cluster, task.hexagonId, task.blockId, true);
    if (task.cluster->incrementAndCompare(
            task.cluster->hexagons[task.hexagonId].neuronBlocks.size()))
    {
        if (hexagon->outputInterface != nullptr) {
            processNeuronsOfOutputHexagon<false>(
                task.cluster->hexagons, hexagon->outputInterface, task.hexagonId, rand());
        }

        if (task.hexagonId == task.cluster->hexagons.size() - 1) {
            handleClientOutput(*task.cluster);
            task.cluster->updateClusterState();
        }
        else {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId + 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
        }
    }
}

/**
 * @brief handle input-hexagons by applying input-values to the input-neurons
 *
 * @param cluster pointer to cluster to process
 * @param doTrain true to run trainging-process
 */
void
WorkerThread::handleInputForward(Cluster& cluster,
                                 InputInterface* inputInterface,
                                 const bool doTrain)
{
    Hexagon* hexagon = nullptr;
    const uint32_t numberOfHexagons = cluster.hexagons.size();

    // process input-hexagons
    for (uint32_t hexagonId = 0; hexagonId < numberOfHexagons; ++hexagonId) {
        hexagon = &cluster.hexagons[hexagonId];

        if (hexagon->header.isInputHexagon) {
            if (doTrain) {
                processNeuronsOfInputHexagon<true>(cluster, inputInterface, hexagon);
            }
            else {
                processNeuronsOfInputHexagon<false>(cluster, inputInterface, hexagon);
            }
        }
    }
}

/**
 * @brief process cluster and train it be creating new synapses
 *
 * @param cluster pointer to cluster to process
 * @param hexagonId id of the hexagon to process
 * @param blockId id of the block within the hexagon
 * @param doTrain true to run trainging-process
 */
void
WorkerThread::processClusterForward(Cluster& cluster,
                                    const uint32_t hexagonId,
                                    const uint32_t blockId,
                                    const bool doTrain)
{
    Hanami::ErrorContainer error;
    Hexagon* hexagon = &cluster.hexagons[hexagonId];

    if (hexagon->header.isInputHexagon) {
        return;
    }

    if (doTrain) {
        processSynapses<true>(cluster, hexagon, blockId);
        if (hexagon->header.isOutputHexagon == false) {
            processNeurons<true>(cluster, hexagon, blockId);
        }
    }
    else {
        processSynapses<false>(cluster, hexagon, blockId);
        if (hexagon->header.isOutputHexagon == false) {
            processNeurons<false>(cluster, hexagon, blockId);
        }
    }
}

/**
 * @brief run the backpropagation over the core the cluster
 *
 * @param cluster pointer to cluster to process
 * @param hexagonId id of the hexagon to process
 * @param blockId id of the block within the hexagon
 */
void
WorkerThread::processClusterBackward(Cluster& cluster,
                                     const uint32_t hexagonId,
                                     const uint32_t blockId)
{
    Hanami::ErrorContainer error;
    Hexagon* hexagon = &cluster.hexagons[hexagonId];
    if (hexagon->header.isInputHexagon) {
        return;
    }

    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(cluster.attachedHost->synapseBlocks);
    backpropagateNeuron(hexagon, blockId);
    backpropagateConnections(hexagon, &cluster.hexagons[0], synapseBlocks, blockId);
}
