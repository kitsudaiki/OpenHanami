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

#include <core/processing/cluster_io_functions.h>
#include <core/processing/cluster_resize.h>
#include <core/processing/cpu/backpropagation.h>
#include <core/processing/cpu/processing.h>
#include <core/processing/cpu/reduction.h>
#include <core/processing/logical_host.h>
#include <core/processing/objects.h>

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
    Brick* brick = &task.cluster->bricks[task.brickId];
    if (task.blockId == UNINIT_STATE_16) {
        // handle input-interface
        if (brick->inputInterface != nullptr) {
            WorkerThread::handleInputForward(*task.cluster, brick->inputInterface, true);
        }

        // handle special-case that there are no neuron-blocks to process
        if (brick->neuronBlocks.size() == 0) {
            if (task.brickId == task.cluster->bricks.size() - 1) {
                updateCluster(*task.cluster);
                task.cluster->updateClusterState();
                return;
            }

            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.brickId = task.brickId + 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
            return;
        }

        // share neuron-blocks to process
        for (uint32_t i = 0; i < brick->neuronBlocks.size(); i++) {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.brickId = task.brickId;
            newTask.blockId = i;
            m_host->addWorkerTaskToQueue(newTask);
        }
        return;
    }

    // run backpropation
    processClusterForward(*task.cluster, task.brickId, task.blockId, true);
    if (task.cluster->incrementAndCompare(task.cluster->bricks[task.brickId].neuronBlocks.size())) {
        if (brick->outputInterface != nullptr) {
            processNeuronsOfOutputBrick<true>(
                task.cluster->bricks, brick->outputInterface, task.brickId, rand());
        }

        if (task.brickId == task.cluster->bricks.size() - 1) {
            updateCluster(*task.cluster);
            task.cluster->updateClusterState();
        }
        else {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.brickId = task.brickId + 1;
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
        Brick* brick = &task.cluster->bricks[task.brickId];

        // handle output-interface
        if (brick->outputInterface != nullptr) {
            backpropagateOutput(task.cluster->bricks,
                                brick->outputInterface,
                                &task.cluster->clusterHeader.settings,
                                task.brickId);
        }

        // handle special-case that there are no neuron-blocks to process
        if (brick->neuronBlocks.size() == 0) {
            if (task.brickId == 0) {
                task.cluster->updateClusterState();
                return;
            }

            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.brickId = task.brickId - 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
            return;
        }

        // share neuron-blocks to process
        for (uint32_t i = 0; i < brick->neuronBlocks.size(); i++) {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.brickId = task.brickId;
            newTask.blockId = i;
            m_host->addWorkerTaskToQueue(newTask);
        }
        return;
    }

    // run backpropation
    processClusterBackward(*task.cluster, task.brickId, task.blockId);
    if (task.cluster->incrementAndCompare(task.cluster->bricks[task.brickId].neuronBlocks.size())) {
        if (task.brickId == 0) {
            task.cluster->updateClusterState();
        }
        else {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.brickId = task.brickId - 1;
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
    /*reduceCluster(*task.cluster, task.brickId, task.blockId);
    if (task.cluster->incrementAndCompare(task.cluster->bricks[task.brickId].neuronBlocks.size())) {
        if (task.brickId == task.cluster->bricks.size() - 1) {
            task.cluster->updateClusterState();
        }
        else {
            m_host->addBrickToTaskQueue(task.cluster, task.brickId + 1);
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
    Brick* brick = &task.cluster->bricks[task.brickId];

    if (task.blockId == UNINIT_STATE_16) {
        // handle input-interface
        if (brick->inputInterface != nullptr) {
            WorkerThread::handleInputForward(*task.cluster, brick->inputInterface, true);
        }

        // handle special-case that there are no neuron-blocks to process
        if (brick->neuronBlocks.size() == 0) {
            if (task.brickId == task.cluster->bricks.size() - 1) {
                task.cluster->updateClusterState();
                return;
            }

            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.brickId = task.brickId + 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
            return;
        }

        // share neuron-blocks to process
        for (uint32_t i = 0; i < brick->neuronBlocks.size(); i++) {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.brickId = task.brickId;
            newTask.blockId = i;
            m_host->addWorkerTaskToQueue(newTask);
        }
        return;
    }

    // run backpropation
    processClusterForward(*task.cluster, task.brickId, task.blockId, true);
    if (task.cluster->incrementAndCompare(task.cluster->bricks[task.brickId].neuronBlocks.size())) {
        if (brick->outputInterface != nullptr) {
            processNeuronsOfOutputBrick<false>(
                task.cluster->bricks, brick->outputInterface, task.brickId, rand());
        }

        if (task.brickId == task.cluster->bricks.size() - 1) {
            handleClientOutput(*task.cluster);
            task.cluster->updateClusterState();
        }
        else {
            CpuHost::WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.brickId = task.brickId + 1;
            newTask.blockId = UNINIT_STATE_16;
            m_host->addWorkerTaskToQueue(newTask);
        }
    }
}

/**
 * @brief handle input-bricks by applying input-values to the input-neurons
 *
 * @param cluster pointer to cluster to process
 * @param doTrain true to run trainging-process
 */
void
WorkerThread::handleInputForward(Cluster& cluster,
                                 InputInterface* inputInterface,
                                 const bool doTrain)
{
    Brick* brick = nullptr;
    const uint32_t numberOfBricks = cluster.bricks.size();

    // process input-bricks
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId) {
        brick = &cluster.bricks[brickId];

        if (brick->isInputBrick) {
            if (doTrain) {
                processNeuronsOfInputBrick<true>(cluster, inputInterface, brick);
            }
            else {
                processNeuronsOfInputBrick<false>(cluster, inputInterface, brick);
            }
        }
    }
}

/**
 * @brief process cluster and train it be creating new synapses
 *
 * @param cluster pointer to cluster to process
 * @param brickId id of the brick to process
 * @param blockId id of the block within the brick
 * @param doTrain true to run trainging-process
 */
void
WorkerThread::processClusterForward(Cluster& cluster,
                                    const uint32_t brickId,
                                    const uint32_t blockId,
                                    const bool doTrain)
{
    Hanami::ErrorContainer error;
    Brick* brick = &cluster.bricks[brickId];

    if (brick->isInputBrick) {
        return;
    }

    if (doTrain) {
        processSynapses<true>(cluster, brick, blockId);
        if (brick->isOutputBrick == false) {
            processNeurons<true>(cluster, brick, blockId);
        }
    }
    else {
        processSynapses<false>(cluster, brick, blockId);
        if (brick->isOutputBrick == false) {
            processNeurons<false>(cluster, brick, blockId);
        }
    }
}

/**
 * @brief run the backpropagation over the core the cluster
 *
 * @param cluster pointer to cluster to process
 * @param brickId id of the brick to process
 * @param blockId id of the block within the brick
 */
void
WorkerThread::processClusterBackward(Cluster& cluster,
                                     const uint32_t brickId,
                                     const uint32_t blockId)
{
    Hanami::ErrorContainer error;
    Brick* brick = &cluster.bricks[brickId];
    if (brick->isInputBrick) {
        return;
    }

    SynapseBlock* synapseBlocks = getItemData<SynapseBlock>(cluster.attachedHost->synapseBlocks);
    backpropagateNeuron(brick, blockId);
    backpropagateConnections(brick, &cluster.bricks[0], synapseBlocks, blockId);
}
