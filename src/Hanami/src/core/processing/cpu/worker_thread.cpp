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
WorkerThread::handleTrainForwardTask(const CpuHost::WorkerTask task)
{
    if (task.brickId == UNINIT_STATE_32) {
        WorkerThread::handleInputForward(*task.cluster, true);
        m_host->addBrickToTaskQueue(task.cluster, 0);
        return;
    }
    else {
        processClusterForward(*task.cluster, task.brickId, task.blockId, true);
        if (task.cluster->incrementAndCompare(
                task.cluster->bricks[task.brickId].numberOfNeuronBlocks))
        {
            if (task.brickId == task.cluster->clusterHeader->bricks.count - 1) {
                updateCluster(*task.cluster);
                task.cluster->updateClusterState();
            }
            else {
                m_host->addBrickToTaskQueue(task.cluster, task.brickId + 1);
                return;
            }
        }
    }
}

/**
 * @brief handle backpropagation task
 *
 * @param task task to handle
 */
void
WorkerThread::handleTrainBackwardTask(const CpuHost::WorkerTask task)
{
    if (task.brickId == UNINIT_STATE_32) {
        WorkerThread::handleOutputBackward(*task.cluster);
        m_host->addBrickToTaskQueue(task.cluster, task.cluster->clusterHeader->bricks.count - 1);
        return;
    }
    else {
        processClusterBackward(*task.cluster, task.brickId, task.blockId);
        if (task.cluster->incrementAndCompare(
                task.cluster->bricks[task.brickId].numberOfNeuronBlocks))
        {
            if (task.brickId == 0) {
                task.cluster->updateClusterState();
            }
            else {
                m_host->addBrickToTaskQueue(task.cluster, task.brickId - 1);
                return;
            }
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
    reduceCluster(*task.cluster, task.brickId, task.blockId);
    if (task.cluster->incrementAndCompare(task.cluster->bricks[task.brickId].numberOfNeuronBlocks))
    {
        if (task.brickId == task.cluster->clusterHeader->bricks.count - 1) {
            task.cluster->updateClusterState();
        }
        else {
            m_host->addBrickToTaskQueue(task.cluster, task.brickId + 1);
            return;
        }
    }
}

/**
 * @brief handle process task
 *
 * @param task task to handle
 */
void
WorkerThread::handleProcessTask(const CpuHost::WorkerTask task)
{
    if (task.brickId == UNINIT_STATE_32) {
        WorkerThread::handleInputForward(*task.cluster, false);
        m_host->addBrickToTaskQueue(task.cluster, 0);
        return;
    }
    else {
        processClusterForward(*task.cluster, task.brickId, task.blockId, false);
        if (task.cluster->incrementAndCompare(
                task.cluster->bricks[task.brickId].numberOfNeuronBlocks))
        {
            if (task.brickId == task.cluster->clusterHeader->bricks.count - 1) {
                handleClientOutput(*task.cluster);
                task.cluster->updateClusterState();
            }
            else {
                m_host->addBrickToTaskQueue(task.cluster, task.brickId + 1);
                return;
            }
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
WorkerThread::handleInputForward(Cluster& cluster, const bool doTrain)
{
    Brick* brick = nullptr;
    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;

    // process input-bricks
    for (uint32_t brickId = 0; brickId < numberOfBricks; ++brickId) {
        brick = &cluster.bricks[brickId];

        if (brick->isInputBrick) {
            if (doTrain) {
                processNeuronsOfInputBrick<true>(cluster, brick);
            }
            else {
                processNeuronsOfInputBrick<false>(cluster, brick);
            }
        }
    }
}

/**
 * @brief run backpropagation over the output-bricks
 *
 * @param cluster pointer to cluster to process
 *
 * @return false, if threshold of the full backpropagion is not reached
 */
bool
WorkerThread::handleOutputBackward(Cluster& cluster)
{
    Brick* brick = nullptr;
    const uint32_t numberOfBricks = cluster.clusterHeader->bricks.count;

    for (int32_t brickId = numberOfBricks - 1; brickId >= 0; --brickId) {
        brick = &cluster.bricks[brickId];

        if (brick->isOutputBrick) {
            if (backpropagateOutput(brick,
                                    cluster.neuronBlocks,
                                    cluster.tempNeuronBlocks,
                                    cluster.outputValues,
                                    cluster.expectedValues,
                                    &cluster.clusterHeader->settings)
                == false)
            {
                return false;
            }
        }
    }

    return true;
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
    float* outputValues = cluster.outputValues;
    NeuronBlock* neuronBlocks = cluster.neuronBlocks;
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

    if (brick->isOutputBrick) {
        processNeuronsOfOutputBrick(brick, outputValues, neuronBlocks, blockId);
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
    backpropagateNeuron(brick, cluster.neuronBlocks, cluster.tempNeuronBlocks, blockId);
    backpropagateConnections(
        brick, cluster.neuronBlocks, cluster.tempNeuronBlocks, synapseBlocks, blockId);
}
