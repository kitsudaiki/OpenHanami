/**
 * @file        cpu_worker_thread.cpp
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

#include "cpu_worker_thread.h"

#include <core/cluster/objects.h>
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
CpuWorkerThread::CpuWorkerThread(CpuHost* host) : WorkerThread() { m_host = host; }

/**
 * @brief destructor
 */
CpuWorkerThread::~CpuWorkerThread() {}

/**
 * @brief handle trainging task
 *
 * @param task task to handle
 */
void
CpuWorkerThread::handleTrainForwardTask(WorkerTask task)
{
    Hexagon* hexagon = &task.cluster->hexagons[task.hexagonId];
    if (task.blockId == UNINIT_STATE_16) {
        // handle input-interface
        if (hexagon->inputInterface != nullptr) {
            processInput(*task.cluster, hexagon, true);
        }

        // handle special-case that there are no neuron-blocks to process
        if (hexagon->neuronBlocks.size() == 0) {
            // in case of the last hexagon
            if (task.hexagonId == task.cluster->hexagons.size() - 1) {
                task.cluster->updateClusterState(task);
                return;
            }

            // in case of a normal hexagon
            WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId + 1;
            newTask.blockId = UNINIT_STATE_16;
            newTask.mode = task.mode;
            task.cluster->hexagons[newTask.hexagonId].attachedHost->addWorkerTaskToQueue(newTask);
            return;
        }

        processConnectionBlocksForward(*task.cluster, hexagon);

        // share neuron-blocks to process
        for (uint32_t i = 0; i < hexagon->neuronBlocks.size(); i++) {
            WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId;
            newTask.blockId = i;
            newTask.mode = task.mode;
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
            processNeuronsOfOutputHexagon<true>(hexagon, rand());
        }

        if (task.hexagonId == task.cluster->hexagons.size() - 1) {
            updateCluster(*task.cluster, hexagon);
            task.cluster->updateClusterState(task);
        }
        else {
            WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId + 1;
            newTask.blockId = UNINIT_STATE_16;
            newTask.mode = task.mode;
            task.cluster->hexagons[newTask.hexagonId].attachedHost->addWorkerTaskToQueue(newTask);
        }
    }
}

/**
 * @brief handle backpropagation task
 *
 * @param task task to handle
 */
void
CpuWorkerThread::handleTrainBackwardTask(WorkerTask task)
{
    if (task.blockId == UNINIT_STATE_16) {
        Hexagon* hexagon = &task.cluster->hexagons[task.hexagonId];

        // handle output-interface
        if (hexagon->outputInterface != nullptr) {
            backpropagateOutput(hexagon);
        }

        // handle special-case that there are no neuron-blocks to process
        if (hexagon->neuronBlocks.size() == 0) {
            if (task.hexagonId == 0) {
                task.cluster->updateClusterState(task);
                return;
            }

            WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId - 1;
            newTask.blockId = UNINIT_STATE_16;
            newTask.mode = task.mode;
            task.cluster->hexagons[newTask.hexagonId].attachedHost->addWorkerTaskToQueue(newTask);
            return;
        }

        // share neuron-blocks to process
        for (uint32_t i = 0; i < hexagon->neuronBlocks.size(); i++) {
            WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId;
            newTask.blockId = i;
            newTask.mode = task.mode;
            m_host->addWorkerTaskToQueue(newTask);
        }

        return;
    }

    // run backpropation
    processClusterBackward(*task.cluster, task.hexagonId, task.blockId);
    if (task.cluster->incrementAndCompare(
            task.cluster->hexagons[task.hexagonId].neuronBlocks.size()))
    {
        processConnectionBlocksBackward(*task.cluster, &task.cluster->hexagons[task.hexagonId]);

        if (task.hexagonId == 0) {
            task.cluster->updateClusterState(task);
        }
        else {
            WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId - 1;
            newTask.blockId = UNINIT_STATE_16;
            newTask.mode = task.mode;
            task.cluster->hexagons[newTask.hexagonId].attachedHost->addWorkerTaskToQueue(newTask);
        }
    }
}

/**
 * @brief handle process task
 *
 * @param task task to handle
 */
void
CpuWorkerThread::handleProcessTask(const WorkerTask task)
{
    Hexagon* hexagon = &task.cluster->hexagons[task.hexagonId];

    if (task.blockId == UNINIT_STATE_16) {
        if (hexagon->inputInterface != nullptr) {
            processInput(*task.cluster, hexagon, true);
        }

        // handle special-case that there are no neuron-blocks to process
        if (hexagon->neuronBlocks.size() == 0) {
            if (task.hexagonId == task.cluster->hexagons.size() - 1) {
                task.cluster->updateClusterState(task);
                return;
            }

            WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId + 1;
            newTask.blockId = UNINIT_STATE_16;
            newTask.mode = task.mode;
            task.cluster->hexagons[newTask.hexagonId].attachedHost->addWorkerTaskToQueue(newTask);
            return;
        }

        processConnectionBlocksForward(*task.cluster, hexagon);

        // share neuron-blocks to process
        for (uint32_t i = 0; i < hexagon->neuronBlocks.size(); i++) {
            WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId;
            newTask.blockId = i;
            newTask.mode = task.mode;
            m_host->addWorkerTaskToQueue(newTask);
        }
        return;
    }

    // run backpropation
    processClusterForward(*task.cluster, task.hexagonId, task.blockId, false);
    if (task.cluster->incrementAndCompare(
            task.cluster->hexagons[task.hexagonId].neuronBlocks.size()))
    {
        if (hexagon->outputInterface != nullptr) {
            processNeuronsOfOutputHexagon<false>(hexagon, rand());
        }

        if (task.hexagonId == task.cluster->hexagons.size() - 1) {
            handleClientOutput(*task.cluster);
            task.cluster->updateClusterState(task);
        }
        else {
            WorkerTask newTask;
            newTask.cluster = task.cluster;
            newTask.hexagonId = task.hexagonId + 1;
            newTask.blockId = UNINIT_STATE_16;
            newTask.mode = task.mode;
            task.cluster->hexagons[newTask.hexagonId].attachedHost->addWorkerTaskToQueue(newTask);
        }
    }
}

/**
 * @brief handle reduction task
 *
 * @param task task to handle
 */
void
CpuWorkerThread::handleReductionTask(const WorkerTask task)
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
