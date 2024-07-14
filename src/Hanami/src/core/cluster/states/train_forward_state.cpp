/**
 * @file        train_forward_state.cpp
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

#include "train_forward_state.h"

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
TrainForward_State::TrainForward_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
TrainForward_State::~TrainForward_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
TrainForward_State::processEvent()
{
    Hanami::ErrorContainer error;
    Task* actualTask = m_cluster->getCurrentTask();
    TrainInfo* info = &std::get<TrainInfo>(actualTask->info);

    for (auto& [hexagonName, input] : info->inputs) {
        InputInterface* inputInterface = &m_cluster->inputInterfaces[hexagonName];
        if (getDataFromDataSet(inputInterface->ioBuffer, input, info->currentCycle, error) != OK) {
            return false;
        }
        uint64_t counter = 0;
        for (const float val : inputInterface->ioBuffer) {
            inputInterface->inputNeurons[counter].value = val;
            counter++;
        }
    }

    for (auto& [hexagonName, output] : info->outputs) {
        OutputInterface* outputInterface = &m_cluster->outputInterfaces[hexagonName];
        if (getDataFromDataSet(outputInterface->ioBuffer, output, info->currentCycle, error) != OK)
        {
            return false;
        }
        uint64_t counter = 0;
        for (const float val : outputInterface->ioBuffer) {
            outputInterface->outputNeurons[counter].exprectedVal = val;
            counter++;
        }
    }

    m_cluster->mode = ClusterProcessingMode::TRAIN_FORWARD_MODE;
    m_cluster->startForwardCycle();

    return true;
}
