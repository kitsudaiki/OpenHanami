/**
 * @file        graph_train_forward_state.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2019 Tobias Anker
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

#include "graph_train_forward_state.h"

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
GraphTrainForward_State::GraphTrainForward_State(Cluster* cluster) { m_cluster = cluster; }

/**
 * @brief destructor
 */
GraphTrainForward_State::~GraphTrainForward_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
GraphTrainForward_State::processEvent()
{
    /*Task* actualTask = m_cluster->getActualTask();

    float lastVal = actualTask->inputData[actualTask->actualCycle + 1];

    float actualVal = 0.0f;
    float newVal = 0.0f;
    uint64_t pos = 0;

    // set input
    InputNode* inputNodes = m_cluster->inputSegments.begin()->second->inputs;
    OutputNode* outputNodes = m_cluster->outputSegments.begin()->second->outputs;
    uint64_t i = actualTask->actualCycle + 1;

    while(i < actualTask->actualCycle + 1 + 365)
    {
        // open-part
        actualVal = actualTask->inputData[i * 2];
        newVal = (100.0f / actualVal) * lastVal;

        if(newVal > 100.0f)
        {
            inputNodes[pos * 2].weight = newVal - 100.0f;
            if(inputNodes[pos * 2].weight > 2.0f) {
                inputNodes[pos * 2].weight = 2.0f;
            }
            inputNodes[pos * 2 + 1].weight = 0.0f;
        }
        else
        {
            inputNodes[pos * 2].weight = 0.0f;
            inputNodes[pos * 2 + 1].weight = 100.0f - newVal;
            if(inputNodes[pos * 2+1].weight > 2.0f) {
                inputNodes[pos * 2+1].weight = 2.0f;
            }
        }

        lastVal = actualVal;
        pos++;

        // close-part
        actualVal = actualTask->inputData[i * 2 + 1];
        newVal = (100.0f / actualVal) * lastVal;

        if(newVal > 100.0f)
        {
            inputNodes[pos * 2].weight = newVal - 100.0f;
            if(inputNodes[pos * 2].weight > 2.0f) {
                inputNodes[pos * 2].weight = 2.0f;
            }
            inputNodes[pos * 2 + 1].weight = 0.0f;
        }
        else
        {
            inputNodes[pos * 2].weight = 0.0f;
            inputNodes[pos * 2 + 1].weight = 100.0f - newVal;
            if(inputNodes[pos * 2+1].weight > 2.0f) {
                inputNodes[pos * 2+1].weight = 2.0f;
            }
        }

        lastVal = actualVal;
        pos++;

        i++;
    }

    // set exprected output
    actualVal = actualTask->inputData[i * 2];
    if(actualVal > lastVal)
    {
        outputNodes[0].shouldValue = 1.0f;
        outputNodes[1].shouldValue = 0.0f;
    }
    else
    {
        outputNodes[0].shouldValue = 0.0f;
        outputNodes[1].shouldValue = 1.0f;
    }
    lastVal = actualVal;

    actualVal = actualTask->inputData[i * 2 + 1];
    if(actualVal > lastVal)
    {
        outputNodes[2].shouldValue = 1.0f;
        outputNodes[3].shouldValue = 0.0f;
    }
    else
    {
        outputNodes[2].shouldValue = 0.0f;
        outputNodes[3].shouldValue = 1.0f;
    }

    m_cluster->mode = Cluster::TRAIN_FORWARD_MODE;
    m_cluster->startForwardCycle();*/

    return true;
}
