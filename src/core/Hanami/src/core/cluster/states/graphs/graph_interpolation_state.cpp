/**
 * @file        graph_interpolation_state.cpp
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

#include "graph_interpolation_state.h"

#include <core/segments/core_segment/core_segment.h>
#include <core/segments/input_segment/input_segment.h>
#include <core/segments/output_segment/output_segment.h>

#include <core/cluster/cluster.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
GraphInterpolation_State::GraphInterpolation_State(Cluster* cluster)
{
    m_cluster = cluster;
}

/**
 * @brief destructor
 */
GraphInterpolation_State::~GraphInterpolation_State() {}

/**
 * @brief prcess event
 *
 * @return alway true
 */
bool
GraphInterpolation_State::processEvent()
{

    Task* actualTask = m_cluster->getActualTask();
    /*if(actualTask->isInit)
    {
        // set input
        InputNode* inputNodes = m_cluster->inputSegments[0]->inputs;
        for(uint64_t i = 4; i < 4*365; i++) {
            inputNodes[i - 4].weight = inputNodes[i].weight;
        }

        OutputNode* outputNodes = m_cluster->outputSegments[0]->outputs;
        inputNodes[4*365 - 4].weight = outputNodes[0].outputWeight * outputNodes[0].maxWeight;
        inputNodes[4*365 - 3].weight = outputNodes[1].outputWeight * outputNodes[1].maxWeight;
        inputNodes[4*365 - 2].weight = outputNodes[2].outputWeight * outputNodes[2].maxWeight;
        inputNodes[4*365 - 1].weight = outputNodes[3].outputWeight * outputNodes[3].maxWeight;

        //std::cout<<"x: "<<(outputNodes[0].outputWeight / 5.0f)<<"   y: "<<(outputNodes[1].outputWeight / 5.0f)<<std::endl;
        const float val1 = outputNodes[0].outputWeight * outputNodes[0].maxWeight;
        const float val2 = outputNodes[1].outputWeight * outputNodes[1].maxWeight;
        const float val3 = outputNodes[2].outputWeight * outputNodes[2].maxWeight;
        const float val4 = outputNodes[3].outputWeight * outputNodes[3].maxWeight;

        std::cout<<"o_up: "<<val1<<"\t   o_down: "<<val2<<"   c_up: "<<val3<<"\t   c_down: "<<val4<<std::endl;

        if(val1 > val2) {
            std::cout<<"x: "<<(val1 - val2)<<"\t   y: "<<0.0f<<std::endl;
        } else {
            std::cout<<"x: "<<0.0f<<"\t   y: "<<(val2 - val1)<<std::endl;
        }

        std::cout<<std::endl;
    }
    else
    {
        const uint64_t numberInputCycles = actualTask->getIntVal("number_of_inputs_per_cycle");
        const float* data = &actualTask->inputData[2 * numberInputCycles - 2*366];

        //std::cout<<"------------------------actualTask->numberOfInputsPerCycle "<<actualTask->numberOfInputsPerCycle<<std::endl;


        float lastVal = data[actualTask->actualCycle + 1];
        float actualVal = 0.0f;
        float newVal = 0.0f;
        uint64_t pos = 0;

        // set input
        InputNode* inputNodes = m_cluster->inputSegments[0]->inputs;
        for(uint64_t i = actualTask->actualCycle + 1;
            i < actualTask->actualCycle + 1 + 365;
            i++)
        {
            // open-part
            actualVal = data[i * 2];
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
            actualVal = data[i * 2 + 1];
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
        }
    }

    actualTask->isInit = true;

    m_cluster->mode = Cluster::NORMAL_MODE;
    m_cluster->startForwardCycle();*/

    return true;
}
