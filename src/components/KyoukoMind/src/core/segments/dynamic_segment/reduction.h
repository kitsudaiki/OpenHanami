/**
 * @file        create_reduce.h
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

#ifndef KYOUKOMIND_CREATE_REDUCE_H
#define KYOUKOMIND_CREATE_REDUCE_H

#include <common.h>

#include <kyouko_root.h>
#include <core/segments/brick.h>

#include "objects.h"
#include "dynamic_segment.h"

/**
 * @brief reduce synapses of a specific section
 */
inline bool
reduceSynapses(DynamicSegment &segment,
               SynapseSection &section)
{
    bool foundEnd = false;

    /*if(section.next != UNINIT_STATE_32)
    {
        // delete if sections is empty
        foundEnd = true;
        if(reduceSynapses(segment, segment.synapseSections[section.next]) == false)
        {
            segment.segmentData.deleteItem(section.next);
            section.next = UNINIT_STATE_32;
            foundEnd = false;
        }
    }

    Synapse* synapse = nullptr;

    // iterate over all synapses in synapse-section
    for(int32_t pos = SYNAPSES_PER_SYNAPSESECTION - 1;
        pos >= 0;
        pos--)
    {
        // skip not connected synapses
        synapse = &section.synapses[pos];

        synapse->activeCounter -= synapse->activeCounter < 100;
        if(synapse->activeCounter < 5) {
            synapse->targetNeuronId = UNINIT_STATE_16;
        }

        if(synapse->targetNeuronId != UNINIT_STATE_16) {
            foundEnd = true;
        }
    }*/

    return foundEnd;
}

/**
 * @brief reduce all synapses within the segment and delete them, if the reach a deletion-border
 */
inline void
reduceNeurons(DynamicSegment &segment)
{
    SynapseSection* section = nullptr;
    DynamicNeuron* sourceNeuron = nullptr;

    /*for(uint32_t neuronId = 0;
        neuronId < segment.segmentHeader->neuronSections.count;
        neuronId++)
    {
        //sourceNeuron = &segment.neurons[neuronId];
        if(sourceNeuron->targetSectionId == UNINIT_STATE_32) {
            continue;
        }

        // set start-values
        section = &segment.synapseSections[sourceNeuron->targetSectionId];

        // delete if sections is empty
        if(reduceSynapses(segment, *section) == false)
        {
            segment.segmentData.deleteItem(sourceNeuron->targetSectionId);
            sourceNeuron->targetSectionId = UNINIT_STATE_32;
        }
    }*/
}

#endif // KYOUKOMIND_CREATE_REDUCE_H
