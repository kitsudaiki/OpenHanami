/**
 * @file        cluster_resize.h
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

#ifndef HANAMI_SECTION_UPDATE_H
#define HANAMI_SECTION_UPDATE_H

#include <core/cluster/cluster.h>
#include <core/cluster/objects.h>
#include <core/processing/cpu/cpu_host.h>
#include <core/processing/cuda/cuda_host.h>
#include <core/processing/logical_host.h>
#include <hanami_root.h>

/**
 * @brief search for an empty target-connection within a target-hexagon
 *
 * @param targetHexagon target-hexagon where to search
 * @param synapseBlockBuffer synapse-block-buffer to allocate new block,
 *                           if search-process was successful
 *
 * @return found empty connection, if seccessfule, else nullptr
 */
inline Connection*
searchTargetInHexagon(Hexagon* targetHexagon, ItemBuffer& synapseBlockBuffer)
{
    uint64_t i = 0;
    uint64_t j = 0;

    const uint64_t numberOfConnectionsBlocks = targetHexagon->connectionBlocks.size();
    if (numberOfConnectionsBlocks == 0) {
        return nullptr;
    }

    uint64_t pos = rand() % numberOfConnectionsBlocks;
    for (i = 0; i < numberOfConnectionsBlocks; ++i) {
        ConnectionBlock* connectionBlock = &targetHexagon->connectionBlocks[pos];
        uint64_t synapseSectionPos = targetHexagon->synapseBlockLinks[pos];

        for (j = 0; j < NUMBER_OF_SYNAPSESECTION; ++j) {
            if (connectionBlock->connections[j].origin.blockId == UNINIT_STATE_16) {
                // initialize a synapse-block if necessary
                if (synapseSectionPos == UNINIT_STATE_64) {
                    SynapseBlock block;
                    synapseSectionPos = synapseBlockBuffer.addNewItem(block);
                    if (synapseSectionPos == UNINIT_STATE_64) {
                        return nullptr;
                    }

                    targetHexagon->synapseBlockLinks[pos] = synapseSectionPos;
                }
                return &connectionBlock->connections[j];
            }
        }
        pos = (pos + 1) % numberOfConnectionsBlocks;
    }

    return nullptr;
}

/**
 * @brief resize the number of connection-blocks of a hexagon
 *
 * @param targetHexagon hexagon to resize
 */
inline void
resizeConnections(Hexagon* targetHexagon)
{
    const uint32_t dimXold = targetHexagon->header.dimX;

    targetHexagon->header.dimX++;

    // resize list
    targetHexagon->connectionBlocks.resize(targetHexagon->header.dimX);
    targetHexagon->synapseBlockLinks.resize(targetHexagon->header.dimX);

    // if there was no scaling in x-dimension, then no re-ordering necessary
    if (targetHexagon->header.dimX == dimXold) {
        return;
    }
    LOG_DEBUG("resized connection-Block: " + std::to_string(dimXold) + " -> "
              + std::to_string(targetHexagon->header.dimX));

    // update content of list for the new size
    targetHexagon->connectionBlocks[targetHexagon->header.dimX - 1] = ConnectionBlock();
    targetHexagon->synapseBlockLinks[targetHexagon->header.dimX - 1] = UNINIT_STATE_64;

    targetHexagon->neuronBlocks.resize(targetHexagon->header.dimX);
    targetHexagon->header.numberOfFreeSections += NUMBER_OF_SYNAPSESECTION;
}

/**
 * @brief allocate a new synapse-section
 *
 * @param cluster cluster to update
 * @param originLocation position of the soruce-neuron, which require the resize
 * @param lowerBound action-offset of the new section
 * @param potentialRange range of the potential, suppored by the section
 * @param synapseBlockBuffer synapse-block-buffer to allocate new blocks, if necessary
 *
 * @return true, if successful, else false
 */
inline bool
createNewSection(Cluster& cluster,
                 const SourceLocationPtr& sourceLocPtr,
                 const float lowerBound,
                 const float potentialRange,
                 ItemBuffer& synapseBlockBuffer)
{
    // get origin object
    SourceLocation sourceLoc = getSourceNeuron(sourceLocPtr, &cluster.hexagons[0]);
    if (sourceLoc.hexagon->header.isOutputHexagon) {
        return false;
    }

    // get target objects
    const uint32_t targetHexagonId
        = sourceLoc.hexagon->possibleHexagonTargetIds[rand() % NUMBER_OF_POSSIBLE_NEXT];
    Hexagon* targetHexagon = &cluster.hexagons[targetHexagonId];

    // get target-connection
    if (targetHexagon->header.numberOfFreeSections < NUMBER_OF_SYNAPSESECTION / 2) {
        resizeConnections(targetHexagon);
    }
    Connection* targetConnection = searchTargetInHexagon(targetHexagon, synapseBlockBuffer);
    if (targetConnection == nullptr) {
        Hanami::ErrorContainer error;
        error.addMessage("no target-section found, even there should be sill "
                         + std::to_string(targetHexagon->header.numberOfFreeSections)
                         + " available");
        LOG_ERROR(error);
        return false;
    }
    targetHexagon->header.numberOfFreeSections--;
    targetHexagon->wasResized = true;

    // initialize new connection
    targetConnection->origin = sourceLocPtr;
    targetConnection->lowerBound = lowerBound;
    targetConnection->potentialRange = potentialRange;
    targetConnection->origin.isInput = sourceLoc.hexagon->header.isInputHexagon;
    sourceLoc.neuron->inUse = 1;

    return true;
}

/**
 * @brief iterate over all neuron and run the resize-process, if necessary. This function is used
 *        in case of a cuda host, where the resize has to be done after the processing
 *
 * @param cluster cluster to resize
 *
 * @return true, if a resize was performed, else false. This is used to avoid unnecessary data-
 *         transfers to the gpu
 */
inline bool
updateCluster(Cluster& cluster)
{
    NeuronBlock* neuronBlock = nullptr;
    Neuron* neuron = nullptr;
    Hexagon* hexagon = nullptr;
    bool found = false;
    uint64_t hexagonId = 0;
    uint64_t blockId = 0;
    uint8_t sourceId = 0;

    // iterate over all neurons and add new synapse-section, if required
    for (hexagonId = 0; hexagonId < cluster.hexagons.size(); ++hexagonId) {
        hexagon = &cluster.hexagons[hexagonId];

        for (blockId = 0; blockId < hexagon->neuronBlocks.size(); ++blockId) {
            neuronBlock = &hexagon->neuronBlocks[blockId];

            for (sourceId = 0; sourceId < NEURONS_PER_NEURONBLOCK; ++sourceId) {
                neuron = &neuronBlock->neurons[sourceId];

                if (neuron->isNew > 0) {
                    found = true;
                    SourceLocationPtr originLocation;
                    originLocation.hexagonId = hexagonId;
                    originLocation.blockId = blockId;
                    originLocation.neuronId = sourceId;

                    createNewSection(cluster,
                                     originLocation,
                                     neuron->newLowerBound,
                                     neuron->potentialRange,
                                     cluster.attachedHost->synapseBlocks);

                    neuron->newLowerBound = 0.0f;
                    neuron->isNew = 0;
                }
            }
        }
    }
    return found;
}

#endif  // HANAMI_SECTION_UPDATE_H
