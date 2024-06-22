/**
 * @file        io_interface.cpp
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

#include "io_interface.h"

#include <core/cluster/cluster.h>
#include <core/cluster/cluster_init.h>
#include <core/processing/logical_host.h>

/**
 * @brief constructor
 */
IO_Interface::IO_Interface() {}

/**
 * @brief destructor
 */
IO_Interface::~IO_Interface() {}

/**
 * @brief serialize a cluster into an unterlying target
 *
 * @param cluster cluster to serialize
 * @param error reference for error-output
 *
 * @return OK-status, if successful, else ERROR-status
 */
ReturnStatus
IO_Interface::serialize(const Cluster& cluster, Hanami::ErrorContainer& error)
{
    const uint64_t totalClusterSize = getClusterSize(cluster);
    initLocalBuffer(totalClusterSize);
    if (initializeTarget(totalClusterSize, error) == false) {
        error.addMessage("Failed to initialize target to serialize cluster");
        return ERROR;
    }

    // write cluster-header to buffer
    if (addObjectToLocalBuffer(&cluster.clusterHeader, error) == false) {
        return ERROR;
    }

    // write number of bricks to buffer
    const uint64_t numberOfBricks = cluster.bricks.size();
    if (addObjectToLocalBuffer(&numberOfBricks, error) == false) {
        return ERROR;
    }

    // write bricks to buffer
    for (const Brick& brick : cluster.bricks) {
        const ReturnStatus ret = serialize(brick, error);
        if (ret != OK) {
            return ret;
        }
    }

    // write remaining data in the cache
    if (writeFromLocalBuffer(m_localBuffer, error) == false) {
        return ERROR;
    }

    return OK;
}

/**
 * @brief deserialize cluster from the unterlying target
 *
 * @param cluster target cluster-object
 * @param totalSize total number of bytes to initialize the local buffer
 * @param error reference for error-output
 *
 * @return return-status based on the result of the process
 */
ReturnStatus
IO_Interface::deserialize(Cluster& cluster, const uint64_t totalSize, Hanami::ErrorContainer& error)
{
    uint64_t positionPtr = 0;
    uint64_t numberOfBricks = 0;
    ReturnStatus ret = OK;

    initLocalBuffer(totalSize);

    // clear old data from the cluster
    cluster.bricks.clear();
    cluster.inputInterfaces.clear();
    cluster.outputInterfaces.clear();

    // read cluster-information
    ret = getObjectFromLocalBuffer(positionPtr, &cluster.clusterHeader, error);
    if (ret != OK) {
        return ret;
    }
    ret = getObjectFromLocalBuffer(positionPtr, &numberOfBricks, error);
    if (ret != OK) {
        return ret;
    }

    // read bricks
    cluster.bricks.resize(numberOfBricks);
    for (uint64_t i = 0; i < numberOfBricks; i++) {
        cluster.bricks[i].cluster = &cluster;
        const ReturnStatus ret = deserialize(cluster.bricks[i], positionPtr, error);
        if (ret != OK) {
            return ret;
        }
    }

    // re-initialize neighbor-list and target-list
    connectAllBricks(&cluster);
    initializeTargetBrickList(&cluster);

    return OK;
}

/**
 * @brief initialize the local buffer
 *
 * @param totalSize number of bytes for the complete target for checks of the upper bound
 */
void
IO_Interface::initLocalBuffer(const uint64_t totalSize)
{
    memset(m_localBuffer.cache, 0, LOCAL_BUFFER_SIZE);
    m_localBuffer.totalSize = totalSize;
    m_localBuffer.startPos = 0;
    m_localBuffer.size = 0;
}

/**
 * @brief calculate the number of bytes necessary to serialize a specific cluster
 *
 * @param cluster cluster, of which the necessary bytes should be calculated
 *
 * @return number of bytes for the cluster
 */
uint64_t
IO_Interface::getClusterSize(const Cluster& cluster) const
{
    uint64_t size = 0;

    size += sizeof(ClusterHeader);
    size += sizeof(uint64_t);

    for (const Brick& brick : cluster.bricks) {
        size += getBrickSize(brick);
    }

    return size;
}

/**
 * @brief calculate the number of bytes necessary to serialize a specific brick
 *
 * @param brick brick, of which the necessary bytes should be calculated
 *
 * @return number of bytes for the brick
 */
uint64_t
IO_Interface::getBrickSize(const Brick& brick) const
{
    uint64_t size = 0;

    size += sizeof(BrickEntry);
    size += brick.neuronBlocks.size() * sizeof(NeuronBlock);

    const uint64_t numberOfConnections = brick.connectionBlocks.size();
    size += numberOfConnections * sizeof(ConnectionBlock);
    size += numberOfConnections * sizeof(SynapseBlock);

    if (brick.inputInterface != nullptr) {
        size += sizeof(InputEntry);
        size += brick.inputInterface->inputNeurons.size() * sizeof(InputNeuron);
    }

    if (brick.outputInterface != nullptr) {
        size += sizeof(OutputEntry);
        size += brick.outputInterface->outputNeurons.size() * sizeof(OutputNeuron);
    }

    return size;
}

/**
 * @brief serialize a single brick
 *
 * @param brick brick, which should be serialized
 * @param error reference for error-output
 *
 * @return OK-status, if successful, else ERROR-status
 */
ReturnStatus
IO_Interface::serialize(const Brick& brick, Hanami::ErrorContainer& error)
{
    // brick-entry
    BrickEntry brickEntry = createBrickEntry(brick);
    if (addObjectToLocalBuffer(&brickEntry, error) == false) {
        return ERROR;
    }

    // neuron-blocks
    for (const NeuronBlock& neuronBlock : brick.neuronBlocks) {
        if (addObjectToLocalBuffer(&neuronBlock, error) == false) {
            return ERROR;
        }
    }

    // connection-blocks and synapse-blocks
    SynapseBlock* synapseBlocks
        = Hanami::getItemData<SynapseBlock>(brick.cluster->attachedHost->synapseBlocks);
    for (const ConnectionBlock& connectionBlock : brick.connectionBlocks) {
        if (addObjectToLocalBuffer(&connectionBlock, error) == false) {
            return ERROR;
        }
        SynapseBlock* synapseBlock = &synapseBlocks[connectionBlock.targetSynapseBlockPos];
        if (addObjectToLocalBuffer(synapseBlock, error) == false) {
            return ERROR;
        }
    }

    // input
    if (brick.inputInterface != nullptr) {
        // create input-entry and write it to the buffer
        InputEntry inputEntry;
        if (inputEntry.name.setName(brick.inputInterface->name) == false) {
            return INVALID_INPUT;
        }
        inputEntry.numberOfInputs = brick.inputInterface->inputNeurons.size();
        inputEntry.targetBrickId = brick.header.brickId;
        if (addObjectToLocalBuffer(&inputEntry, error) == false) {
            return ERROR;
        }

        // write input-neurons to buffer
        for (const InputNeuron& inputNeuron : brick.inputInterface->inputNeurons) {
            if (addObjectToLocalBuffer(&inputNeuron, error) == false) {
                return ERROR;
            }
        }
    }

    // output
    if (brick.outputInterface != nullptr) {
        // create output-entry and write it to the buffer
        OutputEntry outputEntry;
        if (outputEntry.name.setName(brick.outputInterface->name) == false) {
            return INVALID_INPUT;
        }
        outputEntry.numberOfOutputs = brick.outputInterface->outputNeurons.size();
        outputEntry.targetBrickId = brick.header.brickId;
        if (addObjectToLocalBuffer(&outputEntry, error) == false) {
            return ERROR;
        }

        // write output-neurons to buffer
        for (const OutputNeuron& outputNeuron : brick.outputInterface->outputNeurons) {
            if (addObjectToLocalBuffer(&outputNeuron, error) == false) {
                return ERROR;
            }
        }
    }

    return OK;
}
/**
 * @brief IO_Interface::deserialize
 *
 * @param brick target-brick for the deserialied data
 * @param positionPtr referece to track current byte-position
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
ReturnStatus
IO_Interface::deserialize(Brick& brick, uint64_t& positionPtr, Hanami::ErrorContainer& error)
{
    const uint64_t positionOffset = positionPtr;
    ReturnStatus ret = OK;

    // brick-entry
    BrickEntry brickEntry;
    ret = getObjectFromLocalBuffer(positionPtr, &brickEntry, error);
    if (ret != OK) {
        return ret;
    }
    if (checkBrickEntry(brickEntry) == false) {
        error.addMessage("Input-data invalid: Brick-check failed.");
        return INVALID_INPUT;
    }

    brick.header = brickEntry.header;

    if (brickEntry.neuronBlocksPos != 0) {
        // check current position
        if (positionPtr - positionOffset != brickEntry.neuronBlocksPos) {
            error.addMessage("Input-data invalid");
            return INVALID_INPUT;
        }

        // neuron-blocks
        brick.neuronBlocks.clear();
        const uint64_t numberOfNeuronBlocks = brickEntry.numberOfNeuronBytes / sizeof(NeuronBlock);
        brick.neuronBlocks.resize(numberOfNeuronBlocks);
        brick.tempNeuronBlocks.resize(numberOfNeuronBlocks);
        for (uint64_t i = 0; i < numberOfNeuronBlocks; i++) {
            ret = getObjectFromLocalBuffer(positionPtr, &brick.neuronBlocks[i], error);
            if (ret != OK) {
                return ret;
            }
        }
    }

    if (brickEntry.connectionBlocksPos != 0) {
        // check current position
        if (positionPtr - positionOffset != brickEntry.connectionBlocksPos) {
            error.addMessage("Input-data invalid");
            return INVALID_INPUT;
        }

        // connection-blocks and synapse-blocks
        deleteConnections(brick);
        const uint64_t numberOfConnectionBlocks
            = brickEntry.numberOfConnectionBytes / (sizeof(ConnectionBlock) + sizeof(SynapseBlock));
        brick.connectionBlocks.resize(numberOfConnectionBlocks);
        for (uint64_t i = 0; i < numberOfConnectionBlocks; i++) {
            ret = getObjectFromLocalBuffer(positionPtr, &brick.connectionBlocks[i], error);
            if (ret != OK) {
                return ret;
            }
            SynapseBlock synapseBlock;
            ret = getObjectFromLocalBuffer(positionPtr, &synapseBlock, error);
            if (ret != OK) {
                return ret;
            }
            const uint64_t newTargetPosition
                = brick.cluster->attachedHost->synapseBlocks.addNewItem(synapseBlock);
            if (newTargetPosition == UNINIT_STATE_64) {
                return ERROR;
            }
            brick.connectionBlocks[i].targetSynapseBlockPos = newTargetPosition;
        }
    }

    // input
    if (brickEntry.inputInterfacesPos != 0) {
        // check current position
        if (positionPtr - positionOffset != brickEntry.inputInterfacesPos) {
            error.addMessage("Input-data invalid");
            return INVALID_INPUT;
        }

        InputEntry inputEntry;
        ret = getObjectFromLocalBuffer(positionPtr, &inputEntry, error);
        if (ret != OK) {
            return ret;
        }

        InputInterface inputIf;
        inputIf.name = inputEntry.name.getName();
        inputIf.targetBrickId = brick.header.brickId;

        inputIf.inputNeurons.resize(inputEntry.numberOfInputs);
        inputIf.ioBuffer.resize(inputEntry.numberOfInputs);
        for (InputNeuron& inputNeuron : inputIf.inputNeurons) {
            ret = getObjectFromLocalBuffer(positionPtr, &inputNeuron, error);
            if (ret != OK) {
                return ret;
            }
        }

        auto ret = brick.cluster->inputInterfaces.try_emplace(inputIf.name, inputIf);
        if (ret.second == false) {
            error.addMessage("Input-data invalid");
            return INVALID_INPUT;
        }

        brick.inputInterface = &brick.cluster->inputInterfaces[inputIf.name];
    }

    // output
    if (brickEntry.outputsInterfacesPos != 0) {
        // check current position
        if (positionPtr - positionOffset != brickEntry.outputsInterfacesPos) {
            error.addMessage("Input-data invalid");
            return INVALID_INPUT;
        }

        OutputEntry outputEntry;
        ret = getObjectFromLocalBuffer(positionPtr, &outputEntry, error);
        if (ret != OK) {
            return ret;
        }

        OutputInterface outputIf;
        outputIf.name = outputEntry.name.getName();
        outputIf.targetBrickId = brick.header.brickId;

        outputIf.outputNeurons.resize(outputEntry.numberOfOutputs);
        outputIf.ioBuffer.resize(outputEntry.numberOfOutputs);
        for (OutputNeuron& outputNeuron : outputIf.outputNeurons) {
            ret = getObjectFromLocalBuffer(positionPtr, &outputNeuron, error);
            if (ret != OK) {
                return ret;
            }
        }

        auto ret = brick.cluster->outputInterfaces.try_emplace(outputIf.name, outputIf);
        if (ret.second == false) {
            error.addMessage("Input-data invalid");
            return INVALID_INPUT;
        }

        brick.outputInterface = &brick.cluster->outputInterfaces[outputIf.name];
    }

    // check current position
    if (positionPtr - positionOffset != brickEntry.brickSize) {
        error.addMessage("Input-data invalid");
        return INVALID_INPUT;
    }

    return OK;
}

/**
 * @brief check byte-ranges within the read brick-entry to prevent broken input from
 *        crashing the program
 *
 * @param brickEntry brick-entry to check
 *
 * @return true, if all is valid, else false
 */
bool
IO_Interface::checkBrickEntry(const BrickEntry& brickEntry)
{
    // check order
    if (brickEntry.neuronBlocksPos != 0 && brickEntry.neuronBlocksPos < sizeof(BrickEntry)) {
        return false;
    }
    if (brickEntry.connectionBlocksPos != 0
        && brickEntry.connectionBlocksPos < brickEntry.neuronBlocksPos)
    {
        return false;
    }
    if (brickEntry.connectionBlocksPos == 0
        && brickEntry.inputInterfacesPos < brickEntry.connectionBlocksPos)
    {
        return false;
    }
    if (brickEntry.connectionBlocksPos == 0
        && brickEntry.outputsInterfacesPos < brickEntry.connectionBlocksPos)
    {
        return false;
    }

    // check against total brick size
    if (brickEntry.neuronBlocksPos >= brickEntry.brickSize) {
        return false;
    }
    if (brickEntry.connectionBlocksPos >= brickEntry.brickSize) {
        return false;
    }
    if (brickEntry.inputInterfacesPos >= brickEntry.brickSize) {
        return false;
    }
    if (brickEntry.outputsInterfacesPos >= brickEntry.brickSize) {
        return false;
    }

    // check positions
    if (brickEntry.inputInterfacesPos == 0
        && brickEntry.neuronBlocksPos + brickEntry.numberOfNeuronBytes
               != brickEntry.connectionBlocksPos)
    {
        return false;
    }
    if (brickEntry.connectionBlocksPos + brickEntry.numberOfConnectionBytes
            != brickEntry.inputInterfacesPos
        && brickEntry.connectionBlocksPos + brickEntry.numberOfConnectionBytes
               != brickEntry.outputsInterfacesPos
        && brickEntry.connectionBlocksPos + brickEntry.numberOfConnectionBytes
               != brickEntry.brickSize)
    {
        return false;
    }

    // check sizes compared to the object-types
    if (brickEntry.numberOfNeuronBytes % sizeof(NeuronBlock) != 0) {
        return false;
    }
    if (brickEntry.numberOfConnectionBytes % (sizeof(ConnectionBlock) + sizeof(SynapseBlock)) != 0)
    {
        return false;
    }

    if (brickEntry.inputInterfacesPos > 0
        && (brickEntry.brickSize - brickEntry.inputInterfacesPos - sizeof(InputEntry))
                   % sizeof(InputNeuron)
               != 0)
    {
        return false;
    }
    if (brickEntry.outputsInterfacesPos > 0
        && (brickEntry.brickSize - brickEntry.outputsInterfacesPos - sizeof(OutputEntry))
                   % sizeof(OutputNeuron)
               != 0)
    {
        return false;
    }

    // check size against dimentsions in brick-header
    const uint64_t numberOfConnectionBlocks
        = brickEntry.numberOfConnectionBytes / (sizeof(ConnectionBlock) + sizeof(SynapseBlock));
    if (numberOfConnectionBlocks != brickEntry.header.dimX * brickEntry.header.dimY) {
        return false;
    }

    return true;
}

/**
 * @brief create a new brick-entry for a brick
 *
 * @param brick brick for which a new entry should be created
 *
 * @return new created brick-entry
 */
IO_Interface::BrickEntry
IO_Interface::createBrickEntry(const Brick& brick)
{
    BrickEntry brickEntry;

    const uint64_t brickSize = getBrickSize(brick);
    uint64_t posCounter = 0;

    brickEntry.header = brick.header;
    brickEntry.brickSize = brickSize;
    posCounter += sizeof(BrickEntry);

    if (brick.neuronBlocks.size() > 0) {
        brickEntry.neuronBlocksPos = posCounter;
        brickEntry.numberOfNeuronBytes = brick.neuronBlocks.size() * sizeof(NeuronBlock);
        posCounter += brickEntry.numberOfNeuronBytes;
    }

    if (brick.connectionBlocks.size() > 0) {
        brickEntry.connectionBlocksPos = posCounter;
        brickEntry.numberOfConnectionBytes
            = brick.connectionBlocks.size() * (sizeof(ConnectionBlock) + sizeof(SynapseBlock));
        posCounter += brickEntry.numberOfConnectionBytes;
    }

    if (brick.inputInterface != nullptr) {
        brickEntry.inputInterfacesPos = posCounter;
        brickEntry.numberOfInputsBytes
            = sizeof(InputEntry)
              + (brick.inputInterface->inputNeurons.size() * sizeof(InputNeuron));
    }

    if (brick.outputInterface != nullptr) {
        brickEntry.outputsInterfacesPos = posCounter;
        brickEntry.numberOfOutputBytes
            = sizeof(OutputEntry)
              + (brick.outputInterface->outputNeurons.size() * sizeof(OutputNeuron));
    }

    return brickEntry;
}

/**
 * @brief delete all connection-blocks and linked synapse-blocks of a brick to clear the content
 *        before reading new data into it
 *
 * @param brick reference to the brick to clear
 */
void
IO_Interface::deleteConnections(Brick& brick)
{
    for (const ConnectionBlock& connectionBlock : brick.connectionBlocks) {
        brick.cluster->attachedHost->synapseBlocks.deleteItem(
            connectionBlock.targetSynapseBlockPos);
    }
    brick.connectionBlocks.clear();
}
