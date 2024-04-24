/**
 * @file        checkpoint_io.cpp
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

#include "checkpoint_io.h"

#include <core/cluster/cluster.h>
#include <core/processing/logical_host.h>
#include <core/processing/objects.h>
#include <core/processing/physical_host.h>
#include <hanami_common/buffer/item_buffer.h>
#include <hanami_common/methods/file_methods.h>
#include <hanami_root.h>

//=========================================================================================================
//=========================================================================================================
//=========================================================================================================

struct CheckpointHeader {
    const char typeIdentifier[8] = "hanami";
    const char fileIdentifier[32] = "checkpoint";

    uint8_t version = 1;
    uint8_t padding1[7];

    uint64_t fileSize = 0;

    uint64_t numberOfNeuronBrocks = 0;
    uint64_t numberOfOutputNeurons = 0;
    uint64_t numberOfBricks = 0;

    uint64_t clusterHeaderPos = 0;
    uint64_t neuronBlocksPos = 0;
    uint64_t outputNeuronsPos = 0;
    uint64_t bricksPos = 0;
    uint64_t connectionBlocks = 0;

    char name[256];
    uint32_t nameSize = 0;
    char uuid[40];

    uint8_t padding2[3676];

    CheckpointHeader()
    {
        std::fill_n(uuid, 40, '\0');
        std::fill_n(name, 256, '\0');
    }

    /**
     * @brief set new name for the brick
     *
     * @param newName new name
     *
     * @return true, if successful, else false
     */
    bool setName(const std::string& newName)
    {
        // precheck
        if (newName.size() > 255 || newName.size() == 0) {
            return false;
        }

        // copy string into char-buffer and set explicit the escape symbol to be absolut sure
        // that it is set to absolut avoid buffer-overflows
        strncpy(name, newName.c_str(), newName.size());
        name[newName.size()] = '\0';
        nameSize = newName.size();

        return true;
    }

    bool setUuid(const kuuid& uuid)
    {
        const std::string uuidStr = uuid.toString();

        strncpy(this->uuid, uuidStr.c_str(), uuidStr.size());
        this->uuid[uuidStr.size()] = '\0';

        return true;
    }
};
static_assert(sizeof(CheckpointHeader) == 4096);

//=========================================================================================================
//=========================================================================================================
//=========================================================================================================

CheckpointIO::CheckpointIO() {}

//=========================================================================================================
//=========================================================================================================
//=========================================================================================================

/**
 * @brief write the checkpoint of the cluster into a local file
 *
 * @param cluster pointer to the cluster, which should be written to a checkpoint file
 * @param filePath path to the file, where the checkpoint should be written into
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::writeClusterToFile(Cluster* cluster,
                                 const std::string& filePath,
                                 Hanami::ErrorContainer& error)
{
    // precheck cluster
    if (cluster == nullptr) {
        error.addMessage("Invalid cluster given as source for the new checkpoint");
        return false;
    }

    const uint64_t totalFileSize = cluster->getDataSize() + sizeof(CheckpointHeader);

    // initialize checkpoint-file
    Hanami::BinaryFile checkpointFile(filePath);
    if (checkpointFile.allocateStorage(totalFileSize, error) == false) {
        error.addMessage("Failed to allocate '" + std::to_string(totalFileSize)
                         + "' bytes for checkpointfile at path '" + filePath + "'");
        return false;
    }

    // create header
    CheckpointHeader header;
    header.setName(cluster->getName());
    header.setUuid(cluster->clusterHeader.uuid);
    header.fileSize = totalFileSize;
    header.numberOfBricks = cluster->bricks.size();
    header.numberOfNeuronBrocks = cluster->neuronBlocks.size();

    uint64_t position = sizeof(CheckpointHeader);

    // cluster-header
    header.clusterHeaderPos = position;
    if (writeClusterHeaderToFile(cluster, checkpointFile, position, error) == false) {
        return false;
    }

    // neuron-blocks
    header.neuronBlocksPos = position;
    if (writeNeuronBlocksToFile(cluster, checkpointFile, position, error) == false) {
        return false;
    }

    // output-neurons
    header.outputNeuronsPos = position;
    if (writeOutputNeuronsToFile(cluster, checkpointFile, position, error) == false) {
        return false;
    }

    // bricks
    header.bricksPos = position;
    if (writeBricksToFile(cluster, checkpointFile, position, error) == false) {
        return false;
    }

    // connection-blocks
    header.connectionBlocks = position;
    if (writeConnectionBlocksOfBricksToFile(cluster, checkpointFile, position, error) == false) {
        return false;
    }

    // write header of cluster to file
    if (checkpointFile.writeDataIntoFile(&header, 0, sizeof(CheckpointHeader), error) == false) {
        error.addMessage("Failed to write cluster-header for checkpoint into file");
        return false;
    }

    return true;
}

/**
 * @brief write cluster-header into the file
 *
 * @param cluster pointer to the cluster, which should be written to a checkpoint file
 * @param file reference to file-handler
 * @param position position-counter to identify the position where to write into the file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::writeClusterHeaderToFile(Cluster* cluster,
                                       Hanami::BinaryFile& file,
                                       uint64_t& position,
                                       Hanami::ErrorContainer& error)
{
    const uint64_t numberOfBytes = sizeof(ClusterHeader);

    // write static data of cluster to file
    if (file.writeDataIntoFile(&cluster->clusterHeader, position, numberOfBytes, error) == false) {
        error.addMessage("Failed to write cluster-header for checkpoint into file");
        return false;
    }
    position += numberOfBytes;

    return true;
}

/**
 * @brief write bricks into the file
 *
 * @param cluster pointer to the cluster, which should be written to a checkpoint file
 * @param file reference to file-handler
 * @param position position-counter to identify the position where to write into the file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::writeBricksToFile(Cluster* cluster,
                                Hanami::BinaryFile& file,
                                uint64_t& position,
                                Hanami::ErrorContainer& error)
{
    const uint64_t numberOfBytes = cluster->bricks.size() * sizeof(Brick);

    // write static data of cluster to file
    if (file.writeDataIntoFile(&cluster->bricks[0], position, numberOfBytes, error) == false) {
        error.addMessage("Failed to write bricks for checkpoint into file");
        return false;
    }
    position += numberOfBytes;

    return true;
}

/**
 * @brief write neuron-blocks into the file
 *
 * @param cluster pointer to the cluster, which should be written to a checkpoint file
 * @param file reference to file-handler
 * @param position position-counter to identify the position where to write into the file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::writeNeuronBlocksToFile(Cluster* cluster,
                                      Hanami::BinaryFile& file,
                                      uint64_t& position,
                                      Hanami::ErrorContainer& error)
{
    const uint64_t numberOfBytes = cluster->neuronBlocks.size() * sizeof(NeuronBlock);

    // write static data of cluster to file
    if (file.writeDataIntoFile(&cluster->neuronBlocks[0], position, numberOfBytes, error) == false)
    {
        error.addMessage("Failed to write neuron-blocks for checkpoint into file");
        return false;
    }
    position += numberOfBytes;

    return true;
}

/**
 * @brief write output-neurons into the file
 *
 * @param cluster pointer to the cluster, which should be written to a checkpoint file
 * @param file reference to file-handler
 * @param position position-counter to identify the position where to write into the file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::writeOutputNeuronsToFile(Cluster* cluster,
                                       Hanami::BinaryFile& file,
                                       uint64_t& position,
                                       Hanami::ErrorContainer& error)
{
    /*const uint64_t numberOfBytes = cluster->outputNeurons.size() * sizeof(OutputNeuron);

    // write static data of cluster to file
    if (file.writeDataIntoFile(&cluster->outputNeurons[0], position, numberOfBytes, error) == false)
    {
        error.addMessage("Failed to write output-neurons for checkpoint into file");
        return false;
    }
    position += numberOfBytes;*/

    return true;
}

/**
 * @brief write connection-blocks of the bricks into the file
 *
 * @param cluster pointer to the cluster, which should be written to a checkpoint file
 * @param file reference to file-handler
 * @param position position-counter to identify the position where to write into the file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::writeConnectionBlocksOfBricksToFile(Cluster* cluster,
                                                  Hanami::BinaryFile& file,
                                                  uint64_t& position,
                                                  Hanami::ErrorContainer& error)
{
    for (uint64_t i = 0; i < cluster->bricks.size(); i++) {
        const uint64_t numberOfConnections = cluster->bricks[i].connectionBlocks->size();
        for (uint64_t c = 0; c < numberOfConnections; c++) {
            if (writeConnectionBlockToFile(cluster, file, position, i, c, error) == false) {
                return true;
            }
        }
    }

    return true;
}

/**
 * @brief write a block of a brick into the file
 *
 * @param cluster pointer to the cluster, which should be written to a checkpoint file
 * @param file reference to file-handler
 * @param position position-counter to identify the position where to write into the file
 * @param brickId id of the brick
 * @param blockid id of the block within the brick
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::writeConnectionBlockToFile(Cluster* cluster,
                                         Hanami::BinaryFile& file,
                                         uint64_t& position,
                                         const uint64_t brickId,
                                         const uint64_t blockid,
                                         Hanami::ErrorContainer& error)
{
    // write connection-blocks of brick to file
    ConnectionBlock* connectionBlock = &cluster->bricks[brickId].connectionBlocks->at(blockid);
    if (file.writeDataIntoFile(connectionBlock, position, sizeof(ConnectionBlock), error) == false)
    {
        error.addMessage("Failed to write connection-blocks for checkpoint into file");
        return false;
    }
    position += sizeof(ConnectionBlock);

    // write synapse-blocks of brick to file
    if (writeSynapseBlockToFile(
            cluster, file, position, connectionBlock->targetSynapseBlockPos, error)
        == false)
    {
        return false;
    }

    return true;
}

/**
 * @brief write synapse-block into the file
 *
 * @param cluster pointer to the cluster, which should be written to a checkpoint file
 * @param file reference to file-handler
 * @param position position-counter to identify the position where to write into the file
 * @param targetSynapseBlockPos position of the synapse-block within the global buffer
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::writeSynapseBlockToFile(Cluster* cluster,
                                      Hanami::BinaryFile& file,
                                      uint64_t& position,
                                      const uint64_t targetSynapseBlockPos,
                                      Hanami::ErrorContainer& error)
{
    SynapseBlock* synapseBlocks
        = Hanami::getItemData<SynapseBlock>(cluster->attachedHost->synapseBlocks);
    if (file.writeDataIntoFile(
            &synapseBlocks[targetSynapseBlockPos], position, sizeof(SynapseBlock), error)
        == false)
    {
        error.addMessage("Failed to write synapse-blocks for checkpoint into file");
        return false;
    }
    position += sizeof(SynapseBlock);

    return true;
}

//=========================================================================================================
//=========================================================================================================
//=========================================================================================================

/**
 * @brief restore a cluster from a checkpoint-file
 *
 * @param cluster pointer to cluster, in which the data should be restored
 * @param fileLocation path to the local checkpoint-file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::restoreClusterFromFile(Cluster* cluster,
                                     const std::string& fileLocation,
                                     Hanami::ErrorContainer& error)
{
    // precheck cluster
    if (cluster == nullptr) {
        error.addMessage("Invalid cluster given as target for restoring the checkpoint");
        return false;
    }

    // backup the original UUID of the cluster to apply this after reading the checkpoint,
    // because the restored cluster is not allowed to have the old UUID
    const std::string originalUuid = cluster->getUuid();

    // get checkpoint-data
    Hanami::BinaryFile checkpointFile(fileLocation);
    Hanami::DataBuffer checkpointBuffer;
    if (checkpointFile.readCompleteFile(checkpointBuffer, error) == false) {
        error.addMessage("Failed to load checkpoint-data from file '" + fileLocation + "'");
        return false;
    }
    uint8_t* u8Data = static_cast<uint8_t*>(checkpointBuffer.data);

    // check for minimal-size to read at least the size of the checkpoint-header
    if (checkpointBuffer.usedBufferSize < sizeof(CheckpointHeader)) {
        error.addMessage("Given checkpoint-file '" + fileLocation
                         + "' is too small or even empty.");
        return false;
    }

    // convert checkpoint-header
    CheckpointHeader header;
    memcpy(&header, &u8Data[0], sizeof(CheckpointHeader));

    // check size of the read file compared to the expaced size of the header
    if (header.fileSize != checkpointBuffer.usedBufferSize) {
        error.addMessage("Given checkpoint-file '"
                         + fileLocation
                         + "' can not be restored, because the size doesn't "
                           "match the given size of the checkpoint-header.");
        return false;
    }

    // restore data
    if (restoreClusterHeader(cluster, header, u8Data, error) == false) {
        return false;
    }
    if (restoreNeuronBlocks(cluster, header, u8Data, error) == false) {
        return false;
    }
    if (restoreOutputNeurons(cluster, header, u8Data, error) == false) {
        return false;
    }
    if (restoreBricks(cluster, header, u8Data, error) == false) {
        return false;
    }
    if (restoreConnectionBlocks(cluster, header, u8Data, error) == false) {
        return false;
    }

    // write original UUID back to the restored cluster
    strncpy(cluster->clusterHeader.uuid.uuid, originalUuid.c_str(), originalUuid.size());

    return true;
}

/**
 * @brief restore cluster-header from the checkpoint
 *
 * @param cluster pointer to cluster, in which the data should be restored
 * @param header header of the checkpoint
 * @param u8Data pointer to buffer with data of the checkpoint
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::restoreClusterHeader(Cluster* cluster,
                                   const CheckpointHeader& header,
                                   uint8_t* u8Data,
                                   Hanami::ErrorContainer& error)
{
    const uint64_t position = header.clusterHeaderPos;
    const uint64_t size = sizeof(ClusterHeader);

    memcpy(&cluster->clusterHeader, &u8Data[position], size);

    return true;
}

/**
 * @brief restore neuron-blocks from the checkpoint
 *
 * @param cluster pointer to cluster, in which the data should be restored
 * @param header header of the checkpoint
 * @param u8Data pointer to buffer with data of the checkpoint
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::restoreNeuronBlocks(Cluster* cluster,
                                  const CheckpointHeader& header,
                                  uint8_t* u8Data,
                                  Hanami::ErrorContainer& error)
{
    const uint64_t position = header.neuronBlocksPos;
    const uint64_t size = header.numberOfNeuronBrocks * sizeof(NeuronBlock);

    cluster->neuronBlocks.clear();
    cluster->neuronBlocks.resize(header.numberOfNeuronBrocks);

    memcpy(&cluster->neuronBlocks[0], &u8Data[position], size);

    return true;
}

/**
 * @brief restore output-neurons from the checkpoint
 *
 * @param cluster pointer to cluster, in which the data should be restored
 * @param header header of the checkpoint
 * @param u8Data pointer to buffer with data of the checkpoint
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::restoreOutputNeurons(Cluster* cluster,
                                   const CheckpointHeader& header,
                                   uint8_t* u8Data,
                                   Hanami::ErrorContainer& error)
{
    const uint64_t position = header.outputNeuronsPos;
    const uint64_t size = header.numberOfOutputNeurons * sizeof(OutputNeuron);

    /*cluster->outputNeurons.clear();
    cluster->outputNeurons.resize(header.numberOfOutputNeurons);

    memcpy(&cluster->outputNeurons[0], &u8Data[position], size);*/

    return true;
}

/**
 * @brief restore bricks from the checkpoint
 *
 * @param cluster pointer to cluster, in which the data should be restored
 * @param header header of the checkpoint
 * @param u8Data pointer to buffer with data of the checkpoint
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::restoreBricks(Cluster* cluster,
                            const CheckpointHeader& header,
                            uint8_t* u8Data,
                            Hanami::ErrorContainer& error)
{
    const uint64_t position = header.bricksPos;
    const uint64_t size = header.numberOfBricks * sizeof(Brick);

    cluster->bricks.clear();
    cluster->bricks.resize(header.numberOfBricks);

    memcpy(&cluster->bricks[0], &u8Data[position], size);

    for (Brick& brick : cluster->bricks) {
        brick.connectionBlocks = new std::vector<ConnectionBlock>();
    }

    return true;
}

/**
 * @brief restore connection-block and the connected synapse-blocks from the checkpoint
 *
 * @param cluster pointer to cluster, in which the data should be restored
 * @param header header of the checkpoint
 * @param u8Data pointer to buffer with data of the checkpoint
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::restoreConnectionBlocks(Cluster* cluster,
                                      const CheckpointHeader& header,
                                      uint8_t* u8Data,
                                      Hanami::ErrorContainer& error)
{
    uint64_t position = header.connectionBlocks;

    // get initial logical host
    LogicalHost* host = HanamiRoot::physicalHost->getFirstHost();
    if (host == nullptr) {
        error.addMessage("No logical host found for new cluster.");
        return false;
    }

    // convert payload of bricks
    for (Brick& brick : cluster->bricks) {
        brick.connectionBlocks->resize(brick.dimX * brick.dimY);

        const uint64_t numberOfConnections = brick.connectionBlocks->size();
        for (uint64_t c = 0; c < numberOfConnections; c++) {
            // convert connection-block
            memcpy(&brick.connectionBlocks[0][c], &u8Data[position], sizeof(ConnectionBlock));
            position += sizeof(ConnectionBlock);

            // convert synapse-block
            SynapseBlock newSynapseBlock;
            memcpy(&newSynapseBlock, &u8Data[position], sizeof(SynapseBlock));
            const uint64_t itemPos = host->synapseBlocks.addNewItem(newSynapseBlock);
            if (itemPos == UNINIT_STATE_64) {
                error.addMessage("failed allocate synapse-block for checkpoint");
                return false;
            }

            // write new position into the related connection-block
            brick.connectionBlocks[0][c].targetSynapseBlockPos = itemPos;

            position += sizeof(SynapseBlock);
        }
    }

    return true;
}

//=========================================================================================================
//=========================================================================================================
//=========================================================================================================
