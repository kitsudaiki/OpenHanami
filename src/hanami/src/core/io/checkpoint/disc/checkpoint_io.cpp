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
#include <core/cluster/objects.h>
#include <core/processing/logical_host.h>
#include <core/processing/physical_host.h>
#include <hanami_common/buffer/item_buffer.h>
#include <hanami_common/functions/file_functions.h>
#include <hanami_root.h>

struct CheckpointHeader {
    const char typeIdentifier[8] = "hanami";
    const char fileIdentifier[32] = "checkpoint";
    const char version[8] = "v1";
    const char minorVersion[8] = "0alpha";

    uint8_t padding1[4040];
};
static_assert(sizeof(CheckpointHeader) == 4096);

/**
 * @brief constructor
 */
CheckpointIO::CheckpointIO() : IO_Interface() {}

/**
 * @brief destructor
 */
CheckpointIO::~CheckpointIO()
{
    ErrorContainer error;
    if (m_checkpointFile != nullptr) {
        m_checkpointFile->closeFile(error);
        delete m_checkpointFile;
    }
}

/**
 * @brief write the checkpoint of the cluster into a local file
 *
 * @param cluster reference to the cluster, which should be written to a checkpoint file
 * @param filePath path to the file, where the checkpoint should be written into
 * @param error reference for error-output
 *
 * @return OK-status, if successful, else ERROR-status
 */
ReturnStatus
CheckpointIO::writeClusterToFile(Cluster& cluster,
                                 const std::string& filePath,
                                 Hanami::ErrorContainer& error)
{
    m_checkpointFile = new Hanami::BinaryFile(filePath);

    const ReturnStatus status = serialize(cluster, error);
    if (status != OK) {
        error.addMessage("Failed to create checkpoint");
        return status;
    }

    return OK;
}

/**
 * @brief restore a cluster from a checkpoint-file
 *
 * @param cluster reference to cluster, in which the data should be restored
 * @param fileLocation path to the local checkpoint-file
 * @param host initial host to attach the hexagons. if nullptr, use the first cpu-host
 * @param error reference for error-output
 *
 * @return return-status based on the result of the process
 */
ReturnStatus
CheckpointIO::restoreClusterFromFile(Cluster& cluster,
                                     const std::string& fileLocation,
                                     LogicalHost* host,
                                     Hanami::ErrorContainer& error)
{
    // backup the original UUID of the cluster to apply this after reading the checkpoint,
    // because the restored cluster is not allowed to have the old UUID
    const std::string originalUuid = cluster.getUuid();

    m_checkpointFile = new Hanami::BinaryFile(fileLocation);

    // check for minimal-size to read at least the size of the checkpoint-header
    if (m_checkpointFile->fileSize < sizeof(CheckpointHeader)) {
        error.addMessage("Given checkpoint-file '" + fileLocation
                         + "' is too small or even empty.");
        return INVALID_INPUT;
    }

    const ReturnStatus ret
        = deserialize(cluster, m_checkpointFile->fileSize - sizeof(CheckpointHeader), host, error);
    if (ret != OK) {
        error.addMessage("Failed to read checkpoint");
        return ret;
    }

    // write original UUID back to the restored cluster
    strncpy(cluster.clusterHeader.uuid.uuid, originalUuid.c_str(), originalUuid.size());

    return OK;
}

/**
 * @brief initialize the file on the disc
 *
 * @param size total size necessary for the complete serialized cluster
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::initializeTarget(const uint64_t size, Hanami::ErrorContainer& error)
{
    // initialize checkpoint-file
    const uint64_t totalFileSize = size + sizeof(CheckpointHeader);
    if (m_checkpointFile->allocateStorage(totalFileSize, error) == false) {
        error.addMessage("Failed to allocate '" + std::to_string(totalFileSize)
                         + "' bytes for checkpointfile");
        return false;
    }

    // write header of cluster to file
    CheckpointHeader header;
    if (m_checkpointFile->writeDataIntoFile(&header, 0, sizeof(CheckpointHeader), error) == false) {
        error.addMessage("Failed to write cluster-header for checkpoint into file");
        return false;
    }

    return true;
}

/**
 * @brief write data from the local buffer to the disc
 *
 * @param localBuffer local buffer with the data to write
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::writeFromLocalBuffer(const LocalBuffer& localBuffer, Hanami::ErrorContainer& error)
{
    if (m_checkpointFile->writeDataIntoFile(localBuffer.cache,
                                            localBuffer.startPos + sizeof(CheckpointHeader),
                                            localBuffer.size,
                                            error)
        == false)
    {
        error.addMessage("Failed to write cluster-payload from local-buffer into file");
        return false;
    }

    return true;
}

/**
 * @brief read a new block of data from the disc into the local buffer
 *
 * @param localBuffer local-buffer where the data should be written into
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
CheckpointIO::readToLocalBuffer(LocalBuffer& localBuffer, Hanami::ErrorContainer& error)
{
    if (m_checkpointFile->readDataFromFile(localBuffer.cache,
                                           localBuffer.startPos + sizeof(CheckpointHeader),
                                           localBuffer.size,
                                           error)
        == false)
    {
        error.addMessage("Failed to read cluster-payload from file into local-buffer");
        return false;
    }

    return true;
}
