/**
 * @file        save_cluster_state.cpp
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

#include "save_cluster_state.h"

#include <config.h>
#include <database/cluster_snapshot_table.h>
#include <hanami_root.h>
#include <core/cluster/task.h>
#include <core/cluster/cluster.h>
#include <core/cluster/statemachine_init.h>
#include <core/segments/core_segment/core_segment.h>

#include <libKitsunemimiCrypto/hashes.h>
#include <libKitsunemimiCommon/files/binary_file.h>

/**
 * @brief constructor
 *
 * @param cluster pointer to the cluster, where the event and the statemachine belongs to
 */
SaveCluster_State::SaveCluster_State(Cluster* cluster)
{
    m_cluster = cluster;
}

/**
 * @brief destructor
 */
SaveCluster_State::~SaveCluster_State() {}

/**
 * @brief prcess event
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::processEvent()
{
    bool result = false;
    Kitsunemimi::ErrorContainer error;

    do
    {
        Task* actualTask = m_cluster->getActualTask();
        uint64_t totalSize = 0;
        std::string headerMessage = "";

        // create message to shiori and calculate total size of storage of the cluster
        totalSize = m_cluster->clusterData.usedBufferSize;
        headerMessage = "{\"header\":" + std::to_string(totalSize) + ",\"segments\":[";
        for(uint64_t i = 0; i < m_cluster->coreSegments.size(); i++)
        {
            if(i != 0) {
                headerMessage += ",";
            }
            const uint64_t segSize = m_cluster->coreSegments.at(i)->segmentData.buffer.usedBufferSize;
            headerMessage += "{\"size\":"
                             + std::to_string(segSize)
                             + ",\"type\":"
                             + std::to_string(m_cluster->coreSegments.at(i)->getType())
                             + "}";
            totalSize += segSize;
        }
        headerMessage += "]}";

        // send snapshot to shiori
        std::string fileUuid = "";
        // snapshots are created by another internal process, which gives the id's not in the context
        // object, but as normal values
        UserContext userContext;
        userContext.userId = actualTask->userId;
        userContext.projectId = actualTask->projectId;

        // get directory to store data from config
        bool success = false;
        std::string targetFilePath = GET_STRING_CONFIG("storage", "cluster_snapshot_location", success);
        if(success == false)
        {
            error.addMeesage("snapshot-location to store cluster-snapshot is missing in the config");
            break;
        }

        // build absolut file-path to store the file
        if(targetFilePath.at(targetFilePath.size() - 1) != '/') {
            targetFilePath.append("/");
        }
        targetFilePath.append(actualTask->uuid.toString() + "_snapshot_" + actualTask->userId);

        // register in database
        Kitsunemimi::JsonItem dbEntry;
        dbEntry.insert("uuid", actualTask->uuid.toString());
        dbEntry.insert("name", actualTask->snapshotName);
        dbEntry.insert("location", targetFilePath);
        dbEntry.insert("header", headerMessage);
        dbEntry.insert("project_id", actualTask->projectId);
        dbEntry.insert("owner_id", actualTask->userId);
        dbEntry.insert("visibility", "private");
        dbEntry.insert("temp_files", "");

        // add to database
        if(ClusterSnapshotTable::getInstance()->addClusterSnapshot(dbEntry,
                                                                   userContext,
                                                                   error) == false)
        {
            break;
        }

        // write data of cluster to disc
        if(writeData(targetFilePath, totalSize, error) == false) {
            break;
        }

        result = true;
        break;
    }
    while(true);

    m_cluster->goToNextState(FINISH_TASK);

    if(result == false)
    {
        error.addMeesage("Failed to create snapshot of cluster with UUID '"
                         + m_cluster->getUuid()
                         + "'");
        // TODO: cleanup in error-case
        // TODO: give the user a feedback by setting the task to failed-state
    }

    return result;
}

/**
 * @brief send all data of the snapshot to shiori
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SaveCluster_State::writeData(const std::string &filePath,
                             const uint64_t fileSize,
                             Kitsunemimi::ErrorContainer &error)
{
    Kitsunemimi::BinaryFile snapshotFile(filePath);
    if(snapshotFile.allocateStorage(fileSize, error) == false)
    {
        error.addMeesage("Failed to allocate '"
                         + std::to_string(fileSize)
                         + "' bytes for snapshotfile at path '"
                         + filePath
                         + "'");
        return false;
    }

    // global byte-counter to identifiy the position within the complete snapshot
    uint64_t posCounter = 0;
    Kitsunemimi::DataBuffer* buffer = nullptr;

    // write metadata of cluster
    buffer = &m_cluster->clusterData;
    if(snapshotFile.writeDataIntoFile(buffer->data,
                                      posCounter,
                                      buffer->usedBufferSize,
                                      error) == false)
    {
        error.addMeesage("Failed to write metadata of cluster for snapshot into file '"
                         + filePath
                         + "'");
        return false;
    }

    posCounter += buffer->usedBufferSize;


    // write segments of cluster
    for(uint64_t i = 0; i < m_cluster->coreSegments.size(); i++)
    {
        buffer = &m_cluster->coreSegments.at(i)->segmentData.buffer;
        if(snapshotFile.writeDataIntoFile(buffer->data,
                                          posCounter,
                                          buffer->usedBufferSize,
                                          error) == false)
        {
            error.addMeesage("Failed to write segment of cluster for snapshot into file '"
                             + filePath
                             + "'");
            return false;
        }

        posCounter += buffer->usedBufferSize;
    }

    return true;
}
