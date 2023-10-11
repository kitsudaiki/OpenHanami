/**
 * @file        file_upload.cpp
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

#include <api/websocket/file_upload.h>
#include <core/temp_file_handler.h>
#include <database/checkpoint_table.h>
#include <database/data_set_table.h>
#include <hanami_messages.proto3.pb.h>
#include <hanami_root.h>

class Cluster;

/**
 * @brief handleProtobufFileUpload
 * @param data
 * @param dataSize
 * @return
 */
bool
recvFileUploadPackage(const void* data, const uint64_t dataSize)
{
    FileUpload_Message msg;
    if (msg.ParseFromArray(data, dataSize) == false) {
        Hanami::ErrorContainer error;
        error.addMeesage("Got invalid FileUpload-Message");
        LOG_ERROR(error);
        return false;
    }

    if (TempFileHandler::getInstance()->addDataToPos(
            msg.fileuuid(), msg.position(), msg.data().c_str(), msg.data().size())
        == false) {
        // TODO: error-handling
        std::cout << "failed to write data" << std::endl;
        return false;
    }

    if (msg.islast() == false) {
        return false;
    }

    Hanami::ErrorContainer error;

    if (msg.type() == UploadDataType::DATASET_TYPE) {
        if (DataSetTable::getInstance()->setUploadFinish(msg.datasetuuid(), msg.fileuuid(), error)
            == false) {
            // TODO: error-handling
            return false;
        }
    }

    if (msg.type() == UploadDataType::CHECKPOINT_TYPE) {
        if (CheckpointTable::getInstance()->setUploadFinish(
                msg.datasetuuid(), msg.fileuuid(), error)
            == false) {
            // TODO: error-handling
            return false;
        }
    }

    return true;
}
