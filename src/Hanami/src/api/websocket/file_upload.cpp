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

#include <api/endpoint_processing/http_websocket_thread.h>
#include <api/websocket/file_upload.h>
#include <core/temp_file_handler.h>
#include <database/checkpoint_table.h>
#include <database/dataset_table.h>
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
recvFileUploadPackage(FileHandle* fileHandle,
                      const void* data,
                      const uint64_t dataSize,
                      std::string& errorMessage)
{
    FileUpload_Message msg;
    if (msg.ParseFromArray(data, dataSize) == false) {
        errorMessage = "Got invalid FileUpload-Message";
        return false;
    }

    return fileHandle->addDataToPos(
        msg.position(), msg.data().c_str(), msg.data().size(), errorMessage);
}

/**
 * @brief sendResultMessage
 * @param success
 * @param errorMessage
 */
void
sendFileUploadResponse(HttpWebsocketThread* msgClient,
                       const bool success,
                       const std::string& errorMessage)
{
    // build message
    FileUploadResponse_Message response;
    response.set_success(success);
    response.set_errormessage(errorMessage);

    // serialize message
    uint8_t buffer[TRANSFER_SEGMENT_SIZE];
    const uint64_t size = response.ByteSizeLong();
    if (response.SerializeToArray(buffer, size) == false) {
        Hanami::ErrorContainer error;
        error.addMessage("Failed to serialize request-message");
        LOG_ERROR(error);
        return;
    }

    // send message
    Hanami::ErrorContainer error;
    msgClient->sendData(buffer, size);
}
