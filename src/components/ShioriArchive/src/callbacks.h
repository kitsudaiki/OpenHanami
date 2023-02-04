/**
 * @file        callbacks.h
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

#ifndef SHIORIARCHIVE_CALLBACKS_H
#define SHIORIARCHIVE_CALLBACKS_H

#include <core/temp_file_handler.h>
#include <core/data_set_files/data_set_file.h>
#include <database/data_set_table.h>
#include <database/cluster_snapshot_table.h>
#include <database/request_result_table.h>
#include <database/audit_log_table.h>
#include <database/error_log_table.h>
#include <shiori_root.h>

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/files/binary_file.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiCrypto/common.h>

#include <libKitsunemimiSakuraNetwork/session.h>

#include <../../libraries/libKitsunemimiHanamiMessages/protobuffers/shiori_messages.proto3.pb.h>
#include <../../libraries/libKitsunemimiHanamiMessages/message_sub_types.h>

/**
 * @brief handleProtobufFileUpload
 * @param data
 * @param dataSize
 * @return
 */
bool
handleProtobufFileUpload(Kitsunemimi::Sakura::Session* session,
                         const void* data,
                         const uint64_t dataSize)
{
    FileUpload_Message msg;
    if(msg.ParseFromArray(data, dataSize) == false)
    {
        Kitsunemimi::ErrorContainer error;
        error.addMeesage("Got invalid FileUpload-Message");
        LOG_ERROR(error);
        return false;
    }

    if(ShioriRoot::tempFileHandler->addDataToPos(msg.fileuuid(),
                                                 msg.position(),
                                                 msg.data().c_str(),
                                                 msg.data().size()) == false)
    {
        // TODO: error-handling
        std::cout<<"failed to write data"<<std::endl;
        return false;
    }

    if(msg.islast() == false) {
        return false;
    }

    Kitsunemimi::ErrorContainer error;

    if(msg.type() == UploadDataType::DATASET_TYPE)
    {
        if(ShioriRoot::dataSetTable->setUploadFinish(msg.datasetuuid(),
                                                     msg.fileuuid(),
                                                     error) == false)
        {
            // TODO: error-handling
            return false;
        }
    }

    if(msg.type() == UploadDataType::CLUSTER_SNAPSHOT_TYPE)
    {
        if(ShioriRoot::clusterSnapshotTable->setUploadFinish(msg.datasetuuid(),
                                                             msg.fileuuid(),
                                                             error) == false)
        {
            // TODO: error-handling
            return false;
        }
    }

    return true;
}

/**
 * @brief streamDataCallback
 * @param data
 * @param dataSize
 */
void streamDataCallback(void*,
                        Kitsunemimi::Sakura::Session* session,
                        const void* data,
                        const uint64_t dataSize)
{
    if(dataSize <= 40) {
        return;
    }

    handleProtobufFileUpload(session, data, dataSize);
}

/**
 * @brief get the current datetime of the system
 *
 * @return datetime as string
 */
const std::string
getDatetime()
{
    const time_t now = time(nullptr);
    tm *ltm = localtime(&now);

    const std::string datatime =
            std::to_string(1900 + ltm->tm_year)
            + "-"
            + std::to_string(1 + ltm->tm_mon)
            + "-"
            + std::to_string(ltm->tm_mday)
            + " "
            + std::to_string(ltm->tm_hour)
            + ":"
            + std::to_string(ltm->tm_min)
            + ":"
            + std::to_string(ltm->tm_sec);

    return datatime;
}

/**
 * @brief handle cluster-snapshot-message
 *
 * @param msg message to process
 * @param session pointer to the session, which received the message
 * @param blockerId blocker-id for the response
 */
inline void
handleClusterSnapshotRequest(const ClusterSnapshotPull_Message &msg,
                             Kitsunemimi::Sakura::Session* session,
                             const uint64_t blockerId)
{
    Kitsunemimi::ErrorContainer error;

    // init file
    Kitsunemimi::BinaryFile targetFile(msg.location());
    DataSetFile::DataSetHeader header;
    Kitsunemimi::DataBuffer content;
    if(targetFile.readCompleteFile(content, error) == false)
    {
        //TODO: handle error
        LOG_ERROR(error);
    }

    // send data
    if(session->sendResponse(content.data,
                             content.usedBufferSize,
                             blockerId,
                             error) == false)
    {
        LOG_ERROR(error);
    }

    return;
}

/**
 * @brief handle dataset-request-message
 *
 * @param msg message to process
 * @param session pointer to the session, which received the message
 * @param blockerId blocker-id for the response
 */
inline void
handleDataSetRequest(const DatasetRequest_Message &msg,
                     Kitsunemimi::Sakura::Session* session,
                     const uint64_t blockerId)
{
    // init file
    DataSetFile* file = readDataSetFile(msg.location());
    if(file == nullptr) {
        return;
    }

    float* payload = nullptr;

    do
    {
        // get payload
        uint64_t payloadSize = 0;
        payload = file->getPayload(payloadSize, msg.columnname());
        if(payload == nullptr)
        {
            // TODO: error
            break;
        }

        // send data
        Kitsunemimi::ErrorContainer error;
        if(session->sendResponse(payload, payloadSize, blockerId, error) == false) {
            LOG_ERROR(error);
        }

        break;
    }
    while(true);

    delete file;
    if(payload != nullptr) {
        delete[] payload;
    }

    return;
}

/**
 * @brief handle result-push-message
 *
 * @param msg message to process
 * @param session pointer to the session, which received the message
 * @param blockerId blocker-id for the response
 */
inline void
handleResultPush(const ResultPush_Message &msg,
                 Kitsunemimi::Sakura::Session* session,
                 const uint64_t blockerId)
{
    Kitsunemimi::ErrorContainer error;

    Kitsunemimi::JsonItem dataParser;
    if(dataParser.parse(msg.results(), error) == false)
    {
        error.addMeesage("Error while receivind result-data");
        LOG_ERROR(error);
        return;
    }

    Kitsunemimi::JsonItem resultData;
    resultData.insert("uuid", msg.uuid());
    resultData.insert("name", msg.name());
    resultData.insert("data", dataParser.stealItemContent());
    resultData.insert("visibility", "private");

    Kitsunemimi::Hanami::UserContext userContext;
    userContext.userId = msg.userid();
    userContext.projectId = msg.projectid();

    if(ShioriRoot::requestResultTable->addRequestResult(resultData, userContext, error) == false)
    {
        LOG_ERROR(error);

        const std::string ret = "fail";
        if(session->sendResponse(ret.c_str(), ret.size(), blockerId, error) == false) {
            LOG_ERROR(error);
        }
        return;
    }

    const std::string ret = "success";
    if(session->sendResponse(ret.c_str(), ret.size(), blockerId, error) == false) {
        LOG_ERROR(error);
    }
}

/**
 * @brief handle error-log-message
 *
 * @param msg message to process
 */
inline void
handleErrorLog(const ErrorLog_Message &msg)
{
    Kitsunemimi::ErrorContainer error;
    if(ShioriRoot::errorLogTable->addErrorLogEntry(getDatetime(),
                                                   msg.userid(),
                                                   msg.component(),
                                                   msg.context(),
                                                   msg.values(),
                                                   msg.errormsg(),
                                                   error) == false)
    {
        error.addMeesage("ERROR: Failed to write error-log into database");

        // HINT(kitsudaiki): use normal stdout, because LOG_ERROR would trigger this function again
        //       and could create a crash because of a stack-overflow
        std::cout<<error.toString()<<std::endl;
    }
}

/**
 * @brief handle audit-log-message
 *
 * @param msg message to process
 */
inline void
handleAuditLog(const AuditLog_Message &msg)
{
    Kitsunemimi::ErrorContainer error; 
    if(ShioriRoot::auditLogTable->addAuditLogEntry(getDatetime(),
                                                   msg.userid(),
                                                   msg.component(),
                                                   msg.endpoint(),
                                                   msg.type(),
                                                   error) == false)
    {
        error.addMeesage("ERROR: Failed to write audit-log into database");
        LOG_ERROR(error);
    }
}

/**
 * @brief handle errors of message which to requires a response
 *
 * @param msg error-message
 */
inline void
handleFail(const std::string &msg,
           Kitsunemimi::Sakura::Session* session,
           const uint64_t blockerId)
{
    Kitsunemimi::ErrorContainer error;
    error.addMeesage(msg);
    LOG_ERROR(error);

    const std::string ret = "-";
    session->sendResponse(ret.c_str(), ret.size(), blockerId, error);
    return;
}

/**
 * @brief handle generic message-content
 *
 * @param session pointer to the session, which received the message
 * @param data received bytes
 * @param dataSize number of bytes in the received message
 * @param blockerId blocker-id for the response
 */
void
genericMessageCallback(Kitsunemimi::Sakura::Session* session,
                       const uint32_t messageSubType,
                       void* data,
                       const uint64_t dataSize,
                       const uint64_t blockerId)
{
    switch(messageSubType)
    {
        case SHIORI_CLUSTER_SNAPSHOT_PULL_MESSAGE_TYPE:
            {
                ClusterSnapshotPull_Message msg;
                if(msg.ParseFromArray(data, dataSize) == false)
                {
                    handleFail("Receive broken cluster-snapshot-message", session, blockerId);
                    return;
                }

                handleClusterSnapshotRequest(msg, session, blockerId);
            }
            break;
        case SHIORI_DATASET_REQUEST_MESSAGE_TYPE:
            {
                DatasetRequest_Message msg;
                if(msg.ParseFromArray(data, dataSize) == false)
                {
                    handleFail("Receive broken dataset-requests-message", session, blockerId);
                    return;
                }

                handleDataSetRequest(msg, session, blockerId);
            }
            break;
        case SHIORI_RESULT_PUSH_MESSAGE_TYPE:
            {
                ResultPush_Message msg;
                if(msg.ParseFromArray(data, dataSize) == false)
                {
                    handleFail("Receive broken result-push-message", session, blockerId);
                    return;
                }

                handleResultPush(msg, session, blockerId);
            }
            break;
        case SHIORI_AUDIT_LOG_MESSAGE_TYPE:
            {
                AuditLog_Message msg;
                if(msg.ParseFromArray(data, dataSize) == false)
                {
                    Kitsunemimi::ErrorContainer error;
                    error.addMeesage("Receive broken audit-log-message");
                    LOG_ERROR(error);
                    return;
                }

                handleAuditLog(msg);
            }
            break;
        case SHIORI_ERROR_LOG_MESSAGE_TYPE:
            {
                ErrorLog_Message msg;
                if(msg.ParseFromArray(data, dataSize) == false)
                {
                    Kitsunemimi::ErrorContainer error;
                    error.addMeesage("Receive broken error-log-message");
                    LOG_ERROR(error);
                    return;
                }

                handleErrorLog(msg);
            }
            break;
        default:
            handleFail("Received unknown generic message", session, blockerId);
            break;
    }
}

#endif // SHIORIARCHIVE_CALLBACKS_H
