/**
 * @file        datasets.cpp
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

#include <libShioriArchive/datasets.h>

#include <libKitsunemimiCommon/buffer/data_buffer.h>
#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiHanamiCommon/component_support.h>
#include <libKitsunemimiHanamiCommon/structs.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>

#include <../../libKitsunemimiHanamiMessages/protobuffers/shiori_messages.proto3.pb.h>
#include <../../libKitsunemimiHanamiMessages/message_sub_types.h>

using Kitsunemimi::Hanami::HanamiMessaging;
using Kitsunemimi::Hanami::HanamiMessagingClient;
using Kitsunemimi::Hanami::SupportedComponents;

namespace Shiori
{

/**
 * @brief get data-set payload from shiori
 *
 * @param token token for request
 * @param uuid uuid of the data-set to download
 * @param error reference for error-output
 *
 * @return data-buffer with data if successful, else nullptr
 */
Kitsunemimi::DataBuffer*
getDatasetData(const std::string &token,
               const std::string &uuid,
               const std::string &columnName,
               Kitsunemimi::ErrorContainer &error)
{
    Kitsunemimi::Hanami::ResponseMessage response;
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;

    // build request to get train-data from shiori
    Kitsunemimi::Hanami::RequestMessage request;
    request.id = "v1/data_set";
    request.httpType = Kitsunemimi::Hanami::GET_TYPE;
    request.inputValues = "{\"token\":\"" + token + "\""
                          ",\"uuid\":\"" + uuid + "\""
                          "}";

    if(client == nullptr) {
        return nullptr;
    }

    // send request to shiori
    if(client->triggerSakuraFile(response, request, error) == false) {
        return nullptr;
    }

    // check response
    if(response.success == false)
    {
        error.addMeesage(response.responseContent);
        return nullptr;
    }

    // parse result
    Kitsunemimi::JsonItem jsonItem;
    if(jsonItem.parse(response.responseContent, error) == false) {
        return nullptr;
    }

    // create real request
    DatasetRequest_Message msg;
    msg.set_location(jsonItem.get("location").getString());
    msg.set_columnname(columnName);

    uint8_t buffer[96*1024];
    const uint64_t msgSize = msg.ByteSizeLong();
    if(msg.SerializeToArray(buffer, msgSize) == false)
    {
        error.addMeesage("Failed to serialize error-message to shiori");
        return nullptr;
    }

    return client->sendGenericRequest(SHIORI_DATASET_REQUEST_MESSAGE_TYPE, buffer, msgSize, error);
}

/**
 * @brief get information of a specific data-set from shiori
 *
 * @param result reference for result-output
 * @param dataSetUuid uuid of the requested data-set
 * @param token for authetification against shiori
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getDataSetInformation(Kitsunemimi::JsonItem &result,
                      const std::string &dataSetUuid,
                      const std::string &token,
                      Kitsunemimi::ErrorContainer &error)
{
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr) {
        return false;
    }

    Kitsunemimi::Hanami::ResponseMessage response;

    // create request for remote-calls
    Kitsunemimi::Hanami::RequestMessage request;
    request.id = "v1/data_set";
    request.httpType = Kitsunemimi::Hanami::GET_TYPE;
    request.inputValues = "{\"uuid\" : \"" + dataSetUuid + "\","
                          "\"token\":\"" + token + "\"}";

    // send request to the target
    if(client->triggerSakuraFile(response, request, error) == false) {
        return false;
    }

    // check response
    if(response.success == false)
    {
        error.addMeesage(response.responseContent);
        return false;
    }

    // parse result
    if(result.parse(response.responseContent, error) == false) {
        return false;
    }

    return true;
}

}
