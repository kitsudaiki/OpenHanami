/**
 * @file        data_set.cpp
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

#include <libHanamiAiSdk/data_set.h>
#include <libHanamiAiSdk/common/websocket_client.h>
#include <common/http_client.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/items/data_items.h>
#include <libKitsunemimiCommon/files/binary_file.h>

#include <../../../libraries/libKitsunemimiHanamiMessages/protobuffers/hanami_messages.proto3.pb.h>

namespace HanamiAI
{

/**
 * @brief get size of a local file
 *
 * @param filePath path the file to check
 *
 * @return size of file if successful, or -1 if failed
 */
long
getFileSize(const std::string &filePath)
{
    std::ifstream in(filePath, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

/**
 * @brief initialize a new csv-dataset in shiori
 *
 * @param result reference for response-message
 * @param dataSetName name for the new data-set
 * @param inputDataSize size of the file with the input-data
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
createCsvDataSet(std::string &result,
                 const std::string &dataSetName,
                 const uint64_t inputDataSize,
                 Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/csv/data_set";
    const std::string vars = "";
    const std::string jsonBody = "{\"name\":\""    + dataSetName + "\""
                                 ",\"input_data_size\":" + std::to_string(inputDataSize) + "}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false) {
        return false;
    }

    return true;
}

/**
 * @brief finalize data-set
 *
 * @param result reference for response-message
 * @param uuid uuid to identify the data-set
 * @param inputUuid uuid to identify the temporary file with the input-data on server-side
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
finalizeCsvDataSet(std::string &result,
                   const std::string &uuid,
                   const std::string &inputUuid,
                   Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/csv/data_set";
    const std::string vars = "";
    const std::string jsonBody = "{\"uuid\":\""    + uuid + "\""
                                 ",\"uuid_input_file\":\"" + inputUuid + "\"}";

    // send request
    if(request->sendPutRequest(result, path, vars, jsonBody, error) == false) {
        return false;
    }

    return true;
}

/**
 * @brief initialize a new mnist-dataset in shiori
 *
 * @param result reference for response-message
 * @param dataSetName name for the new data-set
 * @param inputDataSize size of the file with the input-data
 * @param labelDataSize  size of the file with the label-data
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
createMnistDataSet(std::string &result,
                   const std::string &dataSetName,
                   const uint64_t inputDataSize,
                   const uint64_t labelDataSize,
                   Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/mnist/data_set";
    const std::string vars = "";
    const std::string jsonBody = "{\"name\":\""    + dataSetName + "\""
                                 ",\"input_data_size\":" + std::to_string(inputDataSize) +
                                 ",\"label_data_size\":" + std::to_string(labelDataSize) + "}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false) {
        return false;
    }

    return true;
}

/**
 * @brief finalize data-set
 *
 * @param result reference for response-message
 * @param uuid uuid to identify the data-set
 * @param inputUuid uuid to identify the temporary file with the input-data on server-side
 * @param labelUuid uuid to identify the temporary file with the label-data on server-side
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
finalizeMnistDataSet(std::string &result,
                     const std::string &uuid,
                     const std::string &inputUuid,
                     const std::string &labelUuid,
                     Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/mnist/data_set";
    const std::string vars = "";
    const std::string jsonBody = "{\"uuid\":\""    + uuid + "\""
                                 ",\"uuid_input_file\":\"" + inputUuid + "\""
                                 ",\"uuid_label_file\":\"" + labelUuid + "\"}";

    // send request
    if(request->sendPutRequest(result, path, vars, jsonBody, error) == false) {
        return false;
    }

    return true;
}

/**
 * @brief send data to shiori
 *
 * @param session session over which the data should be send
 * @param datasetUuid uuid of the dataset where the file belongs to
 * @param fileUuid uuid of the file for identification in shiori
 * @param filePath path to file, which should be send
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
sendFile(WebsocketClient* client,
         const std::string &datasetUuid,
         const std::string &fileUuid,
         const std::string &filePath,
         Kitsunemimi::ErrorContainer &error)
{
    // get data, which should be send
    const uint64_t dataSize = getFileSize(filePath);
    Kitsunemimi::BinaryFile sourceFile(filePath);

    bool success = true;
    uint64_t pos = 0;

    // prepare buffer
    uint64_t segmentSize = 96 * 1024;

    uint8_t readBuffer[96*1024];
    uint8_t sendBuffer[128*1024];

    do
    {
        // check the size for the last segment
        segmentSize = 96 * 1024;
        if(dataSize - pos < segmentSize) {
            segmentSize = dataSize - pos;
        }

        FileUpload_Message message;
        message.set_fileuuid(fileUuid);
        message.set_datasetuuid(datasetUuid);
        message.set_type(UploadDataType::DATASET_TYPE);
        message.set_islast(false);
        if(pos + segmentSize >= dataSize) {
            message.set_islast(true);
        }

        // read segment of the local file
        if(sourceFile.readDataFromFile(&readBuffer[0], pos, segmentSize, error) == false)
        {
            success = false;
            error.addMeesage("Failed to read file '" + filePath + "'");
            break;
        }

        message.set_position(pos);
        message.set_data(readBuffer, segmentSize);

        const uint64_t msgSize = message.ByteSizeLong();
        if(message.SerializeToArray(sendBuffer, msgSize) == false)
        {
            error.addMeesage("Failed to serialize train-message");
            return false;
        }

        // send segment
        if(client->sendMessage(sendBuffer, msgSize, error) == false)
        {
            LOG_ERROR(error);
            success = false;
            break;
        }

        pos += segmentSize;
    }
    while(pos < dataSize);

    return success;
}

/**
 * @brief wait until the upload of all tempfiles are complete
 *
 * @param uuid uuid of the dataset
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
waitUntilFullyUploaded(const std::string &uuid,
                       Kitsunemimi::ErrorContainer &error)
{
    // TODO: add timeout-timer
    bool completeUploaded = false;
    while(completeUploaded == false)
    {
        sleep(1);

        std::string progressStr = "";
        if(getDatasetProgress(progressStr, uuid, error) == false)
        {
            LOG_ERROR(error);
            return false;
        }

        Kitsunemimi::JsonItem progress;
        if(progress.parse(progressStr, error) == false)
        {
            LOG_ERROR(error);
            return false;
        }

        completeUploaded = progress.get("complete").getBool();
    }

    return true;
}

/**
 * @brief upload new csv-data-set to shiori
 *
 * @param result reference for response-message
 * @param dataSetName name for the new data-set
 * @param inputFilePath path to file with the inputs
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
uploadCsvData(std::string &result,
              const std::string &dataSetName,
              const std::string &inputFilePath,
              Kitsunemimi::ErrorContainer &error)
{
    // init new mnist-data-set
    if(createCsvDataSet(result,
                        dataSetName,
                        getFileSize(inputFilePath),
                        error) == false)
    {
        return false;
    }

    // parse output to get the uuid
    Kitsunemimi::JsonItem jsonItem;
    if(jsonItem.parse(result, error) == false)
    {
        LOG_ERROR(error);
        return false;
    }

    // get ids from inital reponse to identify the file-transfer
    const std::string uuid = jsonItem.get("uuid").getString();
    const std::string inputUuid = jsonItem.get("uuid_input_file").getString();

    // init websocket to shiori
    WebsocketClient wsClient;
    std::string websocketUuid = "";
    const bool ret = wsClient.initClient(websocketUuid,
                                         HanamiRequest::getInstance()->getToken(),
                                         "shiori",
                                         HanamiRequest::getInstance()->getHost(),
                                         HanamiRequest::getInstance()->getPort(),
                                         "",
                                         error);
    if(ret == false)
    {
        error.addMeesage("Failed to init websocket to shiori");
        LOG_ERROR(error);
        return false;
    }

    // send file
    if(sendFile(&wsClient, uuid, inputUuid, inputFilePath, error) == false)
    {
        LOG_ERROR(error);
        return false;
    }

    // wait until all data-transfers to shiori are completed
    if(waitUntilFullyUploaded(uuid, error) == false)
    {
        LOG_ERROR(error);
        return false;
    }

    if(finalizeCsvDataSet(result, uuid, inputUuid, error) == false)
    {
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief upload new mnist-data-set to shiori
 *
 * @param result reference for response-message
 * @param dataSetName name for the new data-set
 * @param inputFilePath path to file with the inputs
 * @param labelFilePath path to file with the labels
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
uploadMnistData(std::string &result,
                const std::string &dataSetName,
                const std::string &inputFilePath,
                const std::string &labelFilePath,
                Kitsunemimi::ErrorContainer &error)
{
    // init new mnist-data-set
    if(createMnistDataSet(result,
                          dataSetName,
                          getFileSize(inputFilePath),
                          getFileSize(labelFilePath),
                          error) == false)
    {
        return false;
    }

    // parse output to get the uuid
    Kitsunemimi::JsonItem jsonItem;
    if(jsonItem.parse(result, error) == false)
    {
        LOG_ERROR(error);
        return false;
    }

    // get ids from inital reponse to identify the file-transfer
    const std::string uuid = jsonItem.get("uuid").getString();
    const std::string inputUuid = jsonItem.get("uuid_input_file").getString();
    const std::string labelUuid = jsonItem.get("uuid_label_file").getString();

    // init websocket to shiori
    WebsocketClient wsClient;
    std::string websocketUuid = "";
    const bool ret = wsClient.initClient(websocketUuid,
                                         HanamiRequest::getInstance()->getToken(),
                                         "shiori",
                                         HanamiRequest::getInstance()->getHost(),
                                         HanamiRequest::getInstance()->getPort(),
                                         "",
                                         error);
    if(ret == false)
    {
        error.addMeesage("Failed to init websocket to shiori");
        LOG_ERROR(error);
        return false;
    }

    // send file with inputs
    if(sendFile(&wsClient, uuid, inputUuid, inputFilePath, error) == false)
    {
        error.addMeesage("Failed to send file with input-values");
        LOG_ERROR(error);
        return false;
    }

    // send file with labels
    if(sendFile(&wsClient, uuid, labelUuid, labelFilePath, error) == false)
    {
        error.addMeesage("Failed to send file with labes");
        LOG_ERROR(error);
        return false;
    }

    // wait until all data-transfers to shiori are completed
    if(waitUntilFullyUploaded(uuid, error) == false)
    {
        error.addMeesage("Failed to wait for fully uploaded files");
        LOG_ERROR(error);
        return false;
    }

    if(finalizeMnistDataSet(result, uuid, inputUuid, labelUuid, error) == false)
    {
        error.addMeesage("Failed to finalize MNIST-dataset");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief check values against a data-set to get correctness
 *
 * @param result reference for response-message
 * @param dataUuid uuid of the data-set to compare to
 * @param resultUuid uuid of the result-set to compare
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
checkDataset(std::string &result,
             const std::string &dataUuid,
             const std::string &resultUuid,
             Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/data_set/check";
    const std::string vars = "";
    const std::string jsonBody = "{\"data_set_uuid\":\"" + dataUuid + "\""
                                 ",\"result_uuid\":\"" + resultUuid + "\"}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false) {
        return false;
    }

    return true;
}

/**
 * @brief get metadata of a specific data-set
 *
 * @param result reference for response-message
 * @param dataUuid uuid of the requested data-set
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getDataset(std::string &result,
           const std::string &dataUuid,
           Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/data_set";
    const std::string vars = "uuid=" + dataUuid;

    // send request
    if(request->sendGetRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to get dataset with UUID '" + dataUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief list all data-sets of the user, which are available on shiori
 *
 * @param result reference for response-message
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
listDatasets(std::string &result,
             Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/data_set/all";

    // send request
    if(request->sendGetRequest(result, path, "", error) == false)
    {
        error.addMeesage("Failed to list datasets");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief delete a data-set from shiori
 *
 * @param result reference for response-message
 * @param dataUuid uuid of the data-set to delete
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
deleteDataset(std::string &result,
              const std::string &dataUuid,
              Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/data_set";
    const std::string vars = "uuid=" + dataUuid;

    // send request
    if(request->sendDeleteRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to delete dataset with UUID '" + dataUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief check progress of file-upload
 *
 * @param result reference for response-message
 * @param dataUuid uuid of the data-set to get
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getDatasetProgress(std::string &result,
                   const std::string &dataUuid,
                   Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/v1/data_set/progress";
    const std::string vars = "uuid=" + dataUuid;

    // send request
    if(request->sendGetRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to check upload-state of dataset with UUID '" + dataUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

} // namespace HanamiAI
