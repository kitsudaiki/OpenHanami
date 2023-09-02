/**
 * @file        io.cpp
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

#include <hanami_sdk/io.h>

#include <hanami_sdk/common/websocket_client.h>

#include <../../hanami_messages/protobuffers/hanami_messages.proto3.pb.h>

namespace HanamiAI
{

/**
 * @brief train single value
 *
 * @param wsClient pointer to websocket-client for data-transfer
 * @param inputValues vector with all input-values
 * @param shouldValues vector with all should-values
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
train(WebsocketClient* wsClient,
      std::vector<float> &inputValues,
      std::vector<float> &shouldValues,
      Hanami::ErrorContainer &error)
{
    return train(wsClient,
                 &inputValues[0],
                 inputValues.size(),
                 &shouldValues[0],
                 shouldValues.size(),
                 error);
}

/**
 * @brief request single value
 *
 * @param wsClient pointer to websocket-client for data-transfer
 * @param inputValues vector with all input-values
 * @param numberOfOutputValues reference for returning number of output-values
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
float*
request(WebsocketClient* wsClient,
        std::vector<float> &inputValues,
        uint64_t &numberOfOutputValues,
        Hanami::ErrorContainer &error)
{
    return request(wsClient,
                   &inputValues[0],
                   inputValues.size(),
                   numberOfOutputValues,
                   error);
}

/**
 * @brief train single value
 *
 * @param wsClient pointer to websocket-client for data-transfer
 * @param inputValues float-pointer to array with input-values for input-segment
 * @param numberOfInputValues number of input-values for input-segment
 * @param shouldValues float-pointer to array with should-values for output-segment
 * @param numberOfShouldValues number of should-values for output-segment
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
train(WebsocketClient* wsClient,
      float* inputValues,
      const uint64_t numberOfInputValues,
      float* shouldValues,
      const uint64_t numberOfShouldValues,
      Hanami::ErrorContainer &error)
{
    uint8_t buffer[96*1024];

    // build input-message
    ClusterIO_Message inputMsg;
    inputMsg.set_segmentname("input");
    inputMsg.set_islast(false);
    inputMsg.set_processtype(ClusterProcessType::TRAIN_TYPE);
    inputMsg.set_datatype(ClusterDataType::INPUT_TYPE);
    inputMsg.set_numberofvalues(numberOfInputValues);

    for(uint64_t i = 0; i < numberOfInputValues; i++) {
        inputMsg.add_values(inputValues[i]);
    }

    // add input-values to message
    const uint64_t inputMsgSize = inputMsg.ByteSizeLong();
    if(inputMsg.SerializeToArray(buffer, inputMsgSize) == false)
    {
        Hanami::ErrorContainer error;
        error.addMeesage("Failed to serialize train-message");
        return false;
    }

    // send input
    if(wsClient->sendMessage(buffer, inputMsgSize, error) == false)
    {
        error.addMeesage("Failed to send input-values");
        LOG_ERROR(error);
        return false;
    }

    // receive bogus-response, which is not further processed and exist only because of issue:
    //     https://github.com/kitsudaiki/KyoukoMind/issues/27
    uint64_t numberOfBytes = 0;
    uint8_t* recvData = wsClient->readMessage(numberOfBytes, error);
    if(recvData == nullptr
            || numberOfBytes == 0)
    {
        error.addMeesage("Got no valid response");
        LOG_ERROR(error);
        return false;
    }
    delete[] recvData;

    // build should-message
    ClusterIO_Message shouldMsg;
    shouldMsg.set_segmentname("output");
    shouldMsg.set_islast(true);
    shouldMsg.set_processtype(ClusterProcessType::TRAIN_TYPE);
    shouldMsg.set_datatype(ClusterDataType::SHOULD_TYPE);
    shouldMsg.set_numberofvalues(numberOfShouldValues);

    for(uint64_t i = 0; i < numberOfShouldValues; i++) {
        shouldMsg.add_values(shouldValues[i]);
    }

    // add should-values to message
    const uint64_t shouldMsgSize = shouldMsg.ByteSizeLong();
    if(shouldMsg.SerializeToArray(buffer, shouldMsgSize) == false)
    {
        Hanami::ErrorContainer error;
        error.addMeesage("Failed to serialize train-message");
        return false;
    }

    // send should
    if(wsClient->sendMessage(buffer, shouldMsgSize, error) == false)
    {
        error.addMeesage("Failed to send should-values");
        LOG_ERROR(error);
        return false;
    }

    // receive response
    numberOfBytes = 0;
    recvData = wsClient->readMessage(numberOfBytes, error);
    if(recvData == nullptr
            || numberOfBytes == 0)
    {
        error.addMeesage("Got no valid response");
        LOG_ERROR(error);
        return false;
    }

    // check end-message
    bool success = true;
    ClusterIO_Message response;
    if(response.ParseFromArray(recvData, numberOfBytes) == false)
    {
        success = false;
        error.addMeesage("Got no valid train-end-message");
        LOG_ERROR(error);
    }

    delete[] recvData;

    return success;
}

/**
 * @brief request single value
 *
 * @param wsClient pointer to websocket-client for data-transfer
 * @param inputValues float-pointer to array with input-values for input-segment
 * @param numberOfInputValues number of input-values for input-segment
 * @param numberOfOutputValues reference for returning number of output-values
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
float*
request(WebsocketClient* wsClient,
        float* inputData,
        const uint64_t numberOfInputValues,
        uint64_t &numberOfOutputValues,
        Hanami::ErrorContainer &error)
{
    uint8_t buffer[96*1024];

    // build message
    ClusterIO_Message inputMsg;
    inputMsg.set_segmentname("input");
    inputMsg.set_islast(true);
    inputMsg.set_processtype(ClusterProcessType::REQUEST_TYPE);
    inputMsg.set_datatype(ClusterDataType::INPUT_TYPE);
    inputMsg.set_numberofvalues(numberOfInputValues);

    for(uint64_t i = 0; i < numberOfInputValues; i++) {
        inputMsg.add_values(inputData[i]);
    }

    // add input-values to message
    const uint64_t inputMsgSize = inputMsg.ByteSizeLong();
    if(inputMsg.SerializeToArray(buffer, inputMsgSize) == false)
    {
        Hanami::ErrorContainer error;
        error.addMeesage("Failed to serialize request-message");
        LOG_ERROR(error);
        return nullptr;
    }

    // send message
    if(wsClient->sendMessage(buffer, inputMsgSize, error) == false)
    {
        error.addMeesage("Failed to send input-values");
        LOG_ERROR(error);
        return nullptr;
    }

    // receive response
    uint64_t numberOfBytes = 0;
    uint8_t* recvData = wsClient->readMessage(numberOfBytes, error);
    if(recvData == nullptr
            || numberOfBytes == 0)
    {
        error.addMeesage("Got no valid request response");
        LOG_ERROR(error);
        return nullptr;
    }

    // read message from response
    ClusterIO_Message response;
    if(response.ParseFromArray(recvData, numberOfBytes) == false)
    {
        delete[] recvData;
        error.addMeesage("Got no valid request response");
        LOG_ERROR(error);
        return nullptr;
    }

    // convert output
    numberOfOutputValues = response.values_size();
    float* result = new float[numberOfOutputValues];
    for(uint64_t i = 0; i < numberOfOutputValues; i++) {
        result[i] = response.values(i);
    }

    delete[] recvData;

    return result;
}

} // namespace HanamiAI
