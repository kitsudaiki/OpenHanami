/**
 * @file        protobuf_messages.cpp
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

#include "cluster_io.h"

#include <../../libraries/hanami_messages/protobuffers/hanami_messages.proto3.pb.h>
#include <core/cluster/cluster.h>

/**
 * @brief send output of a cluster as protobuf-message
 *
 * @param cluster cluster, which output-data should send
 */
void
sendClusterOutputMessage(const Cluster* cluster)
{
    if (cluster->msgClient == nullptr) {
        return;
    }

    uint8_t buffer[TRANSFER_SEGMENT_SIZE];

    for (const auto& [name, brick] : cluster->namedBricks) {
        if (brick->isOutputBrick == false) {
            continue;
        }

        // build message
        ClusterIO_Message msg;
        msg.set_islast(false);
        msg.set_brickname(name);
        msg.set_processtype(ClusterProcessType::REQUEST_TYPE);
        msg.set_numberofvalues(cluster->clusterHeader.numberOfOutputs);

        for (uint64_t outputNeuronId = brick->ioBufferPos;
             outputNeuronId < brick->ioBufferPos + cluster->clusterHeader.numberOfOutputs;
             outputNeuronId++)
        {
            msg.add_values(cluster->outputValues[outputNeuronId]);
        }

        // serialize message
        const uint64_t size = msg.ByteSizeLong();
        if (msg.SerializeToArray(buffer, size) == false) {
            Hanami::ErrorContainer error;
            error.addMessage("Failed to serialize request-message");
            return;
        }

        // send message
        HttpWebsocketThread* client = cluster->msgClient;
        Hanami::ErrorContainer error;
        client->sendData(buffer, size);
    }
}

void
sendProtobufGotInputMessage(Cluster* cluster)
{
    if (cluster->msgClient == nullptr) {
        return;
    }

    // build message
    ClusterIO_Message msg;
    msg.set_islast(false);
    msg.set_processtype(ClusterProcessType::TRAIN_TYPE);
    msg.set_numberofvalues(1);
    msg.add_values(0.0);

    // serialize message
    uint8_t buffer[TRANSFER_SEGMENT_SIZE];
    const uint64_t size = msg.ByteSizeLong();
    if (msg.SerializeToArray(buffer, size) == false) {
        Hanami::ErrorContainer error;
        error.addMessage("Failed to serialize request-message");
        LOG_ERROR(error);
        return;
    }

    // send message
    Hanami::ErrorContainer error;
    cluster->msgClient->sendData(buffer, size);
}

/**
 * @brief sendProtobufNormalEndMessage
 * @param cluster
 */
void
sendClusterNormalEndMessage(Cluster* cluster)
{
    if (cluster->msgClient == nullptr) {
        return;
    }

    // TODO: fix end-message, because it was disabled because of problems when closing the socket
    return;

    // build message
    ClusterIO_Message msg;
    msg.set_islast(true);
    msg.set_processtype(ClusterProcessType::REQUEST_TYPE);
    msg.add_values(0.0);
    msg.set_numberofvalues(1);

    // serialize message
    uint8_t buffer[TRANSFER_SEGMENT_SIZE];
    const uint64_t size = msg.ByteSizeLong();
    if (msg.SerializeToArray(buffer, size) == false) {
        Hanami::ErrorContainer error;
        error.addMessage("Failed to serialize request-message");
        return;
    }

    // send message
    Hanami::ErrorContainer error;
    cluster->msgClient->sendData(buffer, size);
}

/**
 * @brief sendProtobufTrainEndMessage
 * @param cluster
 */
void
sendClusterTrainEndMessage(Cluster* cluster)
{
    if (cluster->msgClient == nullptr) {
        return;
    }

    // build message
    ClusterIO_Message msg;
    msg.set_islast(true);
    msg.set_processtype(ClusterProcessType::TRAIN_TYPE);
    msg.add_values(0.0);
    msg.set_numberofvalues(1);

    // serialize message
    uint8_t buffer[TRANSFER_SEGMENT_SIZE];
    const uint64_t size = msg.ByteSizeLong();
    if (msg.SerializeToArray(buffer, size) == false) {
        Hanami::ErrorContainer error;
        error.addMessage("Failed to serialize request-message");
        return;
    }

    // send message
    Hanami::ErrorContainer error;
    cluster->msgClient->sendData(buffer, size);
}

/**
 * @brief process incoming data as protobuf-message
 *
 * @param cluster cluster which receive the data
 * @param data incoming data
 * @param dataSize incoming number of bytes
 *
 * @return false, if message is broken, else true
 */
bool
recvClusterInputMessage(Cluster* cluster, const void* data, const uint64_t dataSize)
{
    // parse incoming data
    ClusterIO_Message msg;
    if (msg.ParseFromArray(data, dataSize) == false) {
        Hanami::ErrorContainer error;
        error.addMessage("Got invalid Protobuf-ClusterIO-Message");
        LOG_ERROR(error);
        return false;
    }

    auto it = cluster->namedBricks.find(msg.brickname());
    if (it == cluster->namedBricks.end()) {
        Hanami::ErrorContainer error;
        error.addMessage("Brick with name '" + msg.brickname() + "' not found for direct-io");
        LOG_ERROR(error);
        return false;
    }

    // fill given data into the target-cluster
    Brick* brick = it->second;
    if (brick->isInputBrick) {
        float* buffer = &cluster->inputValues[brick->ioBufferPos];
        for (uint64_t i = 0; i < msg.numberofvalues(); i++) {
            buffer[i] = msg.values(i);
        }
    }
    if (brick->isOutputBrick) {
        float* buffer = &cluster->expectedValues[brick->ioBufferPos];
        for (uint64_t i = 0; i < msg.numberofvalues(); i++) {
            buffer[i] = msg.values(i);
        }
    }

    if (msg.islast()) {
        // start request
        if (msg.processtype() == ClusterProcessType::REQUEST_TYPE) {
            cluster->mode = ClusterProcessingMode::NORMAL_MODE;
            cluster->startForwardCycle();
        }

        // start train
        if (msg.processtype() == ClusterProcessType::TRAIN_TYPE) {
            cluster->mode = ClusterProcessingMode::TRAIN_FORWARD_MODE;
            cluster->startForwardCycle();
        }
    }
    else {
        sendProtobufGotInputMessage(cluster);
    }

    return true;
}
