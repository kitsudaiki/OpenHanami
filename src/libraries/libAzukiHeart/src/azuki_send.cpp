 /**
 * @file        azuki_send.cpp
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

#include <libAzukiHeart/azuki_send.h>

#include <libKitsunemimiHanamiCommon/component_support.h>
#include <libKitsunemimiHanamiCommon/structs.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>

#include <../../libKitsunemimiHanamiMessages/protobuffers/azuki_messages.proto3.pb.h>
#include <../../libKitsunemimiHanamiMessages/message_sub_types.h>

using Kitsunemimi::Hanami::HanamiMessaging;
using Kitsunemimi::Hanami::HanamiMessagingClient;
using Kitsunemimi::Hanami::SupportedComponents;

namespace Azuki
{

/**
 * @brief send speed-setter-message zu azuki
 *
 * @param client pointer to client, which is connected to azuki
 * @param msg message to send
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
sendSpeedSetMesage(HanamiMessagingClient* client,
                   const SetCpuSpeed_Message &msg,
                   Kitsunemimi::ErrorContainer &error)
{
    uint8_t buffer[1024];
    const uint64_t msgSize = msg.ByteSizeLong();
    if(msg.SerializeToArray(buffer, msgSize) == false)
    {
        error.addMeesage("Failed to serialize speed-set-message for Azuki");
        return false;
    }

    return client->sendGenericMessage(AZUKI_SPEED_SET_MESSAGE_TYPE, buffer, msgSize, error);
}

/**
 * @brief send message to azuki to set cpu-speed to minimum
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
setSpeedToMinimum(Kitsunemimi::ErrorContainer &error)
{
    // get client
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->azukiClient;
    if(client == nullptr) {
        return false;
    }

    // send message
    SetCpuSpeed_Message msg;
    msg.set_type(SpeedState::MINIMUM_SPEED);
    return sendSpeedSetMesage(client, msg, error);
}

/**
 * @brief send message to azuki to set cpu-speed to automatic
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
setSpeedToAutomatic(Kitsunemimi::ErrorContainer &error)
{
    // get client
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->azukiClient;
    if(client == nullptr) {
        return false;
    }

    // send message
    SetCpuSpeed_Message msg;
    msg.set_type(SpeedState::AUTOMATIC_SPEED);
    return sendSpeedSetMesage(client, msg, error);
}

/**
 * @brief send message to azuki to set cpu-speed to maximum
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
setSpeedToMaximum(Kitsunemimi::ErrorContainer &error)
{
    // get client
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->azukiClient;
    if(client == nullptr) {
        return false;
    }

    // send message
    SetCpuSpeed_Message msg;
    msg.set_type(SpeedState::MAXIMUM_SPEED);
    return sendSpeedSetMesage(client, msg, error);
}

}
