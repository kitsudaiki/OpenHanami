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

#ifndef AZUKIHEART_CALLBACKS_H
#define AZUKIHEART_CALLBACKS_H

#include <libKitsunemimiSakuraNetwork/session.h>
#include <libKitsunemimiJson/json_item.h>

#include <azuki_root.h>
#include <libKitsunemimiCpu/cpu.h>
#include <libKitsunemimiSakuraHardware/cpu_core.h>
#include <libKitsunemimiSakuraHardware/cpu_package.h>
#include <libKitsunemimiSakuraHardware/cpu_thread.h>
#include <libKitsunemimiSakuraHardware/host.h>

#include <../../libraries/libKitsunemimiHanamiMessages/protobuffers/azuki_messages.proto3.pb.h>
#include <../../libraries/libKitsunemimiHanamiMessages/message_sub_types.h>

void streamDataCallback(void*,
                        Kitsunemimi::Sakura::Session*,
                        const void*,
                        const uint64_t)
{

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
 * @brief handleSetCpuSpeedRequest
 * @param msg
 * @param session
 * @param blockerId
 */
inline void
handleSetCpuSpeedRequest(const SetCpuSpeed_Message &msg)
{
    // TODO: move the setting of the speed correctly to libKitsunemimiSakuraHardware
    Kitsunemimi::ErrorContainer error;
    uint64_t numberCpuThreads = 0;
    uint64_t minimumSpeed = 0;
    uint64_t maximumSpeed = 0;

    if(Kitsunemimi::getNumberOfCpuThreads(numberCpuThreads, error) == false)
    {

        LOG_ERROR(error);
        return;
    }

    if(Kitsunemimi::getMinimumSpeed(minimumSpeed, 0, error) == false)
    {
        LOG_ERROR(error);
        return;
    }
    if(Kitsunemimi::getMaximumSpeed(maximumSpeed, 0, error) == false)
    {
        LOG_ERROR(error);
        return;
    }

    if(msg.type() == SpeedState::MINIMUM_SPEED)
    {
        for(uint64_t i = 0; i < numberCpuThreads; i++)
        {
            Kitsunemimi::setMinimumSpeed(i, minimumSpeed, error);
            Kitsunemimi::setMaximumSpeed(i, minimumSpeed, error);
        }
    }
    else if(msg.type() == SpeedState::MAXIMUM_SPEED)
    {
        for(uint64_t i = 0; i < numberCpuThreads; i++)
        {
            Kitsunemimi::setMinimumSpeed(i, maximumSpeed, error);
            Kitsunemimi::setMaximumSpeed(i, maximumSpeed, error);
        }
    }
    else
    {
        for(uint64_t i = 0; i < numberCpuThreads; i++) {
            Kitsunemimi::resetSpeed(i, error);
        }
    }

    return;
}

/**
 * @brief genericMessageCallback
 * @param session
 * @param data
 * @param dataSize
 * @param blockerId
 */
void
genericCallback(Kitsunemimi::Sakura::Session* session,
                const uint32_t subtype,
                void* data,
                const uint64_t dataSize,
                const uint64_t blockerId)
{
    switch(subtype)
    {
        case AZUKI_SPEED_SET_MESSAGE_TYPE:
            {
                SetCpuSpeed_Message msg;
                if(msg.ParseFromArray(data, dataSize) == false)
                {
                    handleFail("Receive broken set-cpu-speed-message", session, blockerId);
                    return;
                }

                handleSetCpuSpeedRequest(msg);
            }
            break;
        default:
            handleFail("Received unknown generic message", session, blockerId);
            break;
    }

}


#endif // AZUKIHEART_CALLBACKS_H
