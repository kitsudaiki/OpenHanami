/**
 * @file       session.h
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_SESSION_H
#define KITSUNEMIMI_SAKURA_NETWORK_SESSION_H

#include <iostream>
#include <assert.h>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <hanami_common/logger.h>
#include <hanami_common/statemachine.h>
#include <hanami_common/buffer/data_buffer.h>
#include <hanami_common/buffer/stack_buffer.h>

namespace Hanami
{
struct DataBuffer;
class AbstractSocket;
}

namespace Hanami
{
class SessionHandler;
class SessionController;
class InternalSessionInterface;
class MultiblockIO;
struct CommonMessageHeader;

class Session
{
public:
    ~Session(); 

    // send-messages
    bool sendStreamData(const void* data,
                        const uint64_t size,
                        ErrorContainer &error,
                        const bool replyExpected = false);
    bool sendNormalMessage(const void* data,
                           const uint64_t size,
                           ErrorContainer &error);
    DataBuffer* sendRequest(const void* data,
                            const uint64_t size,
                            const uint64_t timeout,
                            ErrorContainer &error);
    uint64_t sendResponse(const void* data,
                          const uint64_t size,
                          const uint64_t blockerId,
                          ErrorContainer &error);

    // setter for changing callbacks
    void setStreamCallback(void* receiver,
                           void (*processStream)(void*, Session*, const void*, const uint64_t));
    void setRequestCallback(void* receiver,
                            void (*processRequest)(void*, Session*, const uint64_t, DataBuffer*));
    void setErrorCallback(void (*processError)(Session*,  const uint8_t, const std::string));

    // session-controlling functions
    bool closeSession(ErrorContainer &error,
                      bool replyExpected = false);
    uint32_t sessionId() const;
    uint32_t getMaximumSingleSize() const;
    bool isClientSide() const;

    enum errorCodes
    {
        UNDEFINED_ERROR = 0,
        FALSE_VERSION = 1,
        UNKNOWN_SESSION = 2,
        INVALID_MESSAGE_SIZE = 3,
        MESSAGE_TIMEOUT = 4,
        MULTIBLOCK_FAILED = 5,
    };

    uint32_t increaseMessageIdCounter();




    //=====================================================================
    // ALL BELOW IS INTERNAL AND SHOULD NEVER BE USED BY EXTERNAL METHODS!
    //=====================================================================
    Session(AbstractSocket* socket);

    Hanami::Statemachine m_statemachine;
    AbstractSocket* m_socket = nullptr;
    MultiblockIO* m_multiblockIo = nullptr;
    uint32_t m_sessionId = 0;
    std::string m_sessionIdentifier = "";
    ErrorContainer sessionError;

    int m_initState = 0;

    // init session
    bool connectiSession(const uint32_t sessionId,
                         ErrorContainer &error);
    bool makeSessionReady(const uint32_t sessionId,
                          const std::string &sessionIdentifier,
                          ErrorContainer &error);

    // end session
    bool endSession(ErrorContainer &error);
    bool disconnectSession(ErrorContainer &error);

    bool sendHeartbeat();
    void initStatemachine();
    uint64_t getRandId();

    template<typename T>
    bool sendMessage(const T &message,
                     ErrorContainer &error)
    {
        return sendMessage(message.commonHeader,  &message, sizeof(message), error);
    }

    bool sendMessage(const CommonMessageHeader &header,
                     const void* data,
                     const uint64_t size,
                     ErrorContainer &error);

    // callbacks
    void (*m_processCreateSession)(Session*, const std::string);
    void (*m_processCloseSession)(Session*, const std::string);
    void (*m_processStreamData)(void*, Session*, const void*, const uint64_t);
    void (*m_processRequestData)(void*, Session*, const uint64_t, DataBuffer*);
    void (*m_processError)(Session*, const uint8_t, const std::string);
    void* m_streamReceiver = nullptr;
    void* m_standaloneReceiver = nullptr;

    // counter
    std::atomic_flag m_messageIdCounter_lock = ATOMIC_FLAG_INIT;
    uint32_t m_messageIdCounter = 0;
};

}

#endif // KITSUNEMIMI_SAKURA_NETWORK_SESSION_H
