/**
 * @file       session.cpp
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

#include <abstract_socket.h>
#include <hanami_common/logger.h>
#include <hanami_network/session.h>
#include <message_definitions.h>
#include <messages_processing/heartbeat_processing.h>
#include <messages_processing/multiblock_data_processing.h>
#include <messages_processing/session_processing.h>
#include <messages_processing/singleblock_data_processing.h>
#include <messages_processing/stream_data_processing.h>
#include <multiblock_io.h>

enum statemachineItems {
    NOT_CONNECTED = 1,
    CONNECTED = 2,
    SESSION_NOT_READY = 3,
    SESSION_READY = 4,

    CONNECT = 7,
    DISCONNECT = 8,
    START_SESSION = 9,
    STOP_SESSION = 10,
};

namespace Hanami
{

/**
 * @brief constructor
 *
 * @param socket pointer to socket
 */
Session::Session(AbstractSocket* socket)
{
    m_multiblockIo = new MultiblockIO(this);
    m_socket = socket;

    initStatemachine();
}

/**
 * @brief destructor
 */
Session::~Session()
{
    // release lock, for the case, that the session is still in creating-state
    m_initState = -1;

    SessionHandler::m_sessionHandler->removeSession(m_sessionId);
    ErrorContainer error;
    closeSession(error, false);
    if (m_socket != nullptr) {
        m_socket->scheduleThreadForDeletion();
        m_socket = nullptr;
    }
    delete m_multiblockIo;
}

/**
 * @brief send data as stream
 *
 * @param data data-pointer
 * @param size number of bytes
 * @param error reference for error-output
 * @param replyExpected if true, the other side sends a reply-message to check timeouts
 *
 * @return false if session is NOT ready to send, send failed, or message is too big, else true
 */
bool
Session::sendStreamData(const void* data,
                        const uint64_t size,
                        ErrorContainer& error,
                        const bool replyExpected)
{
    // check size
    if (size > MAX_SINGLE_MESSAGE_SIZE) {
        return false;
    }

    // check statemachine and try to send
    if (m_statemachine.isInState(SESSION_READY)) {
        return send_Data_Stream(this, data, size, replyExpected, error);
    }

    return false;
}

/**
 * @brief send normal message without response
 *
 * @param data data-pointer
 * @param size number of bytes
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
Session::sendNormalMessage(const void* data, const uint64_t size, ErrorContainer& error)
{
    if (m_statemachine.isInState(SESSION_READY)) {
        uint64_t id = 0;

        if (size <= MAX_SINGLE_MESSAGE_SIZE) {
            // send as single-block-message, if small enough
            id = getRandId();
            if (send_Data_SingleBlock(this, id, data, static_cast<uint32_t>(size), error)
                == false) {
                return false;
            }
        } else {
            // if too big for one message, send as multi-block-message
            if (m_multiblockIo->sendOutgoingData(data, size, error) == 0) {
                return false;
            }
        }

        return true;
    }
    return false;
}

/**
 * @brief send a request and blocks until the other side had send a response-message or a timeout
 *        appeared
 *
 * @param data data-pointer
 * @param size number of bytes
 * @param timeout time in seconds in which the response is expected
 * @param error reference for error-output
 *
 * @return content of the response message as data-buffer, or nullptr, if session is not active
 */
DataBuffer*
Session::sendRequest(const void* data,
                     const uint64_t size,
                     const uint64_t timeout,
                     ErrorContainer& error)
{
    if (m_statemachine.isInState(SESSION_READY)) {
        uint64_t id = 0;

        if (size <= MAX_SINGLE_MESSAGE_SIZE) {
            // send as single-block-message, if small enough
            id = getRandId();
            if (send_Data_SingleBlock(this, id, data, static_cast<uint32_t>(size), error)
                == false) {
                return nullptr;
            }
        } else {
            // if too big for one message, send as multi-block-message
            id = m_multiblockIo->sendOutgoingData(data, size, error);
            if (id == 0) {
                return nullptr;
            }
        }

        return SessionHandler::m_blockerHandler->blockMessage(id, timeout, this);
    }

    return nullptr;
}

/**
 * @brief send response message as reponse for another requst
 *
 * @param data data-pointer
 * @param size number of bytes
 * @param blockerId id to identify the response and map them to the related request
 * @param error reference for error-output
 *
 * @return multiblock-id, or 0, if session is not active
 */
uint64_t
Session::sendResponse(const void* data,
                      const uint64_t size,
                      const uint64_t blockerId,
                      ErrorContainer& error)
{
    if (m_statemachine.isInState(SESSION_READY)) {
        if (size < MAX_SINGLE_MESSAGE_SIZE) {
            // send as single-block-message, if small enough
            const uint64_t singleblockId = getRandId();
            send_Data_SingleBlock(
                this, singleblockId, data, static_cast<uint32_t>(size), error, blockerId);
            return singleblockId;
        } else {
            // if too big for one message, send as multi-block-message
            return m_multiblockIo->sendOutgoingData(data, size, error, blockerId);
        }
    }

    return 0;
}

/**
 * @brief set callback for stram-message
 */
void
Session::setStreamCallback(void* receiver,
                           void (*processStream)(void*, Session*, const void*, const uint64_t))
{
    m_streamReceiver = receiver;
    m_processStreamData = processStream;
}

/**
 * @brief set callback for requests
 */
void
Session::setRequestCallback(void* receiver,
                            void (*processRequest)(void*, Session*, const uint64_t, DataBuffer*))
{
    m_standaloneReceiver = receiver;
    m_processRequestData = processRequest;
}

/**
 * @brief set callback for errors
 */
void
Session::setErrorCallback(void (*processError)(Session*, const uint8_t, const std::string))
{
    m_processError = processError;
}

/**
 * @brief close the session inclusive multiblock-messages, statemachine, message to the other side
 *        and close the socket
 *
 * @param replyExpected true, to expect a reply-message
 *
 * @return true, if all was successful, else false
 */
bool
Session::closeSession(ErrorContainer& error, const bool replyExpected)
{
    LOG_DEBUG("close session with id " + std::to_string(m_sessionId));
    if (m_statemachine.isInState(SESSION_READY)) {
        SessionHandler::m_replyHandler->removeAllOfSession(m_sessionId);
        m_multiblockIo->removeMultiblockBuffer(0);
        if (replyExpected) {
            return send_Session_Close_Start(this, true, error);
        } else {
            send_Session_Close_Start(this, false, error);
            return endSession(error);
        }
    }

    return false;
}

/**
 * @brief getter for the id of the session
 *
 * @return session-id
 */
uint32_t
Session::sessionId() const
{
    return m_sessionId;
}

/**
 * @brief get maximum stream-message size
 *
 * @return maximum stream-message size
 */
uint32_t
Session::getMaximumSingleSize() const
{
    return MAX_SINGLE_MESSAGE_SIZE;
}

/**
 * @brief check if session is client- or server-side
 *
 * @return true, if session is on client-side, else false
 */
bool
Session::isClientSide() const
{
    if (m_socket == nullptr) {
        return false;
    }
    return m_socket->isClientSide();
}

/**
 * @brief create the network connection of the session
 *
 * @param sessionId id for the session
 * @param sessionIdentifier session-identifier value to identify the session on server-side
 *                          before the first data-message was send
 * @param init true to start the initial message-transfer
 *
 * @return false if session is already init or socker can not be connected, else true
 */
bool
Session::connectiSession(const uint32_t sessionId, ErrorContainer& error)
{
    LOG_DEBUG("CALL session connect: " + std::to_string(m_sessionId));

    // check if already connected
    if (m_statemachine.isInState(NOT_CONNECTED)) {
        if (m_socket == nullptr) {
            return false;
        }

        // connect socket
        if (m_socket->initConnection(error) == false) {
            m_initState = -1;
            return false;
        }

        // git into connected state
        if (m_statemachine.goToNextState(CONNECT) == false) {
            m_initState = -1;
            return false;
        }

        m_sessionId = sessionId;
        m_socket->startThread();

        return true;
    }

    m_initState = -1;

    return false;
}

/**
 * @brief bring the session into ready-state after a successful initial message-transfer
 *
 * @param sessionId final id for the session
 * @param sessionIdentifier session-identifier value to identify the session on server-side
 *                          before the first data-message was send
 *
 * @return false, if session is already in ready-state, else true
 */
bool
Session::makeSessionReady(const uint32_t sessionId,
                          const std::string& sessionIdentifier,
                          ErrorContainer& error)
{
    LOG_DEBUG("CALL make session ready: " + std::to_string(m_sessionId));

    if (m_statemachine.goToNextState(START_SESSION, SESSION_NOT_READY)) {
        m_sessionId = sessionId;
        m_sessionIdentifier = sessionIdentifier;

        m_processCreateSession(this, m_sessionIdentifier);

        // release blocked session on client-side
        m_initState = 1;

        return true;
    }

    m_initState = -1;

    error.addMeesage("Failed to make session ready");

    return false;
}

/**
 * @brief stop the session to prevent it from all data-transfers. Delete the session from the
 *        session-handler and close the socket.
 *
 * @param init true, if the caller of the methods initialize the closing process for the session
 *
 * @return true, if statechange and socket-disconnect were successful, else false
 */
bool
Session::endSession(ErrorContainer& error)
{
    LOG_DEBUG("CALL session close: " + std::to_string(m_sessionId));

    // try to stop the session
    if (m_statemachine.goToNextState(STOP_SESSION)) {
        m_processCloseSession(this, m_sessionIdentifier);
        SessionHandler::m_sessionHandler->removeSession(m_sessionId);
        return disconnectSession(error);
    }

    return false;
}

/**
 * @brief disconnect the socket within the sesson
 *
 * @return true, is state-change of the statemachine and closing sthe socket were successful,
 *         else false
 */
bool
Session::disconnectSession(ErrorContainer& error)
{
    LOG_DEBUG("CALL session disconnect: " + std::to_string(m_sessionId));

    if (m_statemachine.goToNextState(DISCONNECT)) {
        if (m_socket == nullptr) {
            return false;
        }

        if (m_socket->closeSocket() == false) {
            error.addMeesage("Failed to close session");
            return false;
        }

        return true;
    }

    error.addMeesage("Failed to go to DISCONNECT-state for session");

    return false;
}

/**
 * @brief send message over the socket of the session
 *
 * @param session session, where the message should be send
 * @param header reference to the header of the message
 * @param data pointer to the data of the complete data
 * @param size size of the complete data
 *
 * @return true, if successful, else false
 */
bool
Session::sendMessage(const CommonMessageHeader& header,
                     const void* data,
                     const uint64_t size,
                     ErrorContainer& error)
{
    if (m_socket == nullptr) {
        return false;
    }

    if (header.flags & 0x1) {
        SessionHandler::m_replyHandler->addMessage(
            header.type, header.sessionId, header.messageId, this);
    }

    return m_socket->sendMessage(data, size, error);
}

/**
 * @brief send a heartbeat-message
 *
 * @return true, if session is ready, else false
 */
bool
Session::sendHeartbeat()
{
    if (m_socket == nullptr) {
        return false;
    }

    if (m_statemachine.isInState(SESSION_READY)) {
        return send_Heartbeat_Start(this, sessionError);
    }

    return false;
}

/**
 * @brief init the statemachine
 */
void
Session::initStatemachine()
{
    // init states
    assert(m_statemachine.createNewState(NOT_CONNECTED, "not connected"));
    assert(m_statemachine.createNewState(CONNECTED, "connected"));
    assert(m_statemachine.createNewState(SESSION_NOT_READY, "session not ready"));
    assert(m_statemachine.createNewState(SESSION_READY, "session ready"));

    // set child state
    assert(m_statemachine.addChildState(CONNECTED, SESSION_NOT_READY));
    assert(m_statemachine.addChildState(CONNECTED, SESSION_READY));

    // set initial states
    assert(m_statemachine.setInitialChildState(CONNECTED, SESSION_NOT_READY));

    // init transitions
    assert(m_statemachine.addTransition(NOT_CONNECTED, CONNECT, CONNECTED));
    assert(m_statemachine.addTransition(CONNECTED, DISCONNECT, NOT_CONNECTED));
    assert(m_statemachine.addTransition(SESSION_NOT_READY, START_SESSION, SESSION_READY));
    assert(m_statemachine.addTransition(SESSION_READY, STOP_SESSION, SESSION_NOT_READY));
}

/**
 * @brief increase the message-id-counter and return the new id
 *
 * @return new message id
 */
uint32_t
Session::increaseMessageIdCounter()
{
    uint32_t tempId = 0;
    while (m_messageIdCounter_lock.test_and_set(std::memory_order_acquire)) {
        asm("");
    }

    m_messageIdCounter++;
    tempId = m_messageIdCounter;

    m_messageIdCounter_lock.clear(std::memory_order_release);
    return tempId;
}

/**
 * @brief generate a new random 64bit-value, which is not 0
 *
 * @return new 64bit-value
 */
uint64_t
Session::getRandId()
{
    uint64_t newId = 0;

    // 0 is the undefined value and should never be allowed
    while (newId == 0) {
        newId = (static_cast<uint64_t>(rand()) << 32) | static_cast<uint64_t>(rand());
    }

    return newId;
}

}  // namespace Hanami
