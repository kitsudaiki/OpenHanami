/**
 * @file       session_handler.cpp
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

#include "session_handler.h"

#include <handler/reply_handler.h>
#include <handler/message_blocker_handler.h>
#include <handler/session_handler.h>

#include <hanami_network/session.h>
#include <hanami_network/session_controller.h>

#include <hanami_common/logger.h>
#include <abstract_socket.h>

namespace Hanami
{

// init static variables
ReplyHandler* SessionHandler::m_replyHandler = nullptr;
MessageBlockerHandler* SessionHandler::m_blockerHandler = nullptr;
SessionHandler* SessionHandler::m_sessionHandler = nullptr;

/**
 * @brief constructor
 */
SessionHandler::SessionHandler(void (*processCreateSession)(Session*, const std::string),
                               void (*processCloseSession)(Session*, const std::string),
                               void (*processError)(Session*, const uint8_t, const std::string))
{
    m_processCreateSession = processCreateSession;
    m_processCloseSession = processCloseSession;
    m_processError = processError;


    if(m_replyHandler == nullptr)
    {
        m_replyHandler = new ReplyHandler();
        m_replyHandler->startThread();
    }

    if(m_blockerHandler == nullptr)
    {
        m_blockerHandler = new MessageBlockerHandler();
        m_blockerHandler->startThread();
    }

    // check if messages have the size of a multiple of 8
    assert(sizeof(CommonMessageHeader) % 8 == 0);
    assert(sizeof(CommonMessageFooter) % 8 == 0);
    assert(sizeof(Session_Init_Start_Message) % 8 == 0);
    assert(sizeof(Session_Init_Reply_Message) % 8 == 0);
    assert(sizeof(Session_Close_Start_Message) % 8 == 0);
    assert(sizeof(Session_Close_Reply_Message) % 8 == 0);
    assert(sizeof(Heartbeat_Start_Message) % 8 == 0);
    assert(sizeof(Heartbeat_Reply_Message) % 8 == 0);
    assert(sizeof(Error_FalseVersion_Message) % 8 == 0);
    assert(sizeof(Error_UnknownSession_Message) % 8 == 0);
    assert(sizeof(Error_InvalidMessage_Message) % 8 == 0);
    assert(sizeof(Data_StreamReply_Message) % 8 == 0);
    assert(sizeof(Data_SingleBlockReply_Message) % 8 == 0);
    assert(sizeof(Data_MultiFinish_Message) % 8 == 0);
}

/**
 * @brief destructor
 */
SessionHandler::~SessionHandler()
{
    if(m_replyHandler != nullptr)
    {
        m_replyHandler->scheduleThreadForDeletion();
        m_replyHandler = nullptr;
        sleep(1);
    }
    if(m_blockerHandler != nullptr)
    {
        delete m_blockerHandler;
        m_blockerHandler = nullptr;
    }

    lockServerMap();
    m_servers.clear();
    unlockServerMap();

    lockSessionMap();
    m_sessions.clear();
    unlockSessionMap();
}

/**
 * @brief add a new session the the internal list
 *
 * @param id id of the session, which should be added
 * @param session pointer to the session
 */
void
SessionHandler::addSession(const uint32_t id, Session* session)
{
    session->m_processCreateSession = m_processCreateSession;
    session->m_processCloseSession = m_processCloseSession;
    session->m_processError = m_processError;

    lockSessionMap();
    m_sessions.insert(std::make_pair(id, session));
    unlockSessionMap();
}

/**
 * @brief remove a session from the internal list, but doesn't close the session
 *
 * @param id id of the session, which should be removed
 */
Session*
SessionHandler::removeSession(const uint32_t id)
{
    Session* ret = nullptr;

    lockSessionMap();

    std::map<uint32_t, Session*>::iterator it;
    it = m_sessions.find(id);

    if(it != m_sessions.end())
    {
        ret = it->second;
        m_sessions.erase(it);
    }

    unlockSessionMap();

    return ret;
}

/**
 * @brief increase the internal counter by one and returns the new counter-value
 *
 * @return id for the new session
 */
uint16_t
SessionHandler::increaseSessionIdCounter()
{
    uint16_t tempId = 0;

    while (m_sessionIdCounter_lock.test_and_set(std::memory_order_acquire)) {
        asm("");
    }

    m_sessionIdCounter++;
    tempId = m_sessionIdCounter;

    m_sessionIdCounter_lock.clear(std::memory_order_release);

    return tempId;
}

/**
 * @brief SessionHandler::lockSessionMap
 */
void
SessionHandler::lockSessionMap()
{
    while (m_sessionMap_lock.test_and_set(std::memory_order_acquire)) {
        asm("");
    }
}

/**
 * @brief SessionHandler::unlockSessionMap
 */
void
SessionHandler::unlockSessionMap()
{
    m_sessionMap_lock.clear(std::memory_order_release);
}

/**
 * @brief SessionHandler::lockServerMap
 */
void
SessionHandler::lockServerMap()
{
    while (m_serverMap_lock.test_and_set(std::memory_order_acquire)) {
        asm("");
    }
}

/**
 * @brief SessionHandler::unlockServerMap
 */
void
SessionHandler::unlockServerMap()
{
    m_serverMap_lock.clear(std::memory_order_release);
}

/**
 * @brief send a heartbeat to all registered sessions
 */
void
SessionHandler::sendHeartBeats()
{
    lockSessionMap();

    for(auto const& [id, session] : m_sessions) {
        session->sendHeartbeat();
    }

    unlockSessionMap();
}

}
