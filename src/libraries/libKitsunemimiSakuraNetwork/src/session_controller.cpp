/**
 * @file       session_controller.cpp
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

#include <libKitsunemimiSakuraNetwork/session_controller.h>

#include <handler/reply_handler.h>
#include <handler/message_blocker_handler.h>
#include <handler/session_handler.h>
#include <callbacks.h>
#include <messages_processing/session_processing.h>

#include <libKitsunemimiNetwork/tcp/tcp_server.h>
#include <libKitsunemimiNetwork/tcp/tcp_socket.h>
#include <libKitsunemimiNetwork/unix/unix_domain_server.h>
#include <libKitsunemimiNetwork/unix/unix_domain_socket.h>
#include <libKitsunemimiNetwork/tls_tcp/tls_tcp_server.h>
#include <libKitsunemimiNetwork/tls_tcp/tls_tcp_socket.h>
#include <libKitsunemimiNetwork/template_socket.h>
#include <libKitsunemimiNetwork/template_server.h>

#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi::Sakura
{

SessionController* SessionController::m_sessionController = nullptr;

/**
 * @brief constructor
 */
SessionController::SessionController(void (*processCreateSession)(Session*, const std::string),
                                     void (*processCloseSession)(Session*, const std::string),
                                     void (*processError)(Session*,
                                                          const uint8_t,
                                                          const std::string))
{
    m_sessionController = this;

    if(SessionHandler::m_sessionHandler == nullptr)
    {
        SessionHandler::m_sessionHandler = new SessionHandler(processCreateSession,
                                                              processCloseSession,
                                                              processError);
    }
}

/**
 * @brief destructor
 */
SessionController::~SessionController()
{
    cloesAllServers();

    if(SessionHandler::m_sessionHandler != nullptr)
    {
        delete SessionHandler::m_sessionHandler;
        SessionHandler::m_sessionHandler = nullptr;
    }
}

//==================================================================================================

/**
 * @brief add new unix-domain-server
 *
 * @param socketFile file-path for the server
 * @param threadName base-name for server and client threads
 *
 * @return id of the new server if sussessful, else return 0
 */
uint32_t
SessionController::addUnixDomainServer(const std::string &socketFile,
                                       ErrorContainer &error,
                                       const std::string &threadName)
{
    UnixDomainServer udsServer(socketFile);
    TemplateServer<UnixDomainServer>* server;
    server = new TemplateServer<UnixDomainServer>(std::move(udsServer),
                                                                    this,
                                                                    &processConnection_Callback,
                                                                    threadName);

    if(server->initServer(error) == false) {
        return 0;
    }
    server->startThread();

    SessionHandler* sessionHandler = SessionHandler::m_sessionHandler;
    m_serverIdCounter++;
    sessionHandler->lockServerMap();
    sessionHandler->m_servers.insert(std::make_pair(m_serverIdCounter, server));
    sessionHandler->unlockServerMap();

    return m_serverIdCounter;
}

/**
 * @brief add new tcp-server
 *
 * @param port port where the server should listen
 * @param threadName base-name for server and client threads
 *
 * @return id of the new server if sussessful, else return 0
 */
uint32_t
SessionController::addTcpServer(const uint16_t port,
                                ErrorContainer &error,
                                const std::string &threadName)
{
    TcpServer tcpServer(port);
    TemplateServer<TcpServer>* server = nullptr;
    server = new TemplateServer<TcpServer>(std::move(tcpServer),
                                                             this,
                                                             &processConnection_Callback,
                                                             threadName);

    if(server->initServer(error) == false) {
        return 0;
    }
    server->startThread();

    SessionHandler* sessionHandler = SessionHandler::m_sessionHandler;
    m_serverIdCounter++;
    sessionHandler->lockServerMap();
    sessionHandler->m_servers.insert(std::make_pair(m_serverIdCounter, server));
    sessionHandler->unlockServerMap();

    return m_serverIdCounter;
}

/**
 * @brief add new tls-encrypted tcp-server
 *
 * @param port port where the server should listen
 * @param certFile certificate-file for tls-encryption
 * @param keyFile key-file for tls-encryption
 * @param threadName base-name for server and client threads
 *
 * @return id of the new server if sussessful, else return 0
 */
uint32_t
SessionController::addTlsTcpServer(const uint16_t port,
                                   const std::string &certFile,
                                   const std::string &keyFile,
                                   ErrorContainer &error,
                                   const std::string &threadName)
{
    TcpServer tcpServer(port);

    TlsTcpServer tlsTcpServer(std::move(tcpServer),
                                       certFile,
                                       keyFile);
    TemplateServer<TlsTcpServer>* server = nullptr;
    server = new TemplateServer<TlsTcpServer>(std::move(tlsTcpServer),
                                                                this,
                                                                &processConnection_Callback,
                                                                threadName);

    if(server->initServer(error) == false) {
        return 0;
    }
    server->startThread();

    SessionHandler* sessionHandler = SessionHandler::m_sessionHandler;
    m_serverIdCounter++;
    sessionHandler->lockServerMap();
    sessionHandler->m_servers.insert(std::pair<uint32_t, AbstractServer*>(
                                     m_serverIdCounter, server));
    sessionHandler->unlockServerMap();

    return m_serverIdCounter;
}

/**
 * @brief close server
 *
 * @param id id of the server
 *
 * @return false, if id not exist or server can not be closed, else true
 */
bool
SessionController::closeServer(const uint32_t id)
{
    SessionHandler* sessionHandler = SessionHandler::m_sessionHandler;
    sessionHandler->lockServerMap();

    std::map<uint32_t, AbstractServer*>::iterator it;
    it = sessionHandler->m_servers.find(id);

    if(it != sessionHandler->m_servers.end())
    {
        AbstractServer* server = it->second;
        const bool ret = server->closeServer();
        if(ret == false) {
            return false;
        }

        server->scheduleThreadForDeletion();
        sessionHandler->m_servers.erase(it);
        sessionHandler->unlockServerMap();

        return true;
    }

    sessionHandler->unlockServerMap();

    return false;
}

/**
 * @brief SessionController::cloesAllServers
 */
void
SessionController::cloesAllServers()
{
    SessionHandler* sessionHandler = SessionHandler::m_sessionHandler;
    sessionHandler->lockServerMap();

    for(auto const& [id, server] : sessionHandler->m_servers) {
        server->closeServer();
    }

    sessionHandler->unlockServerMap();
}

//==================================================================================================

/**
 * @brief start new unix-domain-socket
 *
 * @param socketFile socket-file-path, where the unix-domain-socket server is listening
 * @param sessionIdentifier additional identifier as help for an upper processing-layer
 *
 * @return true, if session was successfully created and connected, else false
 */
Session*
SessionController::startUnixDomainSession(const std::string &socketFile,
                                          const std::string &sessionIdentifier,
                                          const std::string &threadName,
                                          ErrorContainer &error)
{
    UnixDomainSocket udsSocket(socketFile);
    TemplateSocket<UnixDomainSocket>* unixDomainSocket = nullptr;
    unixDomainSocket = new TemplateSocket<UnixDomainSocket>(std::move(udsSocket),
                                                                              threadName);

    return startSession(unixDomainSocket, sessionIdentifier, error);
}

/**
 * @brief start new tcp-session
 *
 * @param address ip-address of the server
 * @param port port where the server is listening
 * @param sessionIdentifier additional identifier as help for an upper processing-layer
 *
 * @return true, if session was successfully created and connected, else false
 */
Session*
SessionController::startTcpSession(const std::string &address,
                                   const uint16_t port,
                                   const std::string &sessionIdentifier,
                                   const std::string &threadName,
                                   ErrorContainer &error)
{
    TcpSocket tcpSocket(address, port);
    TemplateSocket<TcpSocket>* tcpTemplateSocket = nullptr;
    tcpTemplateSocket = new TemplateSocket<TcpSocket>(std::move(tcpSocket),
                                                                        threadName);
    return startSession(tcpTemplateSocket, sessionIdentifier, error);
}

/**
 * @brief start new tls-tcp-session
 *
 * @param address ip-address of the server
 * @param port port where the server is listening
 * @param certFile path to the certificate-file
 * @param keyFile path to the key-file
 * @param sessionIdentifier additional identifier as help for an upper processing-layer
 *
 * @return true, if session was successfully created and connected, else false
 */
Session*
SessionController::startTlsTcpSession(const std::string &address,
                                      const uint16_t port,
                                      const std::string &certFile,
                                      const std::string &keyFile,
                                      const std::string &sessionIdentifier,
                                      const std::string &threadName,
                                      ErrorContainer &error)
{
    TcpSocket tcpSocket(address, port);
    TlsTcpSocket tlsTcpSocket(std::move(tcpSocket),
                                       certFile,
                                       keyFile);
    TemplateSocket<TlsTcpSocket>* tlsTcpTemplSocket = nullptr;
    tlsTcpTemplSocket = new TemplateSocket<TlsTcpSocket>(std::move(tlsTcpSocket),
                                                                           threadName);
    return startSession(tlsTcpTemplSocket, sessionIdentifier, error);
}

/**
 * @brief start a new session
 *
 * @param socket socket of the new session
 * @param sessionIdentifier additional identifier as help for an upper processing-layer
 *
 * @return true, if session was successfully created and connected, else false
 */
Session*
SessionController::startSession(AbstractSocket* socket,
                                const std::string &sessionIdentifier,
                                ErrorContainer &error)
{
    // precheck
    if(sessionIdentifier.size() > 64000)
    {
        delete socket;
        return nullptr;
    }

    // create new session
    Session* newSession = new Session(socket);
    const uint32_t newId = SessionHandler::m_sessionHandler->increaseSessionIdCounter();
    socket->setMessageCallback(newSession, &processMessage_callback);

    // connect session
    if(newSession->connectiSession(newId, error))
    {
        SessionHandler::m_sessionHandler->addSession(newId, newSession);
        send_Session_Init_Start(newSession, sessionIdentifier, error);

        while(newSession->m_initState == 0) {
            usleep(10000);
        }

        if(newSession->m_initState == -1)
        {
            newSession->closeSession(error);
            sleep(1);
            delete newSession;

            return nullptr;
        }

        return newSession;
    }

    newSession->closeSession(error);
    sleep(1);
    delete newSession;

    return nullptr;
}

//==================================================================================================

}
