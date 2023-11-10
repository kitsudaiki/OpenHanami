/**
 * @file       session_controller.h
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

#ifndef KITSUNEMIMI_SAKURA_NETWORK_SESSION_HANDLER_H
#define KITSUNEMIMI_SAKURA_NETWORK_SESSION_HANDLER_H

#include <hanami_network/session.h>

#include <atomic>
#include <iostream>
#include <map>
#include <vector>

namespace Hanami
{
class AbstractServer;
}

namespace Hanami
{

class SessionController
{
   public:
    SessionController(void (*processCreateSession)(Session*, const std::string),
                      void (*processCloseSession)(Session*, const std::string),
                      void (*processError)(Session*, const uint8_t, const std::string));
    ~SessionController();

    static Hanami::SessionController* m_sessionController;

    // server
    uint32_t addUnixDomainServer(const std::string& socketFile,
                                 ErrorContainer& error,
                                 const std::string& threadName = "UDS");
    uint32_t addTcpServer(const uint16_t port,
                          ErrorContainer& error,
                          const std::string& threadName = "TCP");
    uint32_t addTlsTcpServer(const uint16_t port,
                             const std::string& certFile,
                             const std::string& keyFile,
                             ErrorContainer& error,
                             const std::string& threadName = "TLS_TCP");
    bool closeServer(const uint32_t id);
    void cloesAllServers();

    // session
    Session* startUnixDomainSession(const std::string& socketFile,
                                    const std::string& sessionIdentifier,
                                    const std::string& threadName,
                                    ErrorContainer& error);
    Session* startTcpSession(const std::string& address,
                             const uint16_t port,
                             const std::string& sessionIdentifier,
                             const std::string& threadName,
                             ErrorContainer& error);
    Session* startTlsTcpSession(const std::string& address,
                                const uint16_t port,
                                const std::string& certFile,
                                const std::string& keyFile,
                                const std::string& sessionIdentifier,
                                const std::string& threadName,
                                ErrorContainer& error);

   private:
    uint32_t m_serverIdCounter = 0;

    Session* startSession(AbstractSocket* socket,
                          const std::string& sessionIdentifier,
                          ErrorContainer& error);
};

}  // namespace Hanami

#endif  // KITSUNEMIMI_SAKURA_NETWORK_SESSION_HANDLER_H
