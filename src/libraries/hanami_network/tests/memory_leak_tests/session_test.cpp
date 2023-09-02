/**
 * @file       session_test.cpp
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

#include "session_test.h"

namespace Kitsunemimi::Sakura
{

Kitsunemimi::Sakura::Session_Test* Session_Test::m_instance = nullptr;

/**
 * @brief streamDataCallback
 * @param target
 * @param data
 * @param dataSize
 */
void streamDataCallback(void*,
                        Session*,
                        const void*,
                        const uint64_t)
{
    LOG_DEBUG("TEST: streamDataCallback");
}

/**
 * @brief standaloneDataCallback
 * @param target
 * @param data
 * @param dataSize
 */
void standaloneDataCallback(void*,
                            Session* session,
                            const uint64_t blockerId,
                            DataBuffer* data)
{
    std::string receivedMessage(static_cast<const char*>(data->data), data->usedBufferSize);
    const std::string rep = receivedMessage + "_response";
    delete data;
    session->sendResponse(rep.c_str(), rep.size(), blockerId, session->sessionError);
}

/**
 * @brief errorCallback
 */
void errorCallback(Kitsunemimi::Sakura::Session*,
                   const uint8_t,
                   const std::string message)
{
    std::cout<<"ERROR: "<<message<<std::endl;
}

/**
 * @brief sessionCreateCallback
 * @param session
 * @param sessionIdentifier
 */
void sessionCreateCallback(Kitsunemimi::Sakura::Session* session,
                           const std::string )
{
    session->setStreamCallback(Session_Test::m_instance, &streamDataCallback);
    session->setRequestCallback(Session_Test::m_instance, &standaloneDataCallback);
    if(session->isClientSide() == false) {
        Session_Test::m_instance->m_serverSession = session;
    }
}

void sessionCloseCallback(Kitsunemimi::Sakura::Session*,
                          const std::string)
{
    Session_Test::m_instance->m_numberOfEndSessions++;
}

/**
 * @brief Session_Test::Session_Test
 */
Session_Test::Session_Test()
    : Kitsunemimi::MemoryLeakTestHelpter("Session_Test")
{
    Session_Test::m_instance = this;

    testController();
    testSession();
    testSend();
}

/**
 * @brief initTestCase
 */
void
Session_Test::initTestCase()
{
    m_staticMessage = "hello!!! (static)";
    m_dynamicMessage = "hello!!! (dynamic)";
    m_singleBlockMessage ="------------------------------------------------------------------------"
                          "-------------------------------------#----------------------------------"
                          "------------------------------------------------------------------------"
                          "---#--------------------------------------------------------------------"
                          "-----------------------------------------#------------------------------"
                          "------------------------------------------------------------------------"
                          "-------#----------------------------------------------------------------"
                          "---------------------------------------------#--------------------------"
                          "------------------------------------------------------------------------"
                          "-----------#------------------------------------------------------------"
                          "-------------------------------------------------#----------------------"
                          "-----#";
    m_multiBlockMessage = "------------------------------------------------------------------------"
                          "-------------------------------------#----------------------------------"
                          "------------------------------------------------------------------------"
                          "---#--------------------------------------------------------------------"
                          "-----------------------------------------#------------------------------"
                          "------------------------------------------------------------------------"
                          "-------#----------------------------------------------------------------"
                          "---------------------------------------------#--------------------------"
                          "------------------------------------------------------------------------"
                          "-----------#------------------------------------------------------------"
                          "-------------------------------------------------#----------------------"
                          "------------------------------------------------------------------------"
                          "-------------------------------------#----------------------------------"
                          "------------------------------------------------------------------------"
                          "---#--------------------------------------------------------------------"
                          "-----------------------------------------#------------------------------"
                          "------------------------------------------------------------------------"
                          "-------#----------------------------------------------------------------"
                          "---------------------------------------------#--------------------------"
                          "------------------------------------------------------------------------"
                          "-----------#------------------------------------------------------------"
                          "-------------------------------------------------#----------------------"
                          "------------------------------------------------------------------------"
                          "-------------------------------------#----------------------------------"
                          "------------------------------------------------------------------------"
                          "---#--------------------------------------------------------------------"
                          "-----------------------------------------#------------------------------"
                          "------------------------------------------------------------------------"
                          "-------#----------------------------------------------------------------"
                          "---------------------------------------------#--------------------------"
                          "------------------------------------------------------------------------"
                          "-----------#------------------------------------------------------------"
                          "-------------------------------------------------#----------------------"
                          "------------------------------------------------------------------------"
                          "-------------------------------------#----------------------------------"
                          "------------------------------------------------------------------------"
                          "---#--------------------------------------------------------------------"
                          "-----------------------------------------#------------------------------"
                          "------------------------------------------------------------------------"
                          "-------#----------------------------------------------------------------"
                          "---------------------------------------------#--------------------------"
                          "------------------------------------------------------------------------"
                          "-----------#------------------------------------------------------------"
                          "-------------------------------------------------#----------------------"
                          "-----#";
}

/**
 * @brief test controller create and delete
 */
void
Session_Test::testController()
{
    // init strings
    initTestCase();
    std::string msg = m_singleBlockMessage;
    SessionController* m_controller = nullptr;
    ErrorContainer* error = nullptr;
    uint32_t id = 0;

    // one-time-allocation
    m_controller = new SessionController(&sessionCreateCallback, &sessionCloseCallback, &errorCallback);
    error = new ErrorContainer();
    id = m_controller->addUnixDomainServer("/tmp/sock.uds", *error);
    m_controller->closeServer(id);
    delete error;
    delete m_controller;
    sleep(1);

    REINIT_TEST();
    m_controller = new SessionController(&sessionCreateCallback, &sessionCloseCallback, &errorCallback);
    error = new ErrorContainer();
    id = m_controller->addUnixDomainServer("/tmp/sock.uds", *error);
    m_controller->closeServer(id);
    delete error;
    delete m_controller;
    CHECK_MEMORY();
}

/**
 * @brief test session create and delete
 */
void
Session_Test::testSession()
{
    // init strings
    initTestCase();
    Session* session = nullptr;
    std::string msg = m_singleBlockMessage;
    SessionController* m_controller = nullptr;
    ErrorContainer* error = nullptr;
    uint32_t id = 0;

    // tests
    m_controller = new SessionController(&sessionCreateCallback, &sessionCloseCallback, &errorCallback);
    error = new ErrorContainer();
    id = m_controller->addUnixDomainServer("/tmp/sock.uds", *error);

    REINIT_TEST();

    session = m_controller->startUnixDomainSession("/tmp/sock.uds", "test", "test", *error);
    session->closeSession(*error, false);
    Session_Test::m_instance->m_serverSession->closeSession(*error, false);
    delete session;
    delete Session_Test::m_instance->m_serverSession;
    sleep(2);

    CHECK_MEMORY();

    m_controller->closeServer(id);
    delete error;
    delete m_controller;
    sleep(2);

}

/**
 * @brief test send messages
 */
void
Session_Test::testSend()
{
    Session* session = nullptr;
    DataBuffer* resp = nullptr;
    std::string msg = m_singleBlockMessage;
    SessionController* m_controller = nullptr;
    ErrorContainer* error = nullptr;
    uint32_t id = 0;

    m_controller = new SessionController(&sessionCreateCallback,
                                         &sessionCloseCallback,
                                         &errorCallback);
    error = new ErrorContainer();
    id = m_controller->addUnixDomainServer("/tmp/sock.uds", *error);

    session = m_controller->startUnixDomainSession("/tmp/sock.uds", "test", "test", *error);

    // first message requires a one-time-allocation
    resp = session->sendRequest(msg.c_str(), msg.size(), 10, *error);
    delete resp;

        REINIT_TEST();
        session->sendStreamData(msg.c_str(), msg.size(), *error,  true);
        usleep(100000);

        // test request with single-block

        // 2x single-block
        msg = m_singleBlockMessage;
        resp = session->sendRequest(msg.c_str(), msg.size(), 10, *error);
        delete resp;
        resp = session->sendRequest(msg.c_str(), msg.size(), 10, *error);
        delete resp;

        // 2x multi-block
        msg = m_multiBlockMessage;
        resp = session->sendRequest(msg.c_str(), msg.size(), 10, *error);
        delete resp;
        resp = session->sendRequest(msg.c_str(), msg.size(), 10, *error);
        delete resp;
        CHECK_MEMORY();

    session->closeSession(*error, false);
    delete session;

    m_controller->closeServer(id);
    delete error;
    delete m_controller;
    sleep(2);
}

}
