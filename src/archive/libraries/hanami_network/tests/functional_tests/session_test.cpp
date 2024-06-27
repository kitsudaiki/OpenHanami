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

namespace Hanami
{

Hanami::Session_Test* Session_Test::m_instance = nullptr;

/**
 * @brief streamDataCallback
 * @param target
 * @param data
 * @param dataSize
 */
void
streamDataCallback(void* target, Session*, const void* data, const uint64_t dataSize)
{
    LOG_DEBUG("TEST: streamDataCallback");
    Session_Test* instance = static_cast<Session_Test*>(target);

    std::string receivedMessage(static_cast<const char*>(data), dataSize);

    bool ret = false;

    if (dataSize == instance->m_staticMessage.size()) {
        ret = true;
        instance->compare(receivedMessage, instance->m_staticMessage);
    }

    if (dataSize == instance->m_dynamicMessage.size()) {
        ret = true;
        instance->compare(receivedMessage, instance->m_dynamicMessage);
    }

    instance->compare(ret, true);
}

/**
 * @brief standaloneDataCallback
 * @param target
 * @param data
 * @param dataSize
 */
void
standaloneDataCallback(void* target, Session* session, const uint64_t blockerId, DataBuffer* data)
{
    std::string receivedMessage(static_cast<const char*>(data->data), data->usedBufferSize);
    Session_Test* instance = static_cast<Session_Test*>(target);
    LOG_DEBUG("TEST: receive request with size: " + std::to_string(receivedMessage.size()));

    if (receivedMessage.size() < 1024) {
        instance->compare(receivedMessage, instance->m_singleBlockMessage);
    }
    else {
        instance->compare(receivedMessage, instance->m_multiBlockMessage);
    }

    const std::string responseMessage = receivedMessage + "_response";
    session->sendResponse(
        responseMessage.c_str(), responseMessage.size(), blockerId, session->sessionError);

    delete data;
}

/**
 * @brief errorCallback
 */
void
errorCallback(Hanami::Session*, const uint8_t, const std::string message)
{
    std::cout << "ERROR: " << message << std::endl;
}

/**
 * @brief sessionCreateCallback
 * @param session
 * @param sessionIdentifier
 */
void
sessionCreateCallback(Hanami::Session* session, const std::string sessionIdentifier)
{
    session->setStreamCallback(Session_Test::m_instance, &streamDataCallback);
    session->setRequestCallback(Session_Test::m_instance, &standaloneDataCallback);

    Session_Test::m_instance->compare(session->sessionId(), (uint32_t)131073);
    Session_Test::m_instance->m_numberOfInitSessions++;
    Session_Test::m_instance->compare(sessionIdentifier, std::string("test"));
    Session_Test::m_instance->m_testSession = session;
}

void
sessionCloseCallback(Hanami::Session*, const std::string)
{
    Session_Test::m_instance->m_numberOfEndSessions++;
}

/**
 * @brief Session_Test::Session_Test
 */
Session_Test::Session_Test() : Hanami::CompareTestHelper("Session_Test")
{
    Session_Test::m_instance = this;

    initTestCase();
    runTest();
}

/**
 * @brief initTestCase
 */
void
Session_Test::initTestCase()
{
    m_staticMessage = "hello!!! (static)";
    m_dynamicMessage = "hello!!! (dynamic)";
    m_singleBlockMessage
        = "------------------------------------------------------------------------"
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
    m_multiBlockMessage
        = "------------------------------------------------------------------------"
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
 * @brief Session_Test::sendTestMessages
 * @param session
 */
void
Session_Test::sendTestMessages(Session* session)
{
    bool ret = false;
    ErrorContainer error;

    // stream-message
    const std::string staticTestString = Session_Test::m_instance->m_staticMessage;
    ret = session->sendStreamData(staticTestString.c_str(), staticTestString.size(), error, true);
    Session_Test::m_instance->compare(ret, true);

    // singleblock-message
}

/**
 * @brief runTest
 */
void
Session_Test::runTest()
{
    ErrorContainer error;

    SessionController* m_controller
        = new SessionController(&sessionCreateCallback, &sessionCloseCallback, &errorCallback);

    TEST_EQUAL(m_controller->addUnixDomainServer("/tmp/sock.uds", error), 1);
    bool isNullptr
        = m_controller->startUnixDomainSession("/tmp/sock.uds", "test", "test", error) == nullptr;
    TEST_EQUAL(isNullptr, false);

    isNullptr = m_testSession == nullptr;
    TEST_EQUAL(isNullptr, false);

    if (isNullptr) {
        return;
    }

    // test stream-message
    sendTestMessages(m_testSession);

    usleep(100000);

    // test normal message with single-block
    bool ret = m_testSession->sendNormalMessage(
        m_singleBlockMessage.c_str(), m_singleBlockMessage.size(), error);
    TEST_EQUAL(ret, true);
    usleep(100000);

    // test request with single-block
    DataBuffer* resp = m_testSession->sendRequest(
        m_singleBlockMessage.c_str(), m_singleBlockMessage.size(), 10, error);
    const std::string expectedReponse1 = m_singleBlockMessage + "_response";
    const std::string response1(static_cast<const char*>(resp->data), resp->usedBufferSize);
    TEST_EQUAL(response1, expectedReponse1);

    // test request with multi-block
    resp = m_testSession->sendRequest(
        m_multiBlockMessage.c_str(), m_multiBlockMessage.size(), 10, error);
    const std::string expectedReponse2 = m_multiBlockMessage + "_response";
    const std::string response2(static_cast<const char*>(resp->data), resp->usedBufferSize);
    TEST_EQUAL(response2, expectedReponse2);

    LOG_DEBUG("TEST: close session again");
    ret = m_testSession->closeSession(error);
    TEST_EQUAL(ret, true);
    LOG_DEBUG("TEST: close session finished");

    usleep(100000);

    TEST_EQUAL(m_numberOfInitSessions, 2);
    TEST_EQUAL(m_numberOfEndSessions, 2);

    delete m_controller;
}

}  // namespace Hanami
