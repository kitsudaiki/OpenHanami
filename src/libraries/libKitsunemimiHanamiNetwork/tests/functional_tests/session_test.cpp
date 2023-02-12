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

#include <test_blossom.h>

#include <libKitsunemimiConfig/config_handler.h>

#include <libKitsunemimiHanamiNetwork/blossom.h>
#include <libKitsunemimiSakuraNetwork/session.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>
#include <libKitsunemimiHanamiCommon/structs.h>

#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiCommon/files/text_file.h>

namespace Kitsunemimi::Hanami
{

Kitsunemimi::Hanami::Session_Test* Session_Test::m_instance = nullptr;

/**
 * @brief Session_Test::Session_Test
 */
Session_Test::Session_Test(const std::string &address) :
    Kitsunemimi::CompareTestHelper("Session_Test")
{
    m_address = address;
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
    m_message = "------------------------------------------------------------------------"
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

    ErrorContainer error;
    TestBlossom* newBlossom = new TestBlossom(this);
    HanamiMessaging::getInstance()->addBlossom("test1", "test2", newBlossom);
    HanamiMessaging::getInstance()->addEndpoint("path-test_2/test",
                                               Kitsunemimi::Hanami::GET_TYPE,
                                               Kitsunemimi::Hanami::BLOSSOM_TYPE,
                                               "test1",
                                               "test2");
    Kitsunemimi::writeFile("/tmp/test-config.conf", getTestConfig(), error, true);
}

void genreicMessageCallback(Sakura::Session* session,
                            const uint32_t,
                            void*,
                            const uint64_t,
                            const uint64_t blockerId)
{
    ErrorContainer error;
    session->sendResponse(std::string("poi").c_str(), 3, blockerId, error);
}

void streamDataCallback(void*,
                        Sakura::Session*,
                        const void* data,
                        const uint64_t dataSize)
{
    LOG_DEBUG("TEST: streamDataCallback");
    const std::string recvMsg(static_cast<const char*>(data), dataSize);
    Session_Test::m_instance->compare(recvMsg, Session_Test::m_instance->m_streamMessage);
}

/**
 * @brief runTest
 */
void
Session_Test::runTest()
{
    ErrorContainer error;
    initConfig("/tmp/test-config.conf", error);
    std::vector<std::string> groupNames = {"target"};
    HanamiMessaging* messaging = HanamiMessaging::getInstance();

    m_numberOfTests++;
    TEST_EQUAL(messaging->initialize("client",
                                     groupNames,
                                     Session_Test::m_instance,
                                     &streamDataCallback,
                                     &genreicMessageCallback,
                                     error,
                                     true), true);
    m_numberOfTests++;
    TEST_EQUAL(messaging->initialize("client",
                                     groupNames,
                                     Session_Test::m_instance,
                                     &streamDataCallback,
                                     &genreicMessageCallback,
                                     error,
                                     true), false);


    DataMap inputValues;
    inputValues.insert("input", new DataValue(42));
    inputValues.insert("test_output", new DataValue(""));
    inputValues.insert("token", new DataValue("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
                                              ".eyJleHAiOjE2Mzc2MjMwMDYsImdyb3VwcyI"
                                              "6InRlc3RlciIsImlhdCI6MTYzNzYxOTQwNiw"
                                              "iaXNfYWRtaW4iOiIxIiwibmFtZSI6InRlc3R"
                                              "fdXNlciIsIm5iZiI6MTYzNzYxOTQwNiwicHd"
                                              "faGFzaCI6IjMzMDcyM2M5YWEzNTE4ZGUyNzY"
                                              "2ZGExZWZkMGI5ZWIxNzE0MDI1MDJkOThmMzl"
                                              "kN2Y0NmYwYWVhZmVjNzhkNTAiLCJ1dWlkIjo"
                                              "iNjdiOGRiNmItYmM0NC00YzBmLTg5ZGQtMTYzMGYxY2Y5ZmNhIn0"
                                              ".KmHwboQt6C6K8ISKXGGX3s-JKmA-ON5AUXCwq1x6IGU"));

    ResponseMessage response;
    RequestMessage request;
    HanamiMessagingClient* client = messaging->getOutgoingClient("target");

    request.id = "path-test_2/test";
    request.httpType = GET_TYPE;
    inputValues.remove("test_output");
    request.inputValues = inputValues.toString();
    m_numberOfTests++;
    TEST_EQUAL(client->triggerSakuraFile(response, request, error),  true);


    m_numberOfTests++;
    request.id = "fail";
    TEST_EQUAL(client->triggerSakuraFile(response, request, error), true);
    TEST_EQUAL(response.success, false);

    m_numberOfTests++;
    TEST_EQUAL(response.type, NOT_IMPLEMENTED_RTYPE);

    TEST_EQUAL(client->sendStreamMessage(m_streamMessage.c_str(),
                                         m_streamMessage.size(),
                                         false,
                                         error), true);

    sleep(1);
    m_numberOfTests++;
    TEST_EQUAL(messaging->closeClient("target", error), true);
    sleep(1);

    // check that were no tests silently skipped
    m_numberOfTests++;
    TEST_EQUAL(m_numberOfTests, 9);

    std::cout<<"finish"<<std::endl;
}

/**
 * @brief Session_Test::getTestConfig
 * @return
 */
const std::string
Session_Test::getTestConfig()
{
    const std::string config = "[DEFAULT]\n"
                               "address = \"" + m_address + "\"\n"
                               "port = 12345\n"
                               "endpoints = \"-\"\n"
                               "database = \"-\"\n"
                               "\n"
                               "\n"
                               "[target]\n"
                               "port = 12345\n"
                               "address = \"" + m_address + "\"\n";
    return config;
}

}
