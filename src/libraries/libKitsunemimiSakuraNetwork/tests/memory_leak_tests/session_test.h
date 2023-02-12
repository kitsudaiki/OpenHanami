/**
 * @file       session_test.h
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

#ifndef SESSION_TEST_H
#define SESSION_TEST_H

#include <iostream>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiSakuraNetwork/session_controller.h>
#include <handler/session_handler.h>
#include <libKitsunemimiSakuraNetwork/session.h>
#include <libKitsunemimiNetwork/abstract_socket.h>

#include <libKitsunemimiCommon/test_helper/memory_leak_test_helper.h>

namespace Kitsunemimi::Sakura
{

class Session_Test
        : public Kitsunemimi::MemoryLeakTestHelpter
{
public:
    Session_Test();


    static Session_Test* m_instance;

    std::string m_staticMessage = "";
    std::string m_dynamicMessage = "";
    std::string m_singleBlockMessage = "";
    std::string m_multiBlockMessage = "";

    uint32_t m_numberOfInitSessions = 0;
    uint32_t m_numberOfEndSessions = 0;

    Session* m_serverSession = nullptr;

private:
    void initTestCase();
    void testController();
    void testSession();
    void testSend();

};

}

#endif // SESSION_TEST_H
