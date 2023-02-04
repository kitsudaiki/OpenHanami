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
#include <unistd.h>

#include <libKitsunemimiCommon/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{
namespace Hanami
{
class MessagingClient;

class Session_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    Session_Test(const std::string &address);

    void initTestCase();
    void runTest();

    template<typename  T>
    void compare(T isValue, T shouldValue)
    {
        m_numberOfTests++;
        TEST_EQUAL(isValue, shouldValue);
    }

    static Session_Test* m_instance;
    std::string m_message = "";
    const std::string m_streamMessage = "stream-message";
    Kitsunemimi::Hanami::MessagingClient* m_client = nullptr;

    uint32_t m_numberOfTests = 0;

private:
    const std::string getTestConfig();

    std::string m_address = "";
};

} // namespace Hanami
} // namespace Kitsunemimi

#endif // SESSION_TEST_H
