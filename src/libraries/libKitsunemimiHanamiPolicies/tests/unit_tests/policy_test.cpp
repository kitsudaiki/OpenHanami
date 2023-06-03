/**
 * @file        policy_test.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
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

#include "policy_test.h"

#include <libKitsunemimiHanamiPolicies/policy.h>
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi::Hanami
{

Policy_Test::Policy_Test()
    : Kitsunemimi::CompareTestHelper("Policy_Test")
{
    parse_test();
    checkUserAgainstPolicy();
}

/**
 * @brief parse_test
 */
void
Policy_Test::parse_test()
{
    const std::string testInput = getTestString();
    Kitsunemimi::Hanami::Policy policy;
    ErrorContainer error;

    TEST_EQUAL(policy.parse(testInput, error), true);
}

/**
 * @brief checkUserAgainstPolicy
 */
void
Policy_Test::checkUserAgainstPolicy()
{
    const std::string testInput = getTestString();
    Kitsunemimi::Hanami::Policy policy;
    ErrorContainer error;
    policy.parse(testInput, error);

    TEST_EQUAL(policy.checkUserAgainstPolicy("bogus",      GET_TYPE, "user1"), false);
    TEST_EQUAL(policy.checkUserAgainstPolicy("get_status", GET_TYPE, "bogus"), false);
    TEST_EQUAL(policy.checkUserAgainstPolicy("get_status", GET_TYPE, "user1"), true);
}

/**
 * @brief get string for testing
 */
const std::string
Policy_Test::getTestString()
{
    const std::string testString = "test/get_cluster\n"
                                   "- GET: user1\n"
                                   "test/create_cluster \n"
                                   "- GET: user1\n"
                                   "- POST: user2\n"
                                   "\n"
                                   "get_status \n"
                                   "- GET: user1, user2\n";
    return testString;
}

}
