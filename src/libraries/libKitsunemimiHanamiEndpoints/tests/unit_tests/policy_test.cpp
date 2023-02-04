/**
 * @file        endpoint_test.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2021 Tobias Anker
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

#include "endpoint_test.h"

#include <libKitsunemimiEndpoints/endpoint.h>
#include <libKitsunemimiCommon/common_items/data_items.h>

namespace Kitsunemimi
{
namespace Hanami
{

Endpoint_Test::Endpoint_Test()
    : Kitsunemimi::CompareTestHelper("Endpoint_Test")
{
    parse_test();
    checkUserAgainstEndpoint();
}

/**
 * @brief parse_test
 */
void
Endpoint_Test::parse_test()
{
    const std::string testInput = getTestString();
    Kitsunemimi::Hanami::Endpoint endpoint;
    std::string errorMessage = "";

    TEST_EQUAL(endpoint.parse(testInput, errorMessage), true);
    TEST_EQUAL(endpoint.m_endpointRules->size(), 2);
}

/**
 * @brief checkUserAgainstEndpoint
 */
void
Endpoint_Test::checkUserAgainstEndpoint()
{
    const std::string testInput = getTestString();
    Kitsunemimi::Hanami::Endpoint endpoint;
    std::string errorMessage = "";
    endpoint.parse(testInput, errorMessage);

    TEST_EQUAL(endpoint.checkUserAgainstEndpoint("bogus",      "get_status", "user1"), false);
    TEST_EQUAL(endpoint.checkUserAgainstEndpoint("component2", "bogus",      "user1"), false);
    TEST_EQUAL(endpoint.checkUserAgainstEndpoint("component2", "get_status", "bogus"), false);
    TEST_EQUAL(endpoint.checkUserAgainstEndpoint("component2", "get_status", "user1"), true);
}

/**
 * @brief get string for testing
 */
const std::string
Endpoint_Test::getTestString()
{
    const std::string testString = "component1\n"
                                   "- get_cluster: user1\n"
                                   "- create_cluster: user2\n"
                                   "---\n"
                                   "component2\n"
                                   "- get_status: user1, user2\n";
    return testString;
}

}
}
