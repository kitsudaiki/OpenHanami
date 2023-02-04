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

#include <libKitsunemimiHanamiEndpoints/endpoint.h>

namespace Kitsunemimi
{
namespace Hanami
{

Endpoint_Test::Endpoint_Test()
    : Kitsunemimi::CompareTestHelper("Endpoint_Test")
{
    parse_test();
    mapEndpoint_test();
}

/**
 * @brief parse_test
 */
void
Endpoint_Test::parse_test()
{
    Endpoint endpoint;
    ErrorContainer error;
    std::string testString = getTestInput();

    // check valid string
    TEST_EQUAL(endpoint.parse(testString, error), true);
    TEST_EQUAL(endpoint.parse(testString, error), true);

    // check invalid string
    // replace first "-" by "+"
    testString[6] = '+';
    TEST_EQUAL(endpoint.parse(testString, error), false);
}

/**
 * @brief mapEndpoint_test
 */
void
Endpoint_Test::mapEndpoint_test()
{
    EndpointEntry result;
    Endpoint endpoint;
    ErrorContainer error;
    endpoint.parse(getTestInput(), error);

    // get existing
    TEST_EQUAL(endpoint.mapEndpoint(result, "path-test_2/test", HttpRequestType::POST_TYPE), true);
    TEST_EQUAL(result.type, SakuraObjectType::TREE_TYPE);
    TEST_EQUAL(result.group, "group2");
    TEST_EQUAL(result.name, "test_list2_blossom");

    // get non-existing
    TEST_EQUAL(endpoint.mapEndpoint(result, "path-test_2/test", HttpRequestType::DELETE_TYPE), false);
    TEST_EQUAL(endpoint.mapEndpoint(result, "path-test_2/fail", HttpRequestType::POST_TYPE), false);

    // add new endpoint
    TEST_EQUAL(endpoint.addEndpoint("path-test_2/test",
                                    HttpRequestType::DELETE_TYPE,
                                    SakuraObjectType::TREE_TYPE,
                                    "asdf",
                                    "poi"), true);
    TEST_EQUAL(endpoint.addEndpoint("path-test_2/test",
                                    HttpRequestType::DELETE_TYPE,
                                    SakuraObjectType::TREE_TYPE,
                                    "asdf",
                                    "poi"), false);
    TEST_EQUAL(endpoint.mapEndpoint(result, "path-test_2/test", HttpRequestType::DELETE_TYPE), true);
}

/**
 * @brief Endpoint_Test::getTestInput
 * @return
 */
std::string
Endpoint_Test::getTestInput()
{
    std::string input = "path/test2\n"
                        "- GET  -> blossom : group1 : test_single1_blossom\n"
                        "- POST -> tree : group1 : test_single2_blossom\n"
                        "\n"
                        "path-test_2/test\n"
                        "- GET  -> blossom : test_list1_blossom\n"
                        "- POST -> tree : group2 : test_list2_blossom\n"
                        "\n";
    return input;
}

}
}
