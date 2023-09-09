/**
 * @file        request_result_delete_test.cpp
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

#include "request_result_delete_test.h"

#include <hanami_sdk/request_result.h>

RequestResultDeleteTest::RequestResultDeleteTest(const bool expectSuccess)
          : TestStep(expectSuccess)
{
    m_testName = "delete request-result";
    if(expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
}

bool
RequestResultDeleteTest::runTest(json &inputData,
                                 Hanami::ErrorContainer &error)
{
    const std::string uuid = inputData["request_task_uuid"];

    // delete user by name
    std::string result;
    if(Hanami::deleteRequestResult(result, uuid, error) != m_expectSuccess) {
        return false;
    }

    if(m_expectSuccess == false) {
        return true;
    }

    // parse output
    json jsonItem = json::parse(result, nullptr, false);
    if (jsonItem.is_discarded())
    {
        std::cerr << "parse error" << std::endl;
        return false;
    }

    return true;
}

