/**
 * @file        request_result_get_test.cpp
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

#include "request_result_get_test.h"

#include <hanami_sdk/request_result.h>

RequestResultGetTest::RequestResultGetTest(const bool expectSuccess,
                                           const std::string &uuidOverride)
    : TestStep(expectSuccess)
{
    m_testName = "get request-result";
    if(expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
    m_uuid = uuidOverride;
}

bool
RequestResultGetTest::runTest(Hanami::JsonItem &inputData,
                              Hanami::ErrorContainer &error)
{
    if(m_uuid == "") {
        m_uuid = inputData.get("request_task_uuid").getString();
    }

    // get user by name
    std::string result;
    if(HanamiAI::getRequestResult(result, m_uuid, error) != m_expectSuccess) {
        return false;
    }

    if(m_expectSuccess == false) {
        return true;
    }

    // parse output
    Hanami::JsonItem jsonItem;
    if(jsonItem.parse(result, error) == false) {
        return false;
    }

    return true;
}

