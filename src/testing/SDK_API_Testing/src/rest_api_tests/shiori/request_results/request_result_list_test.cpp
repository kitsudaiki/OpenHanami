/**
 * @file        request_result_list_test.cpp
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

#include "request_result_list_test.h"

#include <hanami_sdk/request_result.h>

RequestResultListTest::RequestResultListTest(const bool expectSuccess)
      : TestStep(expectSuccess)
{
    m_testName = "list request-results";
    if(expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
}

bool
RequestResultListTest::runTest(Hanami::JsonItem &inputData,
                               Hanami::ErrorContainer &error)
{
    // list all data
    std::string result;
    if(Hanami::listRequestResult(result, error) != m_expectSuccess) {
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

    inputData.insert("number_of_request_results", static_cast<long>(jsonItem.size()), true);

    return true;
}

