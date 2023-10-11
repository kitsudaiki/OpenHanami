/**
 * @file        user_get_test.cpp
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

#include "user_get_test.h"

#include <hanami_sdk/user.h>

UserGetTest::UserGetTest(const bool expectSuccess, const std::string &nameOverride)
    : TestStep(expectSuccess)
{
    m_testName = "get user";
    if (expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
    m_id = nameOverride;
}

bool
UserGetTest::runTest(json &inputData, Hanami::ErrorContainer &error)
{
    if (m_id == "") {
        m_id = inputData["user_id"];
    }

    // get user by name
    std::string result;
    if (Hanami::getUser(result, m_id, error) != m_expectSuccess) {
        return false;
    }

    if (m_expectSuccess == false) {
        return true;
    }

    // parse output
    json jsonItem;
    try {
        jsonItem = json::parse(result);
    } catch (const json::parse_error &ex) {
        error.addMeesage("json-parser error: " + std::string(ex.what()));
        return false;
    }

    inputData["user_id"] = jsonItem["id"];

    return true;
}
