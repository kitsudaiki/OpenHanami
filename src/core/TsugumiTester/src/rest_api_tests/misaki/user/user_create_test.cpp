/**
 * @file        user_create_test.cpp
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

#include "user_create_test.h"

#include <libHanamiAiSdk/user.h>

UserCreateTest::UserCreateTest(const bool expectSuccess)
    : TestStep(expectSuccess)
{
    m_testName = "create user";
    if(expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
}

bool
UserCreateTest::runTest(Kitsunemimi::JsonItem &inputData,
                        Kitsunemimi::ErrorContainer &error)
{
    // create new user
    std::string result;
    if(HanamiAI::createUser(result,
                            inputData.get("user_id").getString(),
                            inputData.get("user_name").getString(),
                            inputData.get("password").getString(),
                            inputData.get("is_admin").getBool(),
                            error) != m_expectSuccess)
    {
        return false;
    }

    if(m_expectSuccess == false) {
        return true;
    }

    // parse output
    Kitsunemimi::JsonItem jsonItem;
    if(jsonItem.parse(result, error) == false) {
        return false;
    }

    inputData.insert("user_id", jsonItem.get("id").getString(), true);

    return true;
}
