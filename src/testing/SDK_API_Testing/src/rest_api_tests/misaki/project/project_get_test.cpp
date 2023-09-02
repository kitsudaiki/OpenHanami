/**
 * @file        project_get_test.cpp
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

#include "project_get_test.h"

#include <libHanamiAiSdk/project.h>

ProjectGetTest::ProjectGetTest(const bool expectSuccess,
                               const std::string &nameOverride)
    : TestStep(expectSuccess)
{
    m_testName = "get project";
    if(expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
    m_name = nameOverride;
}

bool
ProjectGetTest::runTest(Hanami::JsonItem &inputData,
                        Hanami::ErrorContainer &error)
{
    if(m_name == "") {
        m_name = inputData.get("project_id").getString();
    }

    // get user by name
    std::string result;
    if(HanamiAI::getProject(result, m_name, error) != m_expectSuccess) {
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

    inputData.insert("project_id", jsonItem.get("id").getString(), true);

    return true;
}
