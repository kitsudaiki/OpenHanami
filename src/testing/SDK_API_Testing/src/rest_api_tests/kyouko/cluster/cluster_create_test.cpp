/**
 * @file        cluster_create_test.cpp
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

#include "cluster_create_test.h"

#include <hanami_sdk/cluster.h>

ClusterCreateTest::ClusterCreateTest(const bool expectSuccess) : TestStep(expectSuccess)
{
    m_testName = "create cluster";
    if (expectSuccess) {
        m_testName += " (success)";
    }
    else {
        m_testName += " (fail)";
    }
}

bool
ClusterCreateTest::runTest(json& inputData, Hanami::ErrorContainer& error)
{
    // create new cluster
    std::string result;
    if (Hanami::createCluster(
            result, inputData["cluster_name"], inputData["cluster_definition"], error)
        != m_expectSuccess)
    {
        return false;
    }

    if (m_expectSuccess == false) {
        return true;
    }

    // parse output
    json jsonItem;
    try {
        jsonItem = json::parse(result);
    }
    catch (const json::parse_error& ex) {
        error.addMeesage("json-parser error: " + std::string(ex.what()));
        return false;
    }

    inputData["cluster_uuid"] = jsonItem["uuid"];

    return true;
}
