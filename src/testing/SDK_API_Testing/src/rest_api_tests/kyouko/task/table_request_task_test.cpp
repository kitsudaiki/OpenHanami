/**
 * @file        table_request_task_test.cpp
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

#include "table_request_task_test.h"

#include <hanami_sdk/task.h>

TableRequestTaskTest::TableRequestTaskTest(const bool expectSuccess)
      : TestStep(expectSuccess)
{
    m_testName = "graph-request-task";
    if(expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
}

bool
TableRequestTaskTest::runTest(json &inputData,
                              Hanami::ErrorContainer &error)
{
    // create new user
    std::string result;
    if(Hanami::createTask(result,
                            inputData["generic_task_name"],
                            "request",
                            inputData["cluster_uuid"],
                            inputData["base_dataset_uuid"],
                            error) != m_expectSuccess)
    {
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

    inputData["request_task_uuid"] = jsonItem["uuid"];

    // wait until task is finished
    do
    {
        sleep(1);
        Hanami::getTask(result,
                        inputData["request_task_uuid"],
                        inputData["cluster_uuid"],
                        error);

        // parse output
        jsonItem = json::parse(result, nullptr, false);
        if (jsonItem.is_discarded())
        {
            std::cerr << "parse error" << std::endl;
            return false;
        }
        std::cout<<jsonItem.dump(4)<<std::endl;
    }
    while(jsonItem["state"] != "finished");

    // get task-result
    //Hanami::getTask(result, m_taskUuid, m_clusterUuid, true, error);

    // parse output
    //if(jsonItem.parse(result, error) == false) {
    //    return false;
    //}
    // std::cout<<jsonItem.dump(4)<<std::endl;

    return true;
}
