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
TableRequestTaskTest::runTest(Hanami::JsonItem &inputData,
                              Hanami::ErrorContainer &error)
{
    // create new user
    std::string result;
    if(HanamiAI::createTask(result,
                            inputData.get("generic_task_name").getString(),
                            "request",
                            inputData.get("cluster_uuid").getString(),
                            inputData.get("base_dataset_uuid").getString(),
                            error) != m_expectSuccess)
    {
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

    inputData.insert("request_task_uuid", jsonItem.get("uuid").getString(), true);

    // wait until task is finished
    do
    {
        sleep(1);
        HanamiAI::getTask(result,
                          inputData.get("request_task_uuid").getString(),
                          inputData.get("cluster_uuid").getString(),
                          error);

        // parse output
        if(jsonItem.parse(result, error) == false) {
            return false;
        }
        std::cout<<jsonItem.toString(true)<<std::endl;
    }
    while(jsonItem.get("state").getString() != "finished");

    // get task-result
    //Hanami::getTask(result, m_taskUuid, m_clusterUuid, true, error);

    // parse output
    //if(jsonItem.parse(result, error) == false) {
    //    return false;
    //}
    // std::cout<<jsonItem.toString(true)<<std::endl;

    return true;
}
