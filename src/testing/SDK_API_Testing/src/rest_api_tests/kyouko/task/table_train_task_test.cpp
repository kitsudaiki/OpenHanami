/**
 * @file        table_train_task_test.cpp
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

#include "table_train_task_test.h"

#include <libHanamiAiSdk/task.h>

TableTrainTaskTest::TableTrainTaskTest(const bool expectSuccess)
  : TestStep(expectSuccess)
{
    m_testName = "table-train-task";
    if(expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
}

bool
TableTrainTaskTest::runTest(Hanami::JsonItem &inputData,
                            Hanami::ErrorContainer &error)
{
    // create new user
    std::string result;
    if(HanamiAI::createTask(result,
                            inputData.get("generic_task_name").getString(),
                            "train",
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

    inputData.insert("train_task_uuid", jsonItem.get("uuid").getString(), true);

    // wait until task is finished
    do
    {
        sleep(1);

        HanamiAI::getTask(result,
                          inputData.get("train_task_uuid").getString(),
                          inputData.get("cluster_uuid").getString(),
                          error);

        // parse output
        if(jsonItem.parse(result, error) == false) {
            return false;
        }
        std::cout<<jsonItem.toString(true)<<std::endl;
    }
    while(jsonItem.get("state").getString() != "finished");

    return true;
}
